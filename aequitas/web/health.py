"""System health for the control room: is every part up, how far behind live, how much room is left.

Each check is ok / warn / down with a one-line detail, so the page reads at a glance. Everything is measured
on this machine; nothing is sent anywhere. The GPU is read with nvidia-smi rather than torch, so the web server
never opens a CUDA context of its own."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

WARN_LAG_S = 15     # live analysis more than this behind the camera: warn
WARN_DISK_GB = 10   # free disk below this: warn
WARN_VRAM = 0.92    # GPU memory used above this share: warn


def _check(name: str, status: str, detail: str, **data) -> dict:
    return {"name": name, "status": status, "detail": detail, **data}


def gpu_info() -> dict | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out:
        return None
    name, used, total, util, temp = [x.strip() for x in out.splitlines()[0].split(",")]
    return {"name": name, "memory_used_mb": int(used), "memory_total_mb": int(total),
            "utilization_pct": int(util), "temperature_c": int(temp)}


def _dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.is_dir() else 0


def collect(settings, store, auth, live, started_at: float) -> dict:
    now_ms = int(time.time() * 1000)
    checks = []

    # the vision-language model server
    vlm = {"backend": settings.vlm.backend, "model": settings.vlm.model, "online": False, "ping_ms": None,
           "loaded": []}
    try:
        from aequitas.understand.vlm_client import make_client

        client = make_client(settings.vlm)
        t0 = time.perf_counter()
        vlm["online"] = bool(client.ping())
        vlm["ping_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if vlm["online"] and settings.vlm.backend == "ollama":
            for m in client.client.ps().get("models", []):
                vlm["loaded"].append({"model": m.get("model") or m.get("name"),
                                      "vram_mb": round((m.get("size_vram") or 0) / 1024**2)})
    except Exception as e:  # the page must load even when the model server is down
        vlm["error"] = str(e)[:200]
    if vlm["online"]:
        loaded = ", ".join(m["model"] for m in vlm["loaded"]) or "none loaded yet (loads on first use)"
        checks.append(_check("AI model server", "ok", f"{vlm['backend']} answering in {vlm['ping_ms']} ms · "
                             f"in memory: {loaded}"))
    else:
        checks.append(_check("AI model server", "down", f"{vlm['backend']} at {settings.vlm.host} is not answering"))

    # GPU
    gpu = gpu_info()
    if gpu:
        share = gpu["memory_used_mb"] / max(gpu["memory_total_mb"], 1)
        checks.append(_check("GPU", "warn" if share > WARN_VRAM else "ok",
                             f"{gpu['name']} · {gpu['memory_used_mb'] / 1024:.1f} of {gpu['memory_total_mb'] / 1024:.1f} GB "
                             f"· {gpu['utilization_pct']} % busy · {gpu['temperature_c']} °C"))
    else:
        checks.append(_check("GPU", "warn", "no NVIDIA GPU found: analysis runs on the CPU, much slower"))

    # live CCTV analysis
    last15 = store.window_stats(now_ms - 15 * 60_000)
    lag_s = last15["lag_ms_p50"] / 1000 if last15["lag_ms_p50"] is not None else None
    running = live is not None and live.pipeline is not None
    if live is not None and live.paused:
        checks.append(_check("Live analysis", "warn", "paused from the dashboard switch"))
    elif running:
        loc = live.localities.get(live.active)
        detail = (f"{loc.name if loc else live.active}: {last15['windows']} windows in 15 min"
                  + (f", {lag_s:.1f} s behind live (p50)" if lag_s is not None else ""))
        checks.append(_check("Live analysis", "warn" if (lag_s or 0) > WARN_LAG_S else "ok", detail))
    else:
        checks.append(_check("Live analysis", "ok", "idle: no locality selected"))
    if last15["windows"] and last15["failed"]:
        share = last15["failed"] / last15["windows"]
        checks.append(_check("Model answers", "warn" if share > 0.05 else "ok",
                             f"{last15['failed']} of {last15['windows']} windows without a valid answer in 15 min"))

    # upload queue
    counts = store.upload_counts()
    waiting = counts.get("queued", 0) + counts.get("processing", 0) + counts.get("verifying", 0)
    checks.append(_check("Video queue", "ok", f"{waiting} waiting or running · {counts.get('done', 0)} done · "
                         f"{counts.get('failed', 0)} failed"))

    # audit chain
    chain = auth.verify_chain()
    checks.append(_check("Audit log", "ok" if chain["ok"] else "down",
                         f"chain intact: {chain['checked']} entries verified" if chain["ok"]
                         else f"chain broken at entry {chain['broken_at']}: {chain['reason']}"))

    # storage
    data = Path(settings.data_dir)
    disk = shutil.disk_usage(data if data.exists() else Path.cwd())
    free_gb = disk.free / 1024**3
    db_bytes = sum(Path(str(store.path) + suffix).stat().st_size for suffix in ("", "-wal")
                   if Path(str(store.path) + suffix).is_file()) if str(store.path) != ":memory:" else 0
    checks.append(_check("Storage", "warn" if free_gb < WARN_DISK_GB else "ok",
                         f"{free_gb:.0f} GB free · database {db_bytes / 1024**2:.1f} MB · "
                         f"videos {_dir_bytes(data / 'uploads') / 1024**3:.2f} GB"))

    order = {"down": 2, "warn": 1, "ok": 0}
    worst = max((c["status"] for c in checks), key=order.__getitem__)
    return {
        "status": worst, "checked_at": now_ms, "uptime_s": round(time.time() - started_at),
        "checks": checks,
        "vlm": vlm, "gpu": gpu,
        "pipeline": {**last15, "lag_s_p50": lag_s, "dropped_15min": store.dropped_since(now_ms - 15 * 60_000),
                     "sample_fps": settings.sample_fps, "mode": settings.analysis.mode},
        "audit": chain,
        "storage": {"free_gb": round(free_gb, 1), "db_mb": round(db_bytes / 1024**2, 1)},
    }
