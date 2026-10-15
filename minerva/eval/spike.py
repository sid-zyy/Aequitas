"""M0 spike: measure detector speed, VLM latency and memory on THIS machine, for each
candidate VLM that is already pulled in Ollama. Writes data/reports/spike-<time>.json."""

from __future__ import annotations

import json
import logging
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import psutil

from minerva.config import Settings, resolve_device
from minerva.eval.report import percentile
from minerva.ingest.ring_buffer import BufferedFrame
from minerva.types import Candidate, Category

log = logging.getLogger(__name__)


def _gpu_info() -> dict:
    info: dict = {}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["vram_total_gb"] = round(p.total_memory / 2**30, 2)
    except ImportError:
        info["torch"] = None
    return info


def _vram_used_gb() -> float | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return round(pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**30, 2)
    except Exception:
        return None


def _ollama_rss_gb() -> float:
    total = 0
    for p in psutil.process_iter(["name", "memory_info"]):
        name = (p.info.get("name") or "").lower()
        if "ollama" in name and p.info.get("memory_info"):
            total += p.info["memory_info"].rss
    return round(total / 2**30, 2)


def _stats(ms: list[float]) -> dict:
    return {"n": len(ms), "p50_ms": round(percentile(ms, .5), 1), "p95_ms": round(percentile(ms, .95), 1),
            "mean_ms": round(statistics.mean(ms), 1)} if ms else {"n": 0}


def _sample_frames(video: str | None, n: int) -> list[np.ndarray]:
    if video:
        cap = cv2.VideoCapture(video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n
        frames = []
        for i in np.linspace(0, max(0, total - 1), n).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, img = cap.read()
            if ok:
                frames.append(img)
        cap.release()
        if frames:
            return frames
    from ultralytics.utils import ASSETS

    img = cv2.imread(str(ASSETS / "bus.jpg"))
    return [img] * n


def run_spike(settings: Settings, models: list[str], video: str | None = None,
              det_iters: int = 30, vlm_iters: int = 3) -> dict:
    from minerva.understand.keyframes import encode_jpeg, make_grid
    from minerva.understand.stages import understand
    from minerva.understand.vlm_client import VlmClient
    from minerva.chat.parser import LlmFilter, _prompt
    from minerva.vision.detector import Detector
    from minerva.vision.pose import PoseEstimator
    from minerva.config import Zones

    device = resolve_device(settings.device)
    report: dict = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "machine": {"platform": platform.platform(), "cpu": platform.processor(),
                    "cpu_cores": psutil.cpu_count(logical=False), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
                    "device": device, **_gpu_info()},
        "settings": {"imgsz": settings.detector.imgsz, "backend": settings.detector.backend,
                     "sample_fps": settings.sample_fps, "tile_px": settings.vlm.tile_px},
    }
    frames = _sample_frames(video, max(det_iters, 4))

    # ---- detector ----
    log.info("detector: %s on %s", settings.detector.weights, device)
    det = Detector(settings.detector, device)
    det(frames[0])  # warm-up
    times, persons = [], 0
    for img in frames[:det_iters]:
        t0 = time.perf_counter()
        d = det(img)
        times.append((time.perf_counter() - t0) * 1000)
        persons = max(persons, int((d.class_id == 0).sum()) if len(d) else 0)
    report["detector"] = {**_stats(times), "max_persons_seen": persons,
                          "max_fps_one_camera": round(1000 / statistics.median(times), 1)}

    # ---- pose on a crop ----
    pose = PoseEstimator(settings.detector.pose_weights, settings.detector.backend, device)
    d = det(frames[0])
    ptimes = []
    if len(d) and (d.class_id == 0).any():
        from minerva.types import TrackObs

        box = tuple(map(float, d.xyxy[d.class_id == 0][0]))
        t = TrackObs(1, "person", 0.9, box)
        pose(frames[0], t)
        for _ in range(10):
            t0 = time.perf_counter()
            pose(frames[0], t)
            ptimes.append((time.perf_counter() - t0) * 1000)
    report["pose_crop"] = _stats(ptimes)

    # ---- keyframe grid ----
    boxes = {}
    if len(d):
        for i, b in enumerate(d.xyxy[:3]):
            boxes[i + 1] = tuple(map(float, b))
    buf = [BufferedFrame(1000 * i, f, 1.0, boxes) for i, f in enumerate(frames[:4])]
    cand = Candidate("spike", Category.FIGHTING, None, 0, 3000, 0.7, tuple(boxes))
    grid_img = make_grid(buf, cand, Zones(), settings.vlm.tile_px)
    grid_jpeg = encode_jpeg(grid_img)

    run_seed = int(time.time() * 1000) % 100_000

    def unique_grid(i: int) -> bytes:
        # Every call, in every run, gets a different image: Ollama caches identical prompts+images,
        # which would make calls look ~10x faster than real events (seen twice in M0).
        img = np.roll(grid_img, shift=17 * (i + 1) + run_seed % 97, axis=1).copy()
        img[0, 0] = (run_seed + i) % 256  # one pixel is enough to change the image hash
        return encode_jpeg(img)
    out_dir = settings.data_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "spike-grid.jpg").write_bytes(grid_jpeg)

    # ---- VLMs ----
    report["vlm"] = {}
    report["memory_before_vlm"] = {"process_rss_gb": round(psutil.Process().memory_info().rss / 2**30, 2),
                                   "vram_used_gb": _vram_used_gb()}
    for model in models:
        client = VlmClient(settings.vlm, model=model)
        if not client.ping():
            report["vlm"][model] = {"skipped": "not pulled in Ollama (run: ollama pull " + model + ")"}
            continue
        log.info("VLM %s: cold call (includes model load)", model)
        t0 = time.perf_counter()
        u = understand(client, client, unique_grid(1000 + len(report["vlm"])), highlighted=True)
        cold_ms = (time.perf_counter() - t0) * 1000
        ans = u.answer
        warm, warm_s, warm_c = [], [], []
        answers, valid = [ans.model_dump() if ans else None], int(ans is not None)
        for i in range(vlm_iters):
            log.info("VLM %s: warm call %d/%d (unique image)", model, i + 1, vlm_iters)
            u = understand(client, client, unique_grid(i), highlighted=True)
            warm.append(u.vlm_ms or 0)
            warm_s.append(u.summarize_ms or 0)
            warm_c.append(u.classify_ms or 0)
            valid += u.answer is not None
            answers.append(u.answer.model_dump() if u.answer else None)
        chat_times = []
        for q in ["Show all tailgating events from camera 2 today", "How many falls this week?"]:
            t0 = time.perf_counter()
            client.structured([{"role": "user", "content": _prompt(q, datetime.now().astimezone(),
                               [{"camera_id": "cam2", "name": "Corridor", "aliases": ["camera 2"]}])}], LlmFilter)
            chat_times.append((time.perf_counter() - t0) * 1000)
        report["vlm"][model] = {
            "cold_ms": round(cold_ms), "warm": _stats(warm), "warm_summarize": _stats(warm_s),
            "warm_classify": _stats(warm_c), "valid_json": f"{valid}/{vlm_iters + 1}",
            "chat_parse": _stats(chat_times), "ollama_rss_gb": _ollama_rss_gb(), "vram_used_gb": _vram_used_gb(),
            "answers": answers,
        }
    path = out_dir / f"spike-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["_path"] = str(path)
    return report


def summarize(report: dict) -> str:
    m = report["machine"]
    lines = [
        f"Machine: {m.get('gpu') or m['cpu']} | device={m['device']} | RAM {m['ram_gb']} GB",
        f"Detector: p50 {report['detector'].get('p50_ms')} ms, p95 {report['detector'].get('p95_ms')} ms "
        f"(~{report['detector'].get('max_fps_one_camera')} fps max)",
        f"Pose crop: p50 {report['pose_crop'].get('p50_ms', '-')} ms",
    ]
    for model, r in report["vlm"].items():
        if "skipped" in r:
            lines.append(f"VLM {model}: skipped - {r['skipped']}")
        else:
            lines.append(f"VLM {model}: cold {r['cold_ms'] / 1000:.1f} s, warm p50 {r['warm']['p50_ms'] / 1000:.1f} s "
                         f"(summarize {r['warm_summarize']['p50_ms'] / 1000:.1f} s + classify "
                         f"{r['warm_classify']['p50_ms'] / 1000:.2f} s), valid JSON {r['valid_json']}, chat parse p50 {r['chat_parse']['p50_ms'] / 1000:.1f} s, "
                         f"Ollama RSS {r['ollama_rss_gb']} GB, VRAM used {r['vram_used_gb']}")
    lines.append(f"Report: {report.get('_path')}")
    return "\n".join(lines)
