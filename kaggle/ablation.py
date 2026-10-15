# %% [markdown]
# # Aequitas ablation: 3 VLMs × 16 / 8 / 4-bit on vLLM
#
# Plan and decision rule: `docs/ABLATION_PLAN.md`. Cheap on purpose: one Kaggle session on **GPU T4 x2**.
#
# 1. Settings → Accelerator **GPU T4 x2**, Internet **On**.
# 2. Optional: *Add Input* → "Real Life Violence Situations" (for the real-footage column).
# 3. Optional: *Add-ons → Secrets* → `HF_TOKEN`.
# 4. *Run all*. Download `ablation.zip` from the Output panel when done.
#
# T4 = Turing: vLLM runs FP16, INT8 W8A8 and W4A16 there, but not BF16 or FP8. FP8 is measured on the
# RTX 4050 laptop afterwards.

# %%
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

WORK = Path(os.environ.get("AEQUITAS_WORK", "/kaggle/working"))
SCRATCH = Path(os.environ.get("AEQUITAS_SCRATCH", "/tmp/aequitas"))  # big ephemeral disk for checkpoints
OUT = WORK / "ablation"
for p in (SCRATCH, OUT):
    p.mkdir(parents=True, exist_ok=True)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv"],
                     capture_output=True, text=True).stdout)

# %% [markdown]
# ## Configuration
# Cut this down if quota is short: `MODELS = ["gemma"]`, `PRECISIONS = ["fp16", "w4a16"]` (~1 h).

# %%
MODELS = {
    "gemma": "google/gemma-4-E2B-it",
    "qwen": "Qwen/Qwen3-VL-2B-Instruct",
    "smolvlm": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
}
PRECISIONS = ["fp16", "w8a8", "w4a16"]
SPEC_EXTRAS = [("mtp", 1), ("mtp", 2), ("ngram", 4)]  # run on Gemma's best precision only
GEMMA_DRAFTER = "google/gemma-4-E2B-it-assistant"
CALIB_SAMPLES = 128
BENCH_REQUESTS = 16
BENCH_CONCURRENCY = [1, 8]
REAL_CLIPS_PER_CLASS = 10
PORT = 8000
REPO = "https://github.com/sid-zyy/Aequitas.git"

n_runs = len(MODELS) * len(PRECISIONS) + (len(SPEC_EXTRAS) if "gemma" in MODELS else 0)
print(f"{n_runs} serving runs, ~8 min each, plus ~20 min quantization per model "
      f"=> roughly {(n_runs * 8 + len(MODELS) * 20) / 60:.1f} h")

# %% [markdown]
# ## Install: separate environments for vLLM and llm-compressor (they pin different versions)

# %%
def sh(cmd: str, check: bool = True) -> str:
    print("$", cmd)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    tail = (r.stdout + r.stderr)[-2000:]
    if r.returncode and check:
        raise RuntimeError(f"failed ({r.returncode}): {cmd}\n{tail}")
    return tail


try:
    from kaggle_secrets import UserSecretsClient

    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded")
except Exception:
    print("no HF_TOKEN secret (fine: these models are not gated)")

REPO_DIR = SCRATCH / "Aequitas"
if not REPO_DIR.exists():
    sh(f"git clone --depth 1 {REPO} {REPO_DIR}")
sys.path.insert(0, str(REPO_DIR))  # the bench/eval code runs in this kernel (light deps)

VLLM_PY, QUANT_PY = SCRATCH / "venv-vllm/bin/python", SCRATCH / "venv-quant/bin/python"
sh("pip install -q uv")
if not VLLM_PY.exists():
    sh(f"uv venv -q {SCRATCH}/venv-vllm --python 3.12 && uv pip install -q --python {VLLM_PY} vllm")
if not QUANT_PY.exists():
    sh(f"uv venv -q {SCRATCH}/venv-quant --python 3.12 && uv pip install -q --python {QUANT_PY} llmcompressor datasets")
print(sh(f"{VLLM_PY} -c 'import vllm, torch; print(\"vllm\", vllm.__version__, \"torch\", torch.__version__)'"))
VLLM = str(VLLM_PY.parent / "vllm")

# %% [markdown]
# ## Real-footage grids (optional): 2×2 key-frame grids from Violence / NonViolence clips

# %%
import cv2
import numpy as np

from aequitas.config import Zones
from aequitas.ingest.ring_buffer import BufferedFrame
from aequitas.understand.keyframes import encode_jpeg, make_grid


def clip_grid(path: Path, tile: int = 448) -> bytes | None:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
    for k in range(4):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((k + 0.5) * n / 4))
        ok, f = cap.read()
        if ok:
            frames.append(BufferedFrame(int(1000 * k * n / 4 / fps), f, 1.0, {}))
    cap.release()
    return encode_jpeg(make_grid(frames, None, Zones(), tile)) if len(frames) == 4 else None


REAL = []  # (name, gold, jpeg)
for root in Path("/kaggle/input").glob("*"):
    for gold, folder in (("fighting", "Violence"), ("none", "NonViolence")):
        vids = sorted(p for p in root.rglob("*.mp4") if p.parent.name == folder)[:REAL_CLIPS_PER_CLASS]
        for v in vids:
            if (g := clip_grid(v)) is not None:
                REAL.append((v.name, gold, g))
print(f"{len(REAL)} real-footage grids" + ("" if REAL else " (dataset not added: real-clip column skipped)"))

# %% [markdown]
# ## Quantize, serve, measure

# %%
import httpx

from aequitas.config import LlmEndpoint
from aequitas.eval.bench import run_bench
from aequitas.eval.classify_eval import load_items, run_eval
from aequitas.taxonomy import DEFAULT_CATEGORIES
from aequitas.understand.vlm_client import OpenAICompatClient

DEV_ITEMS, _ = load_items(REPO_DIR / "config/classify_eval.yaml")
INCIDENT = {"fighting", "riot", "weapon"}


def quantize(model_id: str, scheme: str) -> Path:
    out = SCRATCH / "ckpt" / f"{model_id.split('/')[-1]}-{scheme}"
    if not (out / "config.json").exists():
        sh(f"{QUANT_PY} {REPO_DIR}/scripts/quantize/quantize.py --scheme {scheme} --model {model_id} "
           f"--out {out} --samples {CALIB_SAMPLES} --max-len 512")
    return out


def gpu_used_gb() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return round(sum(int(x) for x in out.split()) / 1024, 2)


def serve(model_path: str, key: str, spec: tuple | None = None):
    args = [VLLM, "serve", model_path, "--served-model-name", "m", "--port", str(PORT), "--dtype", "half",
            "--max-model-len", "4096", "--gpu-memory-utilization", "0.88",
            "--limit-mm-per-prompt", json.dumps({"image": 1})]
    if key == "gemma":
        args += ["--mm-processor-kwargs", json.dumps({"max_soft_tokens": 280})]
    if spec and spec[0] == "mtp":
        args += ["--speculative-config", json.dumps({"method": "mtp", "model": GEMMA_DRAFTER,
                                                     "num_speculative_tokens": spec[1]})]
    elif spec and spec[0] == "ngram":
        args += ["--speculative-config", json.dumps({"method": "ngram", "num_speculative_tokens": spec[1],
                                                     "prompt_lookup_max": 4})]
    log = open(SCRATCH / "vllm.log", "w")
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    t0 = time.time()
    while time.time() - t0 < 1200:
        if proc.poll() is not None:
            break
        try:
            if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                return proc, round(time.time() - t0)
        except Exception:
            pass
        time.sleep(5)
    stop(proc)
    raise RuntimeError("vLLM did not start:\n" + (SCRATCH / "vllm.log").read_text()[-3000:])


def stop(proc) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(60)
    except Exception:
        pass
    time.sleep(5)


def rouge_l(a: str, b: str) -> float:
    x, y = a.lower().split(), b.lower().split()
    if not x or not y:
        return 0.0
    dp = [[0] * (len(y) + 1) for _ in range(len(x) + 1)]
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            dp[i + 1][j + 1] = dp[i][j] + 1 if xi == yj else max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[-1][-1]
    p, r = lcs / len(y), lcs / len(x)
    return round(2 * p * r / (p + r), 4) if p + r else 0.0


REFERENCE = {}  # model key -> {clip name: fp16 summary}


def measure(key: str, precision: str, spec: tuple | None, startup_s: int) -> dict:
    row = {"model": key, "precision": precision, "spec": f"{spec[0]}{spec[1]}" if spec else "none",
           "startup_s": startup_s, "gpu_gb": gpu_used_gb()}
    label = f"{key}-{precision}-{row['spec']}"
    b = run_bench(f"http://127.0.0.1:{PORT}", "m", ["summarize", "classify"], BENCH_CONCURRENCY,
                  BENCH_REQUESTS, label=label, out_dir=OUT / "reports")
    for task in ("summarize", "classify"):
        for c in BENCH_CONCURRENCY:
            r = b["results"][task][str(c)]
            row[f"{task}_c{c}_p50_ms"] = r["total_ms"].get("p50")
            row[f"{task}_c{c}_ok"] = f"{r['ok']}/{r['requests']}"
        row[f"{task}_tpot_ms"] = b["results"][task]["1"]["tpot_ms"].get("p50")
        row[f"{task}_ttft_ms"] = b["results"][task]["1"]["ttft_ms"].get("p50")
    row["window_ms_c8"] = round((row.get("summarize_c8_p50_ms") or 0) + (row.get("classify_c8_p50_ms") or 0), 1)
    spec_r = b["results"]["summarize"]["1"].get("speculation")
    row["spec_acceptance"] = spec_r["acceptance_rate"] if spec_r else None

    client = OpenAICompatClient(LlmEndpoint(backend="openai", host=f"http://127.0.0.1:{PORT}", model="m",
                                            timeout_s=300))
    ev = run_eval(client, DEV_ITEMS, DEFAULT_CATEGORIES, concurrency=8, label=label, out_dir=OUT / "reports")
    row.update(dev_accuracy=ev["accuracy"], dev_macro_f1=ev["macro_f1"], dev_valid=f"{ev['valid']}/{ev['n']}",
               dev_incident_recall=ev["incident_recall"])

    if REAL:
        tp = fp = fn = correct = 0
        sims = []
        for name, gold, jpeg in REAL:
            s, _ = client.summarize(jpeg, highlighted=False)
            text = s.summary if s else ""
            lab = client.classify_text(text, s.visible_evidence, DEFAULT_CATEGORIES)[0] if s else None
            pos, gpos = lab in INCIDENT, gold == "fighting"
            tp, fp, fn = tp + (pos and gpos), fp + (pos and not gpos), fn + (gpos and not pos)
            correct += pos == gpos
            if precision == "fp16" and not spec:
                REFERENCE.setdefault(key, {})[name] = text
            elif name in REFERENCE.get(key, {}):
                sims.append(rouge_l(text, REFERENCE[key][name]))
        row.update(real_n=len(REAL), real_accuracy=round(correct / len(REAL), 3),
                   real_precision=round(tp / (tp + fp), 3) if tp + fp else None,
                   real_recall=round(tp / (tp + fn), 3) if tp + fn else None,
                   rougeL_vs_fp16=round(sum(sims) / len(sims), 3) if sims else None)
    return row


ROWS = []
RESULTS = OUT / "results.jsonl"


def record(row: dict) -> None:
    ROWS.append(row)
    with RESULTS.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=1))


for key, model_id in MODELS.items():
    for precision in PRECISIONS:
        try:
            path = model_id if precision == "fp16" else str(quantize(model_id, precision))
            size = None if precision == "fp16" else round(
                sum(f.stat().st_size for f in Path(path).glob("*.safetensors")) / 2**30, 2)
            proc, startup = serve(path, key)
            try:
                row = measure(key, precision, None, startup)
                row["ckpt_gib"] = size
                record(row)
            finally:
                stop(proc)
        except Exception as e:  # a config that fails is a result too
            record({"model": key, "precision": precision, "spec": "none", "error": str(e)[:500]})
    if key == "gemma" and SPEC_EXTRAS:
        ok = [r for r in ROWS if r["model"] == "gemma" and "error" not in r]
        if ok:
            best = min(ok, key=lambda r: r["window_ms_c8"])
            path = model_id if best["precision"] == "fp16" else str(SCRATCH / "ckpt" / f"{model_id.split('/')[-1]}-{best['precision']}")
            for spec in SPEC_EXTRAS:
                try:
                    proc, startup = serve(path, key, spec)
                    try:
                        record(measure(key, best["precision"], spec, startup))
                    finally:
                        stop(proc)
                except Exception as e:
                    record({"model": key, "precision": best["precision"], "spec": f"{spec[0]}{spec[1]}",
                            "error": str(e)[:500]})
    shutil.rmtree(SCRATCH / "ckpt", ignore_errors=True)  # free disk before the next model

# %% [markdown]
# ## Table and chart for the slide

# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(ROWS)
df.to_csv(OUT / "results.csv", index=False)
cols = [c for c in ["model", "precision", "spec", "ckpt_gib", "gpu_gb", "summarize_c1_p50_ms", "classify_c1_p50_ms",
                    "window_ms_c8", "summarize_tpot_ms", "spec_acceptance", "dev_macro_f1", "dev_incident_recall",
                    "real_accuracy", "real_recall", "rougeL_vs_fp16", "error"] if c in df.columns]
md = ["# Ablation results (Kaggle T4, vLLM)", "",
      "Development set = 48 hand-written summaries (compares configs; not real-world accuracy). "
      f"Real clips: n={len(REAL)}, fighting vs none. T4 latencies, not laptop latencies.", "",
      df[cols].to_markdown(index=False)]
(OUT / "ABLATION.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))

ok = df[df.get("error").isna()] if "error" in df.columns else df
if len(ok):
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"fp16": "o", "w8a8": "s", "w4a16": "^"}
    for (model, prec), g in ok[ok["spec"] == "none"].groupby(["model", "precision"]):
        ax.scatter(g["window_ms_c8"] / 1000, g["dev_macro_f1"], s=40 + 60 * g["gpu_gb"].fillna(1),
                   marker=markers.get(prec, "o"), label=f"{model} {prec}", alpha=0.8)
    ax.set_xlabel("time per window at concurrency 8, summarize + classify (s)  ← faster")
    ax.set_ylabel("stage-2 macro-F1 (dev set)  ↑ better")
    ax.set_title("Aequitas: model × precision (marker size = GPU memory)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "ablation.png", dpi=150)

shutil.make_archive(str(WORK / "ablation"), "zip", OUT)
print("Download ablation.zip from the Output panel.")
