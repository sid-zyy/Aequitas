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
# Quantized checkpoints to keep for running locally (the rest are deleted to save disk). 4-bit ones are
# what fits the 6 GB laptop GPU. Kaggle's /kaggle/working holds 20 GB: these total ~10 GB.
KEEP = ["qwen-w4a16", "smolvlm-w4a16", "gemma-w4a16"]
# Also upload kept checkpoints to private Hugging Face repos <HF_USER>/aequitas-<name> (resumable download
# at home). Needs an HF_TOKEN secret with WRITE access. Empty = only save to /kaggle/working/checkpoints.
HF_USER = ""

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
import base64

import httpx

from aequitas.config import LlmEndpoint
from aequitas.eval.bench import run_bench
from aequitas.eval.classify_eval import load_items, run_eval
from aequitas.taxonomy import DEFAULT_CATEGORIES
from aequitas.understand.prompts import SummaryAnswer, summarize_prompt
from aequitas.understand.vlm_client import OpenAICompatClient, sse_deltas

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


def stream_summary(jpeg: bytes) -> tuple[SummaryAnswer | None, float | None, float]:
    """Stage 1 as the user experiences it: (summary, time to first word s, full response s)."""
    b64 = base64.b64encode(jpeg).decode("ascii")
    body = {"model": "m", "temperature": 0, "max_tokens": 160, "stream": True,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": summarize_prompt(highlighted=False)},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "SummaryAnswer", "schema": SummaryAnswer.model_json_schema()}}}
    t0, first, text = time.perf_counter(), None, ""
    with HTTP.stream("POST", f"http://127.0.0.1:{PORT}/v1/chat/completions", json=body) as r:
        r.raise_for_status()
        for piece in sse_deltas(r.iter_lines()):
            first = first or time.perf_counter()
            text += piece
    full = time.perf_counter() - t0
    try:
        ans = SummaryAnswer.model_validate_json(text)
    except Exception:
        ans = None
    return ans, (round(first - t0, 3) if first else None), round(full, 3)


def secs(ms):
    return round(ms / 1000, 3) if ms is not None else None


def measure(key: str, precision: str, spec: tuple | None, startup_s: int) -> dict:
    row = {"model": key, "precision": precision, "spec": f"{spec[0]}{spec[1]}" if spec else "none",
           "startup_s": startup_s, "gpu_gb": gpu_used_gb()}
    label = f"{key}-{precision}-{row['spec']}"

    # Response times on synthetic inputs (identical requests for every config).
    b = run_bench(f"http://127.0.0.1:{PORT}", "m", ["summarize", "classify", "chat"], BENCH_CONCURRENCY,
                  BENCH_REQUESTS, label=label, out_dir=OUT / "reports")
    for task in ("summarize", "classify", "chat"):
        one = b["results"][task]["1"]
        row[f"{task}_first_word_s"] = secs(one["ttft_ms"].get("p50"))
        row[f"{task}_full_s"] = secs(one["total_ms"].get("p50"))
        row[f"{task}_full_p95_s"] = secs(one["total_ms"].get("p95"))
        row[f"{task}_ok"] = f"{one['ok']}/{one['requests']}"
        row[f"{task}_tokens_per_s"] = round(1000 / one["tpot_ms"]["p50"], 1) if one["tpot_ms"].get("p50") else None
        busy = b["results"][task][str(max(BENCH_CONCURRENCY))]
        row[f"{task}_full_busy_s"] = secs(busy["total_ms"].get("p50"))  # same, with N requests at once
    # One analysis window = summarize + classify; "busy" = 8 cameras/windows in flight at once.
    row["window_s"] = round((row["summarize_full_s"] or 0) + (row["classify_full_s"] or 0), 3)
    row["window_busy_s"] = round((row["summarize_full_busy_s"] or 0) + (row["classify_full_busy_s"] or 0), 3)
    spec_r = b["results"]["chat"]["1"].get("speculation") or b["results"]["summarize"]["1"].get("speculation")
    row["spec_acceptance"] = spec_r["acceptance_rate"] if spec_r else None

    client = OpenAICompatClient(LlmEndpoint(backend="openai", host=f"http://127.0.0.1:{PORT}", model="m",
                                            timeout_s=300))
    ev = run_eval(client, DEV_ITEMS, DEFAULT_CATEGORIES, concurrency=8, label=label, out_dir=OUT / "reports")
    row.update(dev_accuracy=ev["accuracy"], dev_macro_f1=ev["macro_f1"], dev_valid=ev["valid"] / ev["n"],
               dev_incident_recall=ev["incident_recall"])

    if REAL:
        outputs, sims = [], []
        ref = load_outputs(f"{key}-fp16-none")
        for name, gold, jpeg in REAL:
            t0 = time.perf_counter()
            s, first_s, sum_s = stream_summary(jpeg)
            lab, conf, cls_ms = (client.classify_text(s.summary, s.visible_evidence, DEFAULT_CATEGORIES)
                                 if s else (None, None, 0.0))
            out = {"clip": name, "gold": gold, "summary": s.summary if s else None, "label": lab,
                   "confidence": conf, "first_word_s": first_s, "summary_s": sum_s, "label_s": secs(cls_ms),
                   "clip_total_s": round(time.perf_counter() - t0, 3)}
            if name in ref and ref[name] and out["summary"]:
                out["rougeL_vs_fp16"] = rouge_l(out["summary"], ref[name])
                sims.append(out["rougeL_vs_fp16"])
            outputs.append(out)
        save_outputs(label, outputs)
        pos = [o["label"] in INCIDENT for o in outputs]
        gpos = [o["gold"] == "fighting" for o in outputs]
        tp = sum(p and g for p, g in zip(pos, gpos))
        fp = sum(p and not g for p, g in zip(pos, gpos))
        fn = sum(g and not p for p, g in zip(pos, gpos))
        times = sorted(o["clip_total_s"] for o in outputs)
        row.update(real_n=len(outputs), real_accuracy=round(sum(p == g for p, g in zip(pos, gpos)) / len(outputs), 3),
                   real_precision=round(tp / (tp + fp), 3) if tp + fp else None,
                   real_recall=round(tp / (tp + fn), 3) if tp + fn else None,
                   clip_first_word_s=sorted(o["first_word_s"] or 0 for o in outputs)[len(outputs) // 2],
                   clip_total_s=times[len(times) // 2],
                   rougeL_vs_fp16=round(sum(sims) / len(sims), 3) if sims else None)
    return row


def save_outputs(label: str, outputs: list[dict]) -> None:
    d = OUT / "outputs"
    d.mkdir(exist_ok=True)
    (d / f"{label}.jsonl").write_text("\n".join(json.dumps(o) for o in outputs), encoding="utf-8")


def load_outputs(label: str) -> dict:
    f = OUT / "outputs" / f"{label}.jsonl"
    lines = f.read_text(encoding="utf-8").splitlines() if f.exists() else []
    return {o["clip"]: o["summary"] for o in map(json.loads, filter(None, lines))}


def model_gib(path_or_id: str) -> float | None:
    """Weights on disk: the quantized folder, or the original model in the HF cache."""
    p = Path(path_or_id)
    if not p.exists():
        try:
            from huggingface_hub import snapshot_download

            p = Path(snapshot_download(path_or_id, allow_patterns=["*.safetensors"]))
        except Exception:
            return None
    return round(sum(f.stat().st_size for f in p.rglob("*.safetensors")) / 2**30, 2)


def keep_checkpoint(key: str, precision: str, path: str) -> None:
    name = f"{key}-{precision}"
    if name not in KEEP:
        return
    dest = WORK / "checkpoints" / name
    shutil.copytree(path, dest, dirs_exist_ok=True)
    print(f"kept {name} -> {dest}")
    if HF_USER and os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import HfApi

            api, repo = HfApi(token=os.environ["HF_TOKEN"]), f"{HF_USER}/aequitas-{name}"
            api.create_repo(repo, private=True, exist_ok=True)
            api.upload_folder(repo_id=repo, folder_path=str(dest))
            print(f"uploaded to https://huggingface.co/{repo} (private)")
        except Exception as e:
            print(f"HF upload of {name} failed (checkpoint is still in /kaggle/working): {e}")


# Results are appended as each config finishes, so a timeout loses nothing. Re-running the notebook
# (e.g. Save Version after a timeout) skips configs that already succeeded.
RESULTS = OUT / "results.jsonl"
ROWS = [json.loads(x) for x in RESULTS.read_text().splitlines()] if RESULTS.exists() else []
DONE = {(r["model"], r["precision"], r["spec"]) for r in ROWS if "error" not in r}
ROWS = [r for r in ROWS if "error" not in r]  # failed configs get another try
HTTP = httpx.Client(timeout=600)
print(f"resuming: {len(DONE)} configs already done" if DONE else "fresh run")


def record(row: dict) -> None:
    ROWS.append(row)
    with RESULTS.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=1))


def run_config(key: str, model_id: str, precision: str, spec: tuple | None = None) -> None:
    tag = f"{spec[0]}{spec[1]}" if spec else "none"
    if (key, precision, tag) in DONE:
        print(f"skip {key}-{precision}-{tag} (done)")
        return
    try:
        path = model_id if precision == "fp16" else str(quantize(model_id, precision))
        proc, startup = serve(path, key, spec)
        try:
            row = measure(key, precision, spec, startup)
        finally:
            stop(proc)
        row["ckpt_gib"] = model_gib(path)
        record(row)
        if precision != "fp16" and not spec:
            keep_checkpoint(key, precision, path)
    except Exception as e:  # a config that fails is a result too
        record({"model": key, "precision": precision, "spec": tag, "error": str(e)[:500]})


for key, model_id in MODELS.items():
    for precision in PRECISIONS:
        run_config(key, model_id, precision)
    if key == "gemma" and SPEC_EXTRAS:
        ok = [r for r in ROWS if r["model"] == "gemma" and r["spec"] == "none" and "error" not in r]
        if ok:
            best = min(ok, key=lambda r: r["window_busy_s"])
            for spec in SPEC_EXTRAS:
                run_config(key, model_id, best["precision"], spec)
    shutil.rmtree(SCRATCH / "ckpt", ignore_errors=True)  # free disk before the next model
# %% [markdown]
# ## Results: table, chart and the recommended model (decision rule from docs/ABLATION_PLAN.md §4)

# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LAPTOP_GIB = 5.0  # weights must fit the 6 GB RTX 4050 next to the KV cache and the desktop

df = pd.DataFrame(ROWS)
df.to_csv(OUT / "results.csv", index=False)

# The slide table: what a user feels (seconds) next to accuracy and memory.
SHOW = {
    "model": "Model", "precision": "Precision", "spec": "Speculation", "ckpt_gib": "Weights (GiB)",
    "gpu_gb": "GPU mem (GB)", "summarize_first_word_s": "Summary: first word (s)",
    "summarize_full_s": "Summary: full (s)", "classify_full_s": "Label (s)", "window_s": "Window total (s)",
    "window_busy_s": "Window total, 8 at once (s)", "summarize_tokens_per_s": "Tokens/s",
    "chat_first_word_s": "Chat: first word (s)", "chat_full_s": "Chat: full answer (s)",
    "spec_acceptance": "Spec. acceptance", "dev_macro_f1": "Label macro-F1 (dev)",
    "dev_incident_recall": "Incident recall (dev)", "real_accuracy": "Real clips accuracy",
    "clip_total_s": "Real clip: video→label (s)", "rougeL_vs_fp16": "Summary vs FP16 (ROUGE-L)", "error": "Error",
}
table = df[[c for c in SHOW if c in df.columns]].rename(columns=SHOW)
table = table.astype(object).where(table.notna(), "")  # blank, not nan, on the slide


def pick(df: pd.DataFrame):
    """Decision rule, fixed before the run: valid outputs, accuracy close to the model's own 16-bit,
    fits the laptop, then fastest window with 8 in flight; ties -> incident recall."""
    if "error" in df.columns:
        df = df[df["error"].isna()]
    base = df[df["spec"] == "none"].copy()
    if base.empty:
        return None, "no successful configs"
    fp16_f1 = base[base["precision"] == "fp16"].set_index("model")["dev_macro_f1"].to_dict()
    reasons = []
    keep = []
    for _, r in base.iterrows():
        why = []
        if r.get("dev_valid", 1) < 0.95:
            why.append("<95 % valid outputs")
        ref = fp16_f1.get(r["model"])
        if ref is not None and r["dev_macro_f1"] < ref - 0.03:
            why.append(f"F1 {r['dev_macro_f1']:.2f} < 16-bit {ref:.2f} - 0.03")
        if r.get("ckpt_gib") is not None and r["ckpt_gib"] > LAPTOP_GIB:
            why.append(f"{r['ckpt_gib']} GiB does not fit the laptop")
        if why:
            reasons.append(f"- {r['model']} {r['precision']}: out ({'; '.join(why)})")
        else:
            keep.append(r)
    if not keep:
        return None, "\n".join(reasons + ["No config passes every rule: relax LAPTOP_GIB or use CPU offload."])
    best = sorted(keep, key=lambda r: (r["window_busy_s"], -(r.get("dev_incident_recall") or 0)))[0]
    return best, "\n".join(reasons)


best, why = pick(df)
rec = ["## Recommendation", ""]
if best is not None:
    rec.append(f"**{best['model']} {best['precision']}**: window {best['window_s']} s alone, "
               f"{best['window_busy_s']} s with 8 at once; label macro-F1 {best['dev_macro_f1']}; "
               f"weights {best.get('ckpt_gib')} GiB.")
rec += ["", "Excluded:", why or "- none"]

md = ["# Ablation results (Kaggle T4, vLLM)", "",
      "Times are medians in seconds as a user experiences them: *first word* = until the first token "
      "streams back, *full* = until the whole response is back. *8 at once* = 8 requests in flight "
      "(8 cameras). Dev set = 48 hand-written summaries (compares configs; not real-world accuracy). "
      f"Real clips: n={len(REAL)}, fighting vs none. T4 times, not laptop times.", "",
      table.to_markdown(index=False), "", *rec]
(OUT / "ABLATION.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))

ok = df[df["error"].isna()] if "error" in df.columns else df
ok = ok[ok["spec"] == "none"] if len(ok) else ok
if len(ok):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    markers = {"fp16": "o", "w8a8": "s", "w4a16": "^"}
    for (model, prec), g in ok.groupby(["model", "precision"]):
        for ax, x in zip(axes, ["window_s", "window_busy_s"]):
            ax.scatter(g[x], g["dev_macro_f1"], s=40 + 60 * g["gpu_gb"].fillna(1),
                       marker=markers.get(prec, "o"), label=f"{model} {prec}", alpha=0.8)
    for ax, title in zip(axes, ["one window at a time", "8 windows at once (8 cameras)"]):
        ax.set_xlabel("response time per window: summary + label (s)  ← faster")
        ax.set_ylabel("label macro-F1 (dev set)  ↑ better")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Aequitas: model × precision (marker size = GPU memory)")
    fig.tight_layout()
    fig.savefig(OUT / "ablation.png", dpi=150)

    spec = df[(df["model"] == "gemma")] if "gemma" in set(df["model"]) else None
    if spec is not None and "chat_full_s" in spec and len(spec) > 1:
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = [f"{r.precision} {r.spec}" for r in spec.itertuples()]
        ax.bar(labels, spec["chat_full_s"], label="chat: full answer")
        ax.bar(labels, spec["summarize_full_s"], alpha=0.6, label="summary: full")
        ax.set_ylabel("seconds (median)")
        ax.set_title("Gemma 4 E2B: speculative decoding (MTP, n-gram)")
        ax.legend()
        plt.xticks(rotation=30)
        fig.tight_layout()
        fig.savefig(OUT / "speculation.png", dpi=150)

shutil.make_archive(str(WORK / "ablation"), "zip", OUT)
print("Download ablation.zip (results) and, if needed, checkpoints/ from the Output panel.")
