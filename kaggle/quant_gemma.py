# %% [markdown]
# # Aequitas · Gemma 4 E2B quantization: 16-bit vs 8-bit vs 4-bit (Ollama on Kaggle)
#
# The laptop study (results/qwen/README.md) measured Qwen3-VL 2B. This runs the same 72 saved windows through
# **gemma4:e2b** at each precision Ollama publishes, with our real summarise → classify pipeline
# (`scripts/bench_variants.py`, variant `context`, the shipped setting):
#
# | Tag | Scheme | Download |
# |---|---|---|
# | `gemma4:e2b-it-bf16` | 16-bit, the full-precision reference | 10 GB |
# | `gemma4:e2b-it-q8_0` | 8-bit | 6.0 GB |
# | `gemma4:e2b-it-q4_K_M` | 4-bit, what the demo ships (`gemma4:e2b`) | 4.6 GB |
# | `gemma4:e2b-it-qat` | 4-bit, quantization-aware trained by Google | 4.3 GB |
#
# 1. Accelerator **GPU T4 x2**, Internet **On**. Ollama is pinned to one T4 to stay close to a single-GPU setup.
# 2. *Add Input*: your private dataset made from `kaggle/bench_windows.zip` (72 grids + `index.json`).
# 3. **Save Version → Save & Run All**. Download `quant_gemma.zip` → `results/quant_gemma/`.
#
# Times are T4 times, not laptop times; the ranking between precisions is what carries over. Each model is
# deleted after its run (Kaggle disk), and each starts with a warm-up on a random image, so no bench window is
# ever sent twice to the same model (Ollama caches identical images).

# %%
import json
import os
import shutil
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
OUT = WORK / "quant_gemma"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["gemma4:e2b-it-q4_K_M", "gemma4:e2b-it-q8_0", "gemma4:e2b-it-bf16", "gemma4:e2b-it-qat"]
VARIANT = "context"  # the shipped setting for Gemma (analysis.label_context: true)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)


def sh(cmd: str, check: bool = True) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = r.stdout + r.stderr
    with (OUT / "commands.log").open("a") as f:
        f.write(f"\n$ {cmd}\n{out}\n")
    if check and r.returncode:
        raise RuntimeError(f"{cmd} failed:\n{out[-2000:]}")
    return out


# %% [markdown]
# ## Install Ollama and Aequitas, find the windows

# %%
REPO = WORK / "Aequitas"
if not REPO.exists():
    sh(f"git clone --depth 1 https://github.com/sid-zyy/Aequitas.git {REPO}")
sh("pip install -q ollama pydantic pyyaml httpx pillow numpy")
if not shutil.which("ollama"):
    if not shutil.which("zstd"):  # the installer unpacks a .tar.zst; Kaggle's image has no zstd
        sh("(apt-get update -qq && apt-get install -y -qq zstd) || (sudo apt-get update -qq && sudo apt-get install -y -qq zstd)")
    sh("curl -fsSL https://ollama.com/install.sh | sh")
server = subprocess.Popen(["ollama", "serve"], stdout=open(OUT / "ollama.log", "w"), stderr=subprocess.STDOUT,
                          env={**os.environ, "OLLAMA_KEEP_ALIVE": "30m", "CUDA_VISIBLE_DEVICES": "0"})
time.sleep(5)
print(sh("ollama --version"))

WINDOWS = next(p.parent for p in Path("/kaggle/input").rglob("index.json"))
print("windows:", WINDOWS, len(json.loads((WINDOWS / "index.json").read_text())))


# %% [markdown]
# ## Run each precision

# %%
def pull(model: str) -> float:
    t0 = time.time()
    for attempt in range(4):  # ollama.com has returned 503s mid-pull before
        try:
            sh(f"ollama pull {model}")
            return time.time() - t0
        except RuntimeError:
            time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"pull failed: {model}")


def warm_up(model: str) -> None:
    import base64, io, numpy as np, ollama
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8)).save(buf, "JPEG")
    ollama.chat(model=model, messages=[{"role": "user", "content": "Describe this image in one line.",
                                        "images": [base64.b64encode(buf.getvalue()).decode()]}])


summary = []
for model in MODELS:
    tag = model.split(":")[1]
    pull_s = pull(model)
    warm_up(model)
    ps = sh("ollama ps")  # size and GPU/CPU split of the loaded model
    t0 = time.time()
    log = sh(f"cd {REPO} && PYTHONPATH={REPO} {sys.executable} scripts/bench_variants.py --windows {WINDOWS} --variants {VARIANT} "
             f"--model {model} --out {OUT}/rows-{tag}.jsonl")
    summary.append({"model": model, "pull_s": round(pull_s), "bench_s": round(time.time() - t0), "ollama_ps": ps,
                    "table": log[log.index("| variant"):] if "| variant" in log else log[-2000:]})
    print(model, "\n", ps, "\n", summary[-1]["table"], flush=True)
    sh(f"ollama stop {model}", check=False)
    sh(f"ollama rm {model}", check=False)
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))


# %% [markdown]
# ## Table

# %%
ACCEPT = {"fight": {"fighting", "weapon", "harassment", "riot"}, "arrest": {"arrest"}, "fire": {"fire"},
          "normal": {"none"}}
lines = ["| Tag | Loaded size (ollama ps) | Fight / arrest / fire right | Incident windows flagged | "
         "False alarms (normal) | Broken outputs | Time / window p50 · p95 | Total |", "|---|---|---|---|---|---|---|---|"]
for s in summary:
    tag = s["model"].split(":")[1]
    R = [json.loads(l) for l in (OUT / f"rows-{tag}.jsonl").read_text().splitlines() if l.strip()]
    right = " · ".join(f"{sum(r['label'] in ACCEPT[k] for r in R if r['kind'] == k)}/{sum(r['kind'] == k for r in R)}"
                       for k in ("fight", "arrest", "fire"))
    inc = [r for r in R if r["kind"] != "normal"]
    nor = [r for r in R if r["kind"] == "normal"]
    ts = sorted(r["s"] for r in R)
    size = " ".join(s["ollama_ps"].splitlines()[1].split()[2:5]) if len(s["ollama_ps"].splitlines()) > 1 else "?"
    lines.append(f"| `{s['model']}` | {size} | {right} | {sum(r['label'] not in (None, 'none') for r in inc)}/{len(inc)} | "
                 f"{sum(r['label'] not in (None, 'none') for r in nor)}/{len(nor)} | {sum(r['label'] is None for r in R)} | "
                 f"{st.median(ts):.2f} · {ts[int(0.95 * (len(ts) - 1))]:.2f} s | {sum(ts):.0f} s |")
table = "\n".join(lines)
(OUT / "QUANT.md").write_text("# Gemma 4 E2B quantization on a Kaggle T4 (Ollama, 72 saved windows, variant "
                              f"`{VARIANT}`)\n\n{table}\n")
print(table)
shutil.make_archive(str(WORK / "quant_gemma"), "zip", OUT)
