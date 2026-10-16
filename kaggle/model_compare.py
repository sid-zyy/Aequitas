# %% [markdown]
# # Aequitas · which model detects incidents best? (Ollama on Kaggle, SCVD)
#
# Runs our real pipeline (summarize → classify, `aequitas.eval.dataset`) with several open models on the
# **Smart-City CCTV Violence Detection (SCVD)** dataset, the closest public match to the faculty's clips.
# Same backend as the demo laptop (Ollama), so the ranking carries over.
#
# 1. Accelerator **GPU T4 x2**, Internet **On**.
# 2. *Add Input*: **"Smart-City CCTV Violence Detection Dataset (SCVD)"** (folders Violence / Weaponized / NonViolence).
# 3. **Save Version → Save & Run All** (runs in the background). Download `model_compare.zip` when done.
#
# The faculty's 6 sample clips (v001, v003, v006, v011, v026, w001) are skipped, so they stay a clean test set.
#
# `SET` picks the line-up: "small" (default) compares models of Gemma 4 E2B's size, the ones that fit the demo
# laptop's 6 GB GPU at 4-bit; "large" is the first run (E2B against E4B, 12B and Qwen3-VL 4B/8B).
# Cascade rows (a fast model + a bigger re-checker) are computed from the saved per-video results: the
# re-checker's verdict replaces the fast model's only on videos the fast model flagged.

# %%
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
OUT = WORK / "model_compare"  # the zip is named after it
OUT.mkdir(parents=True, exist_ok=True)
QUICK = True  # True: 20 videos per class (~60 videos per model). False: 60 per class.
PER_CLASS = 20 if QUICK else 60
# qwen3-vl's default tags are the thinking variants: with a JSON schema they returned empty answers on Kaggle
# (run 1: 0/60 valid). The -instruct tags answer directly.
SET = "small"
if SET == "small":  # about 2-4 B parameters, 4-bit in Ollama, each fits 6 GB
    MODELS = ["gemma4:e2b", "qwen3-vl:2b-instruct", "qwen2.5vl:3b", "gemma3:4b", "granite3.2-vision:2b",
              "llava-phi3:3.8b"]
    CASCADES = []
else:
    MODELS = ["gemma4:e2b", "gemma4:e4b-it-qat", "gemma4:12b-it-qat", "qwen3-vl:4b-instruct", "qwen3-vl:8b-instruct"]
    CASCADES = [("gemma4:e2b", "gemma4:e4b-it-qat"), ("gemma4:e2b", "gemma4:12b-it-qat"),
                ("gemma4:e2b", "qwen3-vl:8b-instruct")]
EXCLUDE = {"v001", "v003", "v006", "v011", "v026", "w001"}
LABEL_MAP = {"violence": "fighting", "weaponized": "weapon", "weaponised": "weapon", "nonviolence": "none",
             "non_violence": "none", "normal": "none"}
INCIDENT = {"fighting", "riot", "weapon", "harassment", "arrest", "theft", "vandalism", "crowd_incident"}
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)


def sh(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = r.stdout + r.stderr
    with (OUT / "commands.log").open("a") as f:
        f.write(f"\n$ {cmd}\n{out}\n")
    if r.returncode:
        raise RuntimeError(f"{cmd} failed:\n{out[-2000:]}")
    return out


# %% [markdown]
# ## Install Ollama and Aequitas

# %%
REPO = Path("/kaggle/working/Aequitas")
if not REPO.exists():
    sh(f"git clone --depth 1 https://github.com/sid-zyy/Aequitas.git {REPO}")
sys.path.insert(0, str(REPO))
sh("pip install -q ollama pydantic pyyaml httpx")
if not shutil.which("ollama"):
    # The installer unpacks a .tar.zst; Kaggle's image has no zstd (seen 2026-10-16: "requires zstd")
    if not shutil.which("zstd"):
        sh("(apt-get update -qq && apt-get install -y -qq zstd) || (sudo apt-get update -qq && sudo apt-get install -y -qq zstd)")
    sh("curl -fsSL https://ollama.com/install.sh | sh")
server = subprocess.Popen(["ollama", "serve"], stdout=open(OUT / "ollama.log", "w"), stderr=subprocess.STDOUT,
                          env={**os.environ, "OLLAMA_KEEP_ALIVE": "30m"})
time.sleep(5)
print(sh("ollama --version"))

# %% [markdown]
# ## The videos (SCVD), labels from folder names

# %%
from aequitas.eval.dataset import Item, load_dataset  # noqa: E402

rng = random.Random(0)
by_class: dict[str, list[Item]] = {}
for root in Path("/kaggle/input").glob("*"):
    for it in load_dataset(root):
        if it.label is None or it.path.stem.split("_")[0].lower() in EXCLUDE:
            continue
        lab = LABEL_MAP.get(it.label.strip().lower().replace(" ", "_"))
        if lab:
            by_class.setdefault(lab, []).append(Item(it.path, lab))
ITEMS = []
for lab, items in sorted(by_class.items()):
    rng.shuffle(items)
    ITEMS += items[:PER_CLASS]
print({k: len(v) for k, v in by_class.items()}, "-> evaluating", len(ITEMS), "videos per model")
assert ITEMS, "No SCVD videos found: add the dataset via Add Input"

# %% [markdown]
# ## Run every model through the real stages

# %%
from aequitas.config import Settings  # noqa: E402
from aequitas.eval.classify_eval import metrics  # noqa: E402
from aequitas.eval.dataset import run_dataset_eval  # noqa: E402
from aequitas.understand.vlm_client import make_client  # noqa: E402

RESULTS = OUT / "results.jsonl"
done = {json.loads(x)["model"] for x in RESULTS.read_text().splitlines()} if RESULTS.exists() else set()
ROWS = {}
for model in MODELS:
    rows_file = OUT / f"rows-{model.replace(':', '_')}.json"
    if model in done and rows_file.exists():
        ROWS[model] = json.loads(rows_file.read_text())
        continue
    try:
        t0 = time.time()
        sh(f"ollama pull {model}")
        pull_s = time.time() - t0
        s = Settings()
        s.vlm = s.vlm.model_copy(update={"backend": "ollama", "host": "http://127.0.0.1:11434", "model": model})
        client = make_client(s.vlm)
        # load the model before timing anything; give up early if it cannot answer at all
        probe = run_dataset_eval(client, client, ITEMS[:5], s.categories, label="warmup", out_dir=OUT / "reports")
        if probe["classification"]["valid"] == 0:
            raise RuntimeError(f"{model}: 0 of 5 probe videos gave a valid answer, skipped")
        rep = run_dataset_eval(client, client, ITEMS, s.categories, label=model.replace(":", "_"), out_dir=OUT / "reports")
        rows_file.write_text(json.dumps(rep["rows"]))
        ROWS[model] = rep["rows"]
        sh(f"ollama stop {model} || true")
        summary = {"model": model, "pull_s": round(pull_s), "total_execution_s": rep["total_execution_s"],
                   "per_video_s": rep["per_video_s"], **{k: rep["classification"][k] for k in
                   ("accuracy", "macro_f1", "incident_precision", "incident_recall", "valid", "n")}}
    except Exception as e:
        summary = {"model": model, "error": str(e)[-1500:]}
    with RESULTS.open("a") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=1))

# %% [markdown]
# ## Table, cascades, and chart

# %%
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def score(rows, preds, times):
    gold = [r["gold"] for r in rows]
    m = metrics(gold, preds)
    inc_g = [g != "none" for g in gold]
    inc_p = [p in INCIDENT for p in preds]
    det = sum(a == b for a, b in zip(inc_g, inc_p)) / len(gold)
    ts = sorted(times)
    return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "incident_detected_acc": round(det, 3),
            "incident_recall": m["incident_recall"], "incident_precision": m["incident_precision"],
            "time_per_video_p50_s": round(ts[len(ts) // 2], 2), "time_per_video_mean_s": round(sum(ts) / len(ts), 2)}


table = []
for model, rows in ROWS.items():
    ok = [r for r in rows if "error" not in r]
    table.append({"setup": model, **score(ok, [r["pred"] for r in ok], [r["video_total_s"] for r in ok])})
for small, big in CASCADES:
    if small in ROWS and big in ROWS:
        a = {r["video"]: r for r in ROWS[small] if "error" not in r}
        b = {r["video"]: r for r in ROWS[big] if "error" not in r}
        vids = [v for v in a if v in b]
        preds, times, rows, n_flagged = [], [], [], 0
        for v in vids:
            flagged = a[v]["pred"] is None or a[v]["pred"] != "none"
            n_flagged += flagged
            preds.append(b[v]["pred"] if flagged else a[v]["pred"])
            times.append(a[v]["video_total_s"] + (b[v]["video_total_s"] if flagged else 0))
            rows.append(a[v])
        table.append({"setup": f"cascade {small} -> {big}", **score(rows, preds, times),
                      "rechecked_share": round(n_flagged / len(vids), 2)})

df = pd.DataFrame(table).sort_values("incident_detected_acc", ascending=False)
df.to_csv(OUT / "compare.csv", index=False)
md = ["# Model comparison on SCVD (Kaggle T4, Ollama, our two-stage pipeline)", "",
      f"{len(ITEMS)} videos ({PER_CLASS} per class), faculty clips excluded. Gold labels: Violence -> fighting, "
      "Weaponized -> weapon, NonViolence -> none. 'incident detected' = any incident label vs none. "
      "Times are T4 times per whole clip (summary + label).", "", df.to_markdown(index=False)]
(OUT / "COMPARE.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))

fig, ax = plt.subplots(figsize=(8, 5))
for _, r in df.iterrows():
    ax.scatter(r["time_per_video_p50_s"], r["incident_detected_acc"], s=80)
    ax.annotate(r["setup"].replace("cascade ", "C: "), (r["time_per_video_p50_s"], r["incident_detected_acc"]),
                fontsize=7, xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("time per clip, p50 (s)  ← faster")
ax.set_ylabel("incident detected, accuracy  ↑ better")
ax.set_title("Which model: accuracy vs time (SCVD)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "compare.png", dpi=150)
server.terminate()
shutil.make_archive(str(WORK / "model_compare"), "zip", OUT)
print("Download model_compare.zip")
