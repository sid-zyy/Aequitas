# %% [markdown]
# # Aequitas · fine-tune Gemma 4 E2B on 12B's summaries (distillation, QLoRA)
#
# Gemma 4 12B caught 70 % of SCVD incidents in `model_compare`, E2B (what the laptop runs) only 15 %, but 12B
# is ~3x slower and does not fit a 6 GB GPU. This notebook teaches E2B to write 12B's summaries:
#
# 1. **Teacher:** 12B (Ollama) summarises and labels key-frame grids of the *training* clips with our real
#    stage-1 and stage-2 prompts. Only summaries whose label agrees with the dataset label are kept
#    (incident vs none), so 12B's own mistakes are not learned.
# 2. **Student:** E2B is fine-tuned with QLoRA (Unsloth) to produce those summaries from the same grid and prompt.
#    Stage 2 (classifying the summary text) is unchanged, so the three faculty stages stay separate.
# 3. **Test:** base E2B and the fine-tuned E2B on the **same 60 held-out SCVD clips as `model_compare`**, never
#    used in training, through the same pipeline and backend (Ollama, Q4_K_M). The 5 faculty clips are excluded.
#
# **Kaggle setup**
# 1. Accelerator **GPU T4 x2**, Internet **On**. (GPU 0 trains, GPU 1 runs Ollama.)
# 2. *Add Input* (every attached dataset is searched; labels come from folder names):
#    - https://www.kaggle.com/datasets/toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd (needed: the test set)
#    - https://www.kaggle.com/datasets/vulamnguyen/rwf2000
#    - https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset
#    - optional, for arrest / fire / accident / theft: a UCF-Crime *video* dataset with class folders
#      (Arrest, Arson, Explosion, RoadAccidents, Robbery, Shooting, ..., Normal_Videos...).
#    - optional, for **riots** (not on Kaggle; upload as your own private dataset):
#      **XD-Violence** riot (`_label_B4`) and normal (`_label_A`) videos, https://roc-ng.github.io/XD-Violence/ ;
#      and/or **Violent Flows** (crowd violence, request access from the authors), in a dataset named `violent-flows`.
#    - optional, the guardrail negatives: peaceful protests / processions / festivals in a folder `peaceful`.
#      Add these with riots, or the model may learn "crowd = riot".
#
# **How labels stay accurate.** Each clip's label comes from the dataset's human annotators. A 12B summary is
# trained on only if (1) 12B's label is one accepted for that human label (`ACCEPT`), (2) 12B's second look at
# the frames finds nothing invented, (3) it has no guardrail words. `review.html` shows 120 random kept samples
# for a human spot check; 20 % of every extra class is held out as a second test set.
# 3. **Save Version → Save & Run All**. About 3-4 h. Then download from the Output tab:
#    - `finetune_gemma.zip`: results (FINETUNE.md, chart, per-clip rows), the LoRA adapter, the training set list
#    - `aequitas-e2b-ollama.tar` (~3-4 GB): the fine-tuned model packaged for Ollama (see kaggle/README.md)
#
# Guardrails: the teacher uses our stage-1 prompt (actions, not identities); summaries mentioning religion, caste,
# party, flags or slogans are dropped from the training set (docs/POLICE_EDITION.md §8).

# %%
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SMOKE = os.environ.get("AEQUITAS_SMOKE") == "1"  # local check of the data cells only (no models)
WORK = Path(os.environ.get("AEQUITAS_WORK", "/kaggle/working"))
INPUT = Path(os.environ.get("AEQUITAS_INPUT", "/kaggle/input"))
OUT = WORK / "finetune_gemma"
OUT.mkdir(parents=True, exist_ok=True)
TMP = Path("/tmp/aequitas_ft") if not SMOKE else WORK / "tmp"
TMP.mkdir(parents=True, exist_ok=True)

TEACHER = "gemma4:12b-it-qat"
BASE_OLLAMA = "gemma4:e2b"
STUDENT_HF = "unsloth/gemma-4-E2B-it"
STUDENT_OLLAMA = "aequitas-e2b"
TRAIN_PER_CLASS = 150   # clips per class from the training pool (fighting, weapon, none; UCF classes capped at 40)
UCF_PER_CLASS = 40
GRIDS_PER_CLIP = 2      # the whole clip + one random 4 s window (what the live pipeline sends)
EPOCHS = 2
TEST_PER_CLASS = 20     # identical to model_compare (QUICK = True)
EXCLUDE = {"v001", "v006", "v011", "v026", "w001"}  # faculty clips: never trained on, never tested here
INCIDENT = {"fighting", "riot", "weapon", "harassment", "arrest", "theft", "vandalism", "crowd_incident",
            "fire", "accident", "chase", "person_down"}
# folder name -> our category. SCVD / RWF-2000 / RLVS, then UCF-Crime class folders.
LABEL_MAP = {"violence": "fighting", "fight": "fighting", "fighting": "fighting", "weaponized": "weapon",
             "weaponised": "weapon", "nonviolence": "none", "non_violence": "none", "nonfight": "none",
             "non_fight": "none", "normal": "none"}
UCF_MAP = {"arrest": "arrest", "arson": "fire", "explosion": "fire", "roadaccidents": "accident",
           "road_accidents": "accident", "robbery": "theft", "stealing": "theft", "shoplifting": "theft",
           "burglary": "theft", "shooting": "weapon", "assault": "fighting", "abuse": "fighting",
           "vandalism": "vandalism", "riot": "riot", "riots": "riot", "stone_pelting": "riot"}
# XD-Violence keeps its labels in the file name ("..._label_B4-0-0.mp4"): A normal, B1 fighting, B2 shooting,
# B4 riot, B5 abuse, B6 car accident, G explosion (Wu et al., ECCV 2020). Several labels: the first in this order.
XD_MAP = [("B4", "riot"), ("B1", "fighting"), ("B5", "fighting"), ("B2", "weapon"), ("B6", "accident"), ("G", "fire")]
# Folders whose "violence" means a violent CROWD (Violent Flows: 246 YouTube clips of crowd violence).
CROWD_HINTS = ("violentflow", "violent_flow", "crowd_violence", "crowdviolence")
# Your own peaceful gatherings (protests, processions, festivals, queues): always none, the guardrail case.
PEACEFUL = {"peaceful", "protest", "protests", "procession", "processions", "gathering", "gatherings",
            "festival", "festivals", "rally", "rallies", "queue", "crowd_normal", "normal_crowd"}
# What the teacher's label must be for a summary to be kept, per dataset label (stricter than incident-vs-none:
# a riot clip that 12B calls "accident" is dropped, its summary probably describes the wrong thing).
ACCEPT = {"fighting": {"fighting", "weapon", "harassment", "riot"}, "weapon": {"weapon", "fighting"},
          "riot": {"riot", "fighting", "vandalism", "crowd_incident"}, "arrest": {"arrest"}, "fire": {"fire"},
          "accident": {"accident"}, "theft": {"theft"}, "vandalism": {"vandalism", "riot"}, "none": {"none"}}
HOLDOUT = 0.2           # share of each non-SCVD class (riot, arrest, fire...) kept back as a second test set
EXTRA_TEST_PER_CLASS = 20
LONG_WINDOWS = 3        # untrimmed videos (UCF-Crime, XD-Violence): more 4 s windows per video
REVIEW_N = 120          # random kept samples in review.html, for a human spot check
# never train on summaries that describe people by these (CLAUDE.md guardrails)
BANNED = re.compile(r"\b(relig\w*|caste|hindu\w*|muslim\w*|islam\w*|christian\w*|sikh\w*|jain\w*|buddhis\w*|"
                    r"part(y|ies) (flag|worker|member)s?|flags?|banners?|slogans?|saffron|hijab\w*|burqa\w*|"
                    r"turban\w*|skull ?cap\w*|ethnic\w*|race|racial)\b", re.I)
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}
rng = random.Random(0)

if not SMOKE:
    # one GPU each: training must not compete with Ollama for memory
    import subprocess as _sp
    n_gpu = len([x for x in _sp.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines() if x])
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    OLLAMA_GPU = "1" if n_gpu > 1 else "0"
    print(_sp.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)


def sh(cmd: str, check: bool = True) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = r.stdout + r.stderr
    with (OUT / "commands.log").open("a") as f:
        f.write(f"\n$ {cmd}\n{out[-20000:]}\n")
    if check and r.returncode:
        raise RuntimeError(f"{cmd} failed:\n{out[-2000:]}")
    return out


# %% [markdown]
# ## Install Aequitas, Ollama and Unsloth

# %%
REPO = WORK / "Aequitas"
if not SMOKE:
    if not REPO.exists():
        sh(f"git clone --depth 1 https://github.com/sid-zyy/Aequitas.git {REPO}")
    sh("pip install -q ollama pydantic pyyaml httpx")
    # the versions Unsloth's Gemma 4 E2B vision notebook uses
    sh("pip install -q unsloth")
    sh('pip install -q --no-deps transformers==5.5.0 "tokenizers>=0.22.0,<=0.23.0"')
    sh("pip install -q --no-deps --upgrade timm")
    if not shutil.which("ollama"):
        if not shutil.which("zstd"):  # the installer unpacks a .tar.zst; Kaggle's image has no zstd
            sh("(apt-get update -qq && apt-get install -y -qq zstd) || (sudo apt-get update -qq && sudo apt-get install -y -qq zstd)")
        sh("curl -fsSL https://ollama.com/install.sh | sh")
    OLLAMA_MODELS = Path("/root/.ollama/models")
    server = subprocess.Popen(["ollama", "serve"], stdout=open(OUT / "ollama.log", "w"), stderr=subprocess.STDOUT,
                              env={**os.environ, "OLLAMA_KEEP_ALIVE": "30m", "CUDA_VISIBLE_DEVICES": OLLAMA_GPU,
                                   "OLLAMA_MODELS": str(OLLAMA_MODELS)})
    time.sleep(5)
    print(sh("ollama --version"))
else:
    REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# %% [markdown]
# ## Clips: the test set first (same 60 SCVD clips as model_compare), then the training pool

# %%
from aequitas.eval.dataset import Item, clip_grid, load_dataset  # noqa: E402


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def label_of(path: Path) -> tuple[str | None, str]:
    """Category from the file name (XD-Violence) or the nearest folder name that we know, and the dataset family."""
    m = re.search(r"_label_([A-Z0-9-]+)", path.name)
    if m:
        tags = set(m.group(1).split("-"))
        if tags <= {"A", "0"}:
            return "none", "xd"
        return next((lab for tag, lab in XD_MAP if tag in tags), None), "xd"
    crowd = any(h in norm(p) for p in path.parts for h in CROWD_HINTS)
    for part in reversed(path.parts[:-1]):
        n = norm(part)
        if n in PEACEFUL:
            return "none", "peaceful"
        if n in LABEL_MAP:
            if crowd:
                return ("riot" if LABEL_MAP[n] != "none" else "none"), "crowd"
            return LABEL_MAP[n], "violence"
        if n.replace("_", "") in UCF_MAP or n in UCF_MAP:
            return UCF_MAP.get(n, UCF_MAP.get(n.replace("_", ""))), "ucf"
        if n.startswith("normal_videos") or n.startswith("testing_normal") or n.startswith("training_normal"):
            return "none", "ucf"
    return None, ""


def faculty(path: Path) -> bool:
    return path.stem.split("_")[0].lower() in EXCLUDE


def scvd_root() -> Path | None:
    """The dataset folder that contains a 'Weaponized' class folder (only SCVD has one)."""
    for p in INPUT.rglob("*"):
        if p.is_dir() and norm(p.name) in ("weaponized", "weaponised"):
            for parent in p.parents:  # the top folder of that dataset under /kaggle/input[/datasets/<owner>]
                if parent.parent in (INPUT, INPUT / "datasets") or parent.parent.parent == INPUT / "datasets":
                    return parent
    return None


# Test set: exactly model_compare's selection (load_dataset order, Random(0), sorted classes, 20 per class)
SCVD = scvd_root()
TEST: list[Item] = []
if SCVD is not None:
    t_rng = random.Random(0)
    by_class: dict[str, list[Item]] = {}
    mc_map = {"violence": "fighting", "weaponized": "weapon", "weaponised": "weapon", "nonviolence": "none",
              "non_violence": "none", "normal": "none"}
    for it in load_dataset(SCVD):
        if it.label is None or faculty(it.path):
            continue
        lab = mc_map.get(it.label.strip().lower().replace(" ", "_"))
        if lab:
            by_class.setdefault(lab, []).append(Item(it.path, lab))
    for lab, items in sorted(by_class.items()):
        t_rng.shuffle(items)
        TEST += items[:TEST_PER_CLASS]
test_paths = {str(it.path.resolve()) for it in TEST}
print("SCVD:", SCVD, "| test clips:", len(TEST))

# Training pool: every other labelled clip in every attached dataset
pool: dict[str, list[Path]] = {}
for p in sorted(INPUT.rglob("*")):
    if p.suffix.lower() not in VIDEO_EXTS or faculty(p) or str(p.resolve()) in test_paths:
        continue
    lab, fam = label_of(p.relative_to(INPUT))
    if lab:
        pool.setdefault(f"{fam}:{lab}", []).append(p)
TRAIN: list[Item] = []
TEST_EXTRA: list[Item] = []  # held out from the other datasets: riot, arrest, fire, peaceful crowds...
for key, paths in sorted(pool.items()):
    rng.shuffle(paths)
    if not key.startswith("violence:"):  # SCVD / RWF / RLVS are tested on SCVD; the rest gets its own test
        n_out = min(EXTRA_TEST_PER_CLASS, max(1, round(len(paths) * HOLDOUT))) if len(paths) >= 3 else 0
        TEST_EXTRA += [Item(p, key.split(":")[1]) for p in paths[:n_out]]
        paths = paths[n_out:]
    cap = UCF_PER_CLASS if key.startswith(("ucf:", "xd:")) else TRAIN_PER_CLASS
    if key == "violence:fighting":  # spread fights over the datasets (SCVD, RWF-2000, RLVS)
        by_ds: dict[str, list[Path]] = {}
        for p in paths:
            by_ds.setdefault(p.relative_to(INPUT).parts[0 if p.relative_to(INPUT).parts[0] != "datasets" else 2], []).append(p)
        picked, lists = [], list(by_ds.values())
        while len(picked) < cap and any(lists):
            for lst in lists:
                if lst and len(picked) < cap:
                    picked.append(lst.pop())
        paths = picked
    TRAIN += [Item(p, key.split(":")[1]) for p in paths[:cap]]
FAMILY = {str(p): k.split(":")[0] for k, ps in pool.items() for p in ps}
print({k: len(v) for k, v in pool.items()}, "-> training clips:", len(TRAIN), "| extra test clips:", len(TEST_EXTRA))
held = test_paths | {str(it.path.resolve()) for it in TEST_EXTRA}
assert not held & {str(it.path.resolve()) for it in TRAIN}, "a test clip leaked into training"
if not any(it.label == "riot" for it in TRAIN):
    print("NOTE: no riot clips attached (XD-Violence or Violent Flows), riots are not trained in this run")
(OUT / "test_clips.txt").write_text("\n".join(f"{it.label}\t{it.path}" for it in TEST))
(OUT / "test_extra_clips.txt").write_text("\n".join(f"{it.label}\t{it.path}" for it in TEST_EXTRA))
(OUT / "train_clips.txt").write_text("\n".join(f"{it.label}\t{it.path}" for it in TRAIN))
if not SMOKE:
    assert TEST, "SCVD not found: add it via Add Input (it is the test set)"
    assert TRAIN, "no training clips found"

# %% [markdown]
# ## Key-frame grids (the stage-1 input)

# %%
import cv2  # noqa: E402

from aequitas.config import Zones  # noqa: E402
from aequitas.ingest.ring_buffer import BufferedFrame  # noqa: E402
from aequitas.understand.keyframes import encode_jpeg, make_grid  # noqa: E402


def window_grid(path: Path, seconds: float = 4.0, tile: int = 448, n: int = 4) -> bytes | None:
    """n frames from one random `seconds` window, like a live 4 s window."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    span = int(seconds * fps)
    if total <= span * 1.5:  # short clip: the whole-clip grid already is a 4 s window
        cap.release()
        return None
    start = rng.randrange(0, total - span)
    frames = []
    for k in range(n):
        idx = start + int((k + 0.5) * span / n)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if ok:
            frames.append(BufferedFrame(int(1000 * idx / fps), f, 1.0, {}))
    cap.release()
    return encode_jpeg(make_grid(frames, None, Zones(), tile)) if frames else None


GRIDS = TMP / "grids"
GRIDS.mkdir(exist_ok=True)
samples = []  # {"grid": path, "gold": category, "clip": path}
for i, it in enumerate(TRAIN):
    n_win = LONG_WINDOWS if FAMILY.get(str(it.path)) in ("ucf", "xd") else GRIDS_PER_CLIP - 1
    makers = [lambda p: clip_grid(p, 448)] + [window_grid] * n_win
    for j, make in enumerate(makers):
        out = GRIDS / f"{i:05d}_{j}.jpg"
        if not out.exists():
            g = make(it.path)
            if g is None:
                continue
            out.write_bytes(g)
        samples.append({"grid": str(out), "gold": it.label, "clip": str(it.path)})
print(len(samples), "training grids")

# %% [markdown]
# ## Teacher: 12B summarises and labels every training grid (our real stage 1 + stage 2)

# %%
from aequitas.config import Settings  # noqa: E402
from aequitas.understand.stages import understand  # noqa: E402
from aequitas.understand.vlm_client import make_client  # noqa: E402

S = Settings()
CATS = S.categories


def ollama_client(model: str):
    s = Settings()
    s.vlm = s.vlm.model_copy(update={"backend": "ollama", "host": "http://127.0.0.1:11434", "model": model})
    return make_client(s.vlm)


from pydantic import BaseModel  # noqa: E402


class Check(BaseModel):
    every_statement_visible: bool
    wrong_part: str = ""


CHECK_PROMPT = ("Here is a 2x2 grid of four frames from a camera and a description of it:\n\n\"{summary}\"\n\n"
                "Look at the frames again. Is every statement in the description visible in the frames: the "
                "number of people, what they do, objects, police, fire, vehicles? Answer false if anything is "
                "invented or wrong, and name the wrong part in a few words. Respond with JSON only.")

TEACH = OUT / "teacher.jsonl"
if not SMOKE:
    sh(f"ollama pull {TEACHER}")
    teacher = ollama_client(TEACHER)
    done = {json.loads(x)["grid"] for x in TEACH.read_text().splitlines()} if TEACH.exists() else set()
    t0 = time.time()
    with TEACH.open("a") as f:
        for k, smp in enumerate(samples):
            if smp["grid"] in done:
                continue
            grid = Path(smp["grid"]).read_bytes()
            s, _ = teacher.summarize(grid, highlighted=False)
            label, chk = None, None
            if s is not None:
                label, _, _ = teacher.classify_text(s.summary, s.visible_evidence, CATS)
                # second look: 12B checks its own summary against the frames (drops invented details)
                chk = teacher.structured([{"role": "user", "content": CHECK_PROMPT.format(summary=s.summary),
                                           "images": [grid]}], Check, max_tokens=80)
            f.write(json.dumps({**smp, "summary": s.summary if s else None,
                                "visible_evidence": s.visible_evidence if s else None, "teacher_label": label,
                                "checked": chk.every_statement_visible if chk else None,
                                "wrong_part": chk.wrong_part if chk else None}) + "\n")
            f.flush()
            if k % 50 == 0:
                print(f"teacher {k}/{len(samples)}  {time.time() - t0:.0f} s")
    sh(f"ollama stop {TEACHER}", check=False)


def why_dropped(r: dict) -> str | None:
    """Three checks before a summary becomes training data: the guardrails, 12B's label agrees with the
    dataset's human label (ACCEPT), and 12B's second look found nothing invented."""
    if not r.get("summary") or r.get("teacher_label") is None:
        return "teacher failed"
    if BANNED.search(r["summary"] + " " + (r.get("visible_evidence") or "")):
        return "guardrail words"
    if r["teacher_label"] not in ACCEPT.get(r["gold"], {r["gold"]}):
        return "label disagrees"
    if r.get("checked") is False:
        return "check found an invented detail"
    return None


rows = [json.loads(x) for x in TEACH.read_text().splitlines()] if TEACH.exists() else []
kept = [r for r in rows if why_dropped(r) is None]
by_gold: dict[str, list] = {}
for r in kept:
    by_gold.setdefault(r["gold"], []).append(r)
reasons: dict[str, int] = {}
for r in rows:
    if (w := why_dropped(r)) is not None:
        reasons[w] = reasons.get(w, 0) + 1
per_class = {g: f"{sum(r['gold'] == g for r in kept)}/{sum(r['gold'] == g for r in rows)}"
             for g in sorted({r["gold"] for r in rows})}
stats = {"teacher_grids": len(rows), "kept": len(kept), "kept_of_total_per_class": per_class, "dropped": reasons}
print(json.dumps(stats, indent=1))
(OUT / "teacher_stats.json").write_text(json.dumps(stats, indent=1))

# review.html: random kept samples (grid, summary, labels) for a human spot check before trusting the run
if rows:
    import base64

    pick = random.Random(1).sample(kept, min(REVIEW_N, len(kept)))
    cards = []
    for i, r in enumerate(pick):
        im = cv2.imread(r["grid"])
        im = cv2.resize(im, (448, 448 * im.shape[0] // im.shape[1]))
        b64 = base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 70])[1]).decode()
        cards.append(f'<div class="c"><img src="data:image/jpeg;base64,{b64}"><p><b>#{i + 1}</b> dataset: '
                     f'<b>{r["gold"]}</b> · 12B: <b>{r["teacher_label"]}</b></p><p>{r["summary"]}</p>'
                     f'<p class="e">{r.get("visible_evidence") or ""}</p></div>')
    (OUT / "review.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Training sample review</title><style>body{font:14px sans-serif;"
        "margin:16px}.g{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}.c{border:"
        "1px solid #ccc;border-radius:6px;padding:8px}img{width:100%}.e{color:#666}</style><h1>Training sample "
        f"review ({len(pick)} of {len(kept)} kept)</h1><p>Note the numbers of cards whose summary is wrong or "
        "whose label is wrong. Under ~5 % wrong is fine to train on.</p><div class=g>" + "".join(cards) + "</div>",
        encoding="utf-8")

# %% [markdown]
# ## Baseline: base E2B on the 60 test clips (same pipeline and backend as model_compare)

# %%
from aequitas.eval.dataset import run_dataset_eval  # noqa: E402

RESULTS, EXTRA = {}, {}  # model -> report on the SCVD test / on the extra test (riot, arrest, fire, peaceful)


def evaluate(name: str, vlm, clf, label: str) -> None:
    RESULTS[name] = run_dataset_eval(vlm, clf, TEST, CATS, label=label, out_dir=OUT / "reports")
    if TEST_EXTRA:
        EXTRA[name] = run_dataset_eval(vlm, clf, TEST_EXTRA, CATS, label=label + "_extra", out_dir=OUT / "reports")


if not SMOKE:
    sh(f"ollama pull {BASE_OLLAMA}")
    base = ollama_client(BASE_OLLAMA)
    run_dataset_eval(base, base, TEST[:5], CATS, label="warmup", out_dir=OUT / "reports")  # load before timing
    evaluate("gemma4:e2b (base)", base, base, "base_e2b")
    sh(f"ollama stop {BASE_OLLAMA}", check=False)
    print({k: v["classification"]["incident_recall"] for k, v in RESULTS.items()})

# %% [markdown]
# ## Train the student: E2B, QLoRA (Unsloth), on GPU 0

# %%
from aequitas.understand.prompts import summarize_prompt  # noqa: E402

PROMPT = summarize_prompt(highlighted=False, concise=S.vlm.concise_summary, view="cctv")


def to_conversation(r: dict, image) -> dict:
    target = json.dumps({"summary": r["summary"].strip(), "visible_evidence": (r.get("visible_evidence") or "").strip()})
    return {"messages": [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]},
        {"role": "assistant", "content": [{"type": "text", "text": target}]},
    ]}


LORA = OUT / "lora"
if not SMOKE:
    from PIL import Image
    from unsloth import FastVisionModel, get_chat_template, is_bf16_supported  # noqa: E402
    from unsloth.trainer import UnslothVisionDataCollator  # noqa: E402
    from trl import SFTConfig, SFTTrainer  # noqa: E402
    import torch  # noqa: E402

    model, processor = FastVisionModel.from_pretrained(STUDENT_HF, load_in_4bit=True,
                                                       use_gradient_checkpointing="unsloth")
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=True, finetune_language_layers=True, finetune_attention_modules=True,
        finetune_mlp_modules=True, r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=3407,
        target_modules="all-linear")
    processor = get_chat_template(processor, "gemma-4")
    rng.shuffle(kept)
    data = [to_conversation(r, Image.open(r["grid"]).convert("RGB")) for r in kept]
    FastVisionModel.for_training(model)
    trainer = SFTTrainer(
        model=model, train_dataset=data, processing_class=processor.tokenizer,
        data_collator=UnslothVisionDataCollator(model, processor),
        args=SFTConfig(
            per_device_train_batch_size=1, gradient_accumulation_steps=8, num_train_epochs=EPOCHS,
            learning_rate=1e-4, warmup_ratio=0.05, max_grad_norm=0.3, lr_scheduler_type="cosine",
            optim="adamw_8bit", weight_decay=0.001, logging_steps=10, save_strategy="no", seed=3407,
            fp16=not is_bf16_supported(), bf16=is_bf16_supported(), output_dir=str(TMP / "outputs"),
            report_to="none", remove_unused_columns=False, dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True}, max_length=2048))
    t0 = time.time()
    train_stats = trainer.train()
    train_s = time.time() - t0
    (OUT / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    model.save_pretrained(str(LORA))
    processor.save_pretrained(str(LORA))
    print(f"trained {len(data)} samples x {EPOCHS} epochs in {train_s / 60:.1f} min; "
          f"peak GPU {torch.cuda.max_memory_reserved() / 2**30:.1f} GB")

# %% [markdown]
# ## Export: merged 16-bit weights -> Ollama (Q4_K_M, same precision as the base gemma4:e2b)

# %%
MERGED = TMP / "merged"
OLLAMA_OK = False
if not SMOKE:
    model.save_pretrained_merged(str(MERGED), processor)
    (TMP / "Modelfile").write_text(f"FROM {MERGED}\n")
    try:
        print(sh(f"ollama create {STUDENT_OLLAMA} --quantize q4_K_M -f {TMP / 'Modelfile'}"))
        OLLAMA_OK = True
    except RuntimeError as e:
        print("ollama import failed, evaluating with Unsloth instead:", str(e)[-800:])

# %% [markdown]
# ## Test: fine-tuned E2B on the same 60 clips

# %%
if not SMOKE and OLLAMA_OK:  # Ollama runs on GPU 1, so the trained model can stay on GPU 0 for the fallback
    student = ollama_client(STUDENT_OLLAMA)
    probe = run_dataset_eval(student, student, TEST[:5], CATS, label="warmup", out_dir=OUT / "reports")
    if probe["classification"]["valid"] == 0:  # the import loaded but cannot answer: use Unsloth
        OLLAMA_OK = False
    else:
        evaluate("aequitas-e2b (fine-tuned)", student, student, "finetuned_e2b")
if not SMOKE and not OLLAMA_OK:
    # Fallback: stage 1 from the fine-tuned weights in transformers (slower, so times are not comparable);
    # stage 2 with base gemma4:e2b in Ollama (it only reads text, and the fine-tune did not change stage 2)
    from aequitas.understand.prompts import SummaryAnswer

    FastVisionModel.for_inference(model)

    class HfSummarizer:
        def summarize(self, grid_jpeg, highlighted=False, view="cctv"):
            import io
            t = time.perf_counter()
            img = Image.open(io.BytesIO(grid_jpeg)).convert("RGB")
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
            text = processor.apply_chat_template(msgs, add_generation_prompt=True)
            inp = processor(img, text, add_special_tokens=False, return_tensors="pt").to("cuda")
            out = model.generate(**inp, max_new_tokens=160, do_sample=False)
            raw = processor.tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
            m = re.search(r"\{.*\}", raw, re.S)
            try:
                ans = SummaryAnswer.model_validate_json(m.group(0)) if m else None
            except Exception:
                ans = None
            return ans, (time.perf_counter() - t) * 1000

    base = ollama_client(BASE_OLLAMA)
    evaluate("fine-tuned E2B (transformers; stage 2 base)", HfSummarizer(), base, "finetuned_e2b_hf")

# %% [markdown]
# ## Results table, chart, and the downloads

# %%
if not SMOKE:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    table = []
    for name, rep in RESULTS.items():
        c = rep["classification"]
        table.append({"model": name, "incident_recall": c["incident_recall"], "incident_precision": c["incident_precision"],
                      "accuracy": c["accuracy"], "macro_f1": c["macro_f1"], "valid": f'{c["valid"]}/{c["n"]}',
                      "time_per_clip_p50_s": rep["per_video_s"]["p50"]})
    table.append({"model": "gemma4:12b-it-qat (teacher, from model_compare run 1)", "incident_recall": 0.70,
                  "incident_precision": 0.8235, "accuracy": 0.3333, "macro_f1": 0.3172, "valid": "60/60",
                  "time_per_clip_p50_s": 5.11})
    df = pd.DataFrame(table)
    md = ["# Fine-tuned Gemma 4 E2B vs base (Kaggle T4, Ollama, our two-stage pipeline)", "",
          f"Test: the 60 held-out SCVD clips of model_compare ({TEST_PER_CLASS} per class), none used in training; "
          "faculty clips excluded. Student trained on "
          f"{len(kept)} teacher summaries ({json.dumps(stats['kept_of_total_per_class'])}), {EPOCHS} epochs, QLoRA r=16.",
          "Note: the SCVD training clips come from the same dataset as the test clips (different videos), so this "
          "measures in-domain gain; the 21 Bengaluru demo clips are the out-of-domain check.", "",
          df.to_markdown(index=False)]
    if EXTRA:
        # per class: share of held-out clips given exactly an accepted label; for none, share left as none
        classes = sorted({it.label for it in TEST_EXTRA})
        md += ["", f"## Second test set: {len(TEST_EXTRA)} held-out clips from the other datasets "
               "(riot, arrest, fire, peaceful crowds...)", "",
               "Share of clips given an accepted label (for none: left as none, i.e. no false alarm).", "",
               "| model | " + " | ".join(f"{c} (n={sum(it.label == c for it in TEST_EXTRA)})" for c in classes) + " |",
               "|---|" + "---|" * len(classes)]
        for name, rep in EXTRA.items():
            ok = [r for r in rep["rows"] if "error" not in r]
            cells = []
            for c in classes:
                rs = [r for r in ok if r["gold"] == c]
                cells.append(f'{sum(r["pred"] in ACCEPT.get(c, {c}) for r in rs)}/{len(rs)}')
            md.append(f"| {name} | " + " | ".join(cells) + " |")
    if not OLLAMA_OK:
        md += ["", "Ollama import failed: the fine-tuned row ran in transformers, so its time is not comparable."]
    md += ["", f"Training data checks: {json.dumps(stats['dropped'])} dropped. Spot check: review.html."]
    (OUT / "FINETUNE.md").write_text("\n".join(md), encoding="utf-8")
    df.to_csv(OUT / "finetune.csv", index=False)
    print("\n".join(md))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, r in df.iterrows():
        ax.scatter(r["time_per_clip_p50_s"], r["incident_recall"] or 0, s=80)
        ax.annotate(r["model"], (r["time_per_clip_p50_s"], r["incident_recall"] or 0), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("time per clip, p50 (s)  ← faster")
    ax.set_ylabel("incidents caught (recall)  ↑ better")
    ax.set_title("Fine-tuned E2B vs base and teacher (SCVD test clips)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "finetune.png", dpi=150)

    # the Ollama model as one tar: manifest + blobs, laid out like ~/.ollama/models
    if OLLAMA_OK:
        man = OLLAMA_MODELS / "manifests/registry.ollama.ai/library" / STUDENT_OLLAMA / "latest"
        meta = json.loads(man.read_text())
        digests = [meta["config"]["digest"]] + [layer["digest"] for layer in meta["layers"]]
        files = [man.relative_to(OLLAMA_MODELS)] + [Path("blobs") / d.replace(":", "-") for d in digests]
        sh(f"tar -cf {WORK / 'aequitas-e2b-ollama.tar'} -C {OLLAMA_MODELS} " + " ".join(str(f) for f in files))
        print("aequitas-e2b-ollama.tar:", round((WORK / "aequitas-e2b-ollama.tar").stat().st_size / 2**30, 2), "GB")
    server.terminate()
    shutil.make_archive(str(WORK / "finetune_gemma"), "zip", OUT)
    print("Download finetune_gemma.zip and aequitas-e2b-ollama.tar from the Output tab")
