# %% [markdown]
# # Aequitas · violence clip classifier
#
# Trains a small 3D CNN that looks at a **2-second clip** and scores *violence vs not*. It needs no
# person tracking, so it works where Aequitas's fight rules fail (low-res CCTV, crowds, occlusion).
#
# **Kaggle setup**
# 1. Settings → Accelerator → **GPU T4 x2** (or P100). Internet **on** (for the pretrained weights).
# 2. *Add data* (any of these; every attached dataset is searched):
#    - **Smart-City CCTV Violence Detection (SCVD)**: real CCTV, `Violence` / `Weaponized` / `NonViolence`.
#      The faculty's sample clips (v001, w001...) look like SCVD: it is the closest match to their data.
#    - **RWF-2000** (`Fight/NonFight`, real CCTV) and/or **Real Life Violence Situations** (`Violence/NonViolence`).
#    - Optional: your own hard negatives (festivals, sports, dancing, processions) in a folder `NonViolence`.
# 3. Edit `SOURCES` below if your paths differ, then *Run all*.
# 4. Download `/kaggle/working/violence_clip.zip` and unzip it into Aequitas's `models/` folder.
#
# Licences: check each dataset's terms; RWF-2000 and most violence datasets are research-only.

# %%
import json
import math
import os
import random
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset

SMOKE = os.environ.get("AEQUITAS_SMOKE") == "1"  # tiny synthetic run for local testing
WORK = Path(os.environ.get("AEQUITAS_WORK", "/kaggle/working"))

# %% [markdown]
# ## Configuration
# `clip_fps` and `clip_len` must match how Aequitas samples video. The default (4 fps × 8 frames =
# 2 s) matches Aequitas's default `sample_fps: 4`. When you move to a GPU and raise `sample_fps` to 8,
# retrain with `clip_fps=8, clip_len=16`.

# %%
# Every dataset attached to the notebook is searched; labels come from folder names.
SOURCES = sorted(str(p) for p in Path("/kaggle/input").glob("*")) if Path("/kaggle/input").exists() else []
# The faculty's sample clips are the test set: never train on them, or their score means nothing.
EXCLUDE_STEMS = {"v001", "v006", "v011", "v026", "w001"}
CFG = dict(
    arch="r3d_18",          # r3d_18 | mc3_18 | r2plus1d_18 (torchvision, Kinetics-400 pretrained)
    clip_fps=4, clip_len=8, short_side=128, crop=112,
    epochs=12, batch=16, lr=1e-4, head_lr=1e-3, weight_decay=1e-4,
    workers=4, seed=0, val_frac=0.15, degrade_p=0.3,
    cache_dir=str(WORK / "cache"), out_dir=str(WORK / "violence_clip"),
)
POSITIVE = {"fight", "fights", "violence", "violent", "riot", "fighting",
            "weaponized", "weaponised", "weapon", "weapons", "weaponizedviolence"}
NEGATIVE = {"nonfight", "non_fight", "nofight", "nonviolence", "non_violence", "non-violence",
            "nonviolent", "noviolence", "normal", "negative"}
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".mpg", ".mpeg", ".webm", ".m4v"}

# %% [markdown]
# ## Smoke-test data (only when `AEQUITAS_SMOKE=1`)
# Synthetic clips: "violent" = two squares jittering into each other, "normal" = one square drifting.

# %%
def make_synthetic(root: Path, n_per_class: int = 6) -> None:
    for label, split_n in (("Fight", n_per_class), ("NonFight", n_per_class)):
        for split in ("train", "val"):
            d = root / split / label
            d.mkdir(parents=True, exist_ok=True)
            for i in range(split_n if split == "train" else 2):
                vw = cv2.VideoWriter(str(d / f"{i}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 25, (160, 120))
                rng = np.random.default_rng(i + (100 if label == "Fight" else 0))
                for t in range(75):
                    img = np.full((120, 160, 3), 40, np.uint8)
                    if label == "Fight":
                        for c in ((60, 60), (90, 60)):
                            x, y = c[0] + int(rng.integers(-12, 12)), c[1] + int(rng.integers(-12, 12))
                            cv2.rectangle(img, (x, y), (x + 20, y + 30), (200, 200, 200), -1)
                    else:
                        x = 20 + t
                        cv2.rectangle(img, (x, 50), (x + 20, 80), (200, 200, 200), -1)
                    vw.write(img)
                vw.release()


if SMOKE:
    synth = WORK / "synthetic"
    if not synth.exists():
        make_synthetic(synth)
    SOURCES = [str(synth)]
    CFG.update(epochs=2, batch=4, workers=0)

random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| GPUs:", torch.cuda.device_count(), "| sources:", SOURCES)

# %% [markdown]
# ## 1 · Find videos and labels
# The label comes from a folder name in the path (`Fight`, `NonFight`, `Violence`, `NonViolence`...).
# The split comes from a `train` / `val` / `test` folder if present; otherwise it is random.

# %%
def label_of(path: Path) -> int | None:
    parts = [p.lower() for p in path.parts]
    if any(p in NEGATIVE for p in parts):
        return 0
    if any(p in POSITIVE for p in parts):
        return 1
    return None


def split_of(path: Path) -> str | None:
    parts = {p.lower() for p in path.parts}
    if parts & {"val", "valid", "validation", "test"}:
        return "val"
    if "train" in parts:
        return "train"
    return None


items = []
for src in SOURCES:
    root = Path(src)
    if not root.exists():
        print(f"WARNING: {src} not found, skipped (add it via 'Add data' or edit SOURCES)")
        continue
    for p in root.rglob("*"):
        if p.stem.split("_")[0].lower() in EXCLUDE_STEMS:
            continue  # held-out faculty clips
        if p.suffix.lower() in VIDEO_EXTS and (lab := label_of(p.relative_to(root))) is not None:
            split = split_of(p.relative_to(root)) or ("val" if random.random() < CFG["val_frac"] else "train")
            items.append({"path": str(p), "label": lab, "split": split})
assert items, "No labelled videos found: check SOURCES and folder names"
for s in ("train", "val"):
    sub = [i for i in items if i["split"] == s]
    print(f"{s}: {len(sub)} videos, {sum(i['label'] for i in sub)} violent")

# %% [markdown]
# ## 2 · Decode once into a frame cache
# Each video is decoded at `clip_fps`, resized to `short_side`, and stored as a `uint8` array, so
# training epochs don't decode video again. (RWF-2000 at these settings: ~1 GB.)

# %%
def cache_one(args):
    path, out, fps, short = args
    if os.path.exists(out):
        return out
    cap = cv2.VideoCapture(path)
    native = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step, nxt, i, frames = native / fps, 0.0, 0, []
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i + 1e-6 >= nxt:
            ok, img = cap.retrieve()
            if ok:
                h, w = img.shape[:2]
                s = short / min(h, w)
                frames.append(cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)[..., ::-1])
            nxt += step
        i += 1
    cap.release()
    if len(frames) < 2:
        return None
    np.save(out, np.stack(frames))
    return out


cache = Path(CFG["cache_dir"]); cache.mkdir(parents=True, exist_ok=True)
jobs = [(it["path"], str(cache / f"{k:06d}.npy"), CFG["clip_fps"], CFG["short_side"]) for k, it in enumerate(items)]
t0 = time.time()
if CFG["workers"]:
    with Pool(CFG["workers"]) as pool:
        outs = pool.map(cache_one, jobs, chunksize=4)
else:
    outs = [cache_one(j) for j in jobs]
for it, out in zip(items, outs):
    it["cache"] = out
items = [it for it in items if it["cache"]]
print(f"cached {len(items)} videos in {time.time() - t0:.0f}s")

# %% [markdown]
# ## 3 · Dataset and augmentation
# Training clips start at a random time and get random crop, flip, brightness/contrast jitter and,
# with probability `degrade_p`, a **CCTV-style degradation** (downscale + upscale + JPEG artefacts).

# %%
MEAN = np.array([0.43216, 0.394666, 0.37645], np.float32)  # Kinetics-400, as used by torchvision video models
STD = np.array([0.22803, 0.22145, 0.216989], np.float32)


def degrade(frames: np.ndarray) -> np.ndarray:
    t, h, w, _ = frames.shape
    s = random.uniform(0.35, 0.6)
    q = random.randint(25, 60)
    out = np.empty_like(frames)
    for i in range(t):
        small = cv2.resize(frames[i], (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
        ok, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, q])
        small = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED) if ok else small
        out[i] = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


class Clips(Dataset):
    def __init__(self, items, train: bool):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        it = self.items[k]
        v = np.load(it["cache"], mmap_mode="r")
        n, L = len(v), CFG["clip_len"]
        start = random.randint(0, max(0, n - L)) if self.train else max(0, (n - L) // 2)
        idx = [min(n - 1, start + i) for i in range(L)]
        clip = np.asarray(v[idx])
        _, h, w, _ = clip.shape
        c = CFG["crop"]
        if self.train:
            y, x = random.randint(0, h - c), random.randint(0, w - c)
        else:
            y, x = (h - c) // 2, (w - c) // 2
        clip = clip[:, y:y + c, x:x + c]
        if self.train:
            if random.random() < 0.5:
                clip = clip[:, :, ::-1]
            if random.random() < CFG["degrade_p"]:
                clip = degrade(np.ascontiguousarray(clip))
            clip = clip.astype(np.float32) * random.uniform(0.75, 1.25) + random.uniform(-20, 20)
            clip = np.clip(clip, 0, 255)
        x = (clip.astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(x.transpose(3, 0, 1, 2))), it["label"]


train_items = [i for i in items if i["split"] == "train"]
val_items = [i for i in items if i["split"] == "val"]
dl_train = DataLoader(Clips(train_items, True), CFG["batch"], shuffle=True, num_workers=CFG["workers"],
                      drop_last=len(train_items) > CFG["batch"], pin_memory=DEVICE == "cuda")
dl_val = DataLoader(Clips(val_items, False), CFG["batch"], num_workers=CFG["workers"])

# %% [markdown]
# ## 4 · Model
# A Kinetics-400 pretrained 3D ResNet from torchvision with a new 2-class head.

# %%
def build_model(arch: str, pretrained: bool) -> nn.Module:
    from torchvision.models import video as tvv

    weights = {"r3d_18": tvv.R3D_18_Weights, "mc3_18": tvv.MC3_18_Weights,
               "r2plus1d_18": tvv.R2Plus1D_18_Weights}[arch].KINETICS400_V1 if pretrained else None
    m = getattr(tvv, arch)(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


model = build_model(CFG["arch"], pretrained=not SMOKE).to(DEVICE)
net = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
head = list(model.fc.parameters())
body = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
opt = torch.optim.AdamW([{"params": body, "lr": CFG["lr"]}, {"params": head, "lr": CFG["head_lr"]}],
                        weight_decay=CFG["weight_decay"])
steps = max(1, CFG["epochs"] * len(dl_train))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[CFG["lr"], CFG["head_lr"]], total_steps=steps, pct_start=0.1)
scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

# %% [markdown]
# ## 5 · Train
# The best epoch by validation F1 is kept.

# %%
def evaluate(m) -> tuple[np.ndarray, np.ndarray]:
    m.eval()
    logits, labels = [], []
    with torch.no_grad():
        for x, y in dl_val:
            with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
                logits.append(m(x.to(DEVICE)).float().cpu())
            labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def f1_at(p: np.ndarray, y: np.ndarray, thr: float) -> tuple[float, float, float]:
    pred = p >= thr
    tp = float((pred & (y == 1)).sum()); fp = float((pred & (y == 0)).sum()); fn = float((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return (2 * prec * rec / (prec + rec) if prec + rec else 0.0), prec, rec


def softmax1(logits: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = logits / temp
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(1)


out_dir = Path(CFG["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
best_f1, history = -1.0, []
for epoch in range(CFG["epochs"]):
    net.train()
    t0, total, n = time.time(), 0.0, 0
    for x, y in dl_train:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE)
        with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
            loss = loss_fn(net(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); sched.step()
        total += loss.item() * len(y); n += len(y)
    lg, yv = evaluate(net)
    p = softmax1(lg)
    f1, prec, rec = f1_at(p, yv, 0.5)
    acc = float(((p >= 0.5) == (yv == 1)).mean())
    history.append({"epoch": epoch + 1, "train_loss": total / max(1, n), "val_acc": acc, "val_f1": f1})
    print(f"epoch {epoch + 1}/{CFG['epochs']}  loss {total / max(1, n):.4f}  val acc {acc:.3f}  "
          f"F1 {f1:.3f} (P {prec:.3f} R {rec:.3f})  {time.time() - t0:.0f}s")
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), out_dir / "best.pt")

# %% [markdown]
# ## 6 · Calibrate
# Temperature scaling on the validation set makes the score behave like a probability. The
# operating threshold is the one that maximises validation F1.

# %%
model.load_state_dict(torch.load(out_dir / "best.pt", map_location=DEVICE))
lg, yv = evaluate(model)
t_param = torch.zeros(1, requires_grad=True)  # log-temperature
lg_t, y_t = torch.from_numpy(lg), torch.from_numpy(yv).long()
lbfgs = torch.optim.LBFGS([t_param], lr=0.1, max_iter=100)


def closure():
    lbfgs.zero_grad()
    loss = nn.functional.cross_entropy(lg_t / t_param.exp(), y_t)
    loss.backward()
    return loss


lbfgs.step(closure)
temperature = float(t_param.exp())
p = softmax1(lg, temperature)
grid = np.linspace(0.05, 0.95, 91)
f1s = [f1_at(p, yv, t)[0] for t in grid]
threshold = float(grid[int(np.argmax(f1s))])
f1, prec, rec = f1_at(p, yv, threshold)
metrics = {"val_videos": int(len(yv)), "temperature": temperature, "threshold": threshold,
           "val_f1": f1, "val_precision": prec, "val_recall": rec,
           "val_acc_at_threshold": float(((p >= threshold) == (yv == 1)).mean()), "history": history}
print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))

# %% [markdown]
# ## 7 · Export
# ONNX with a dynamic batch axis, plus `metadata.json` describing exactly how to preprocess clips.
# Aequitas reads both.

# %%
model_cpu = build_model(CFG["arch"], pretrained=False)
model_cpu.load_state_dict(torch.load(out_dir / "best.pt", map_location="cpu"))
model_cpu.eval()
dummy = torch.zeros(1, 3, CFG["clip_len"], CFG["crop"], CFG["crop"])
onnx_path = out_dir / "violence_clip.onnx"
torch.onnx.export(model_cpu, dummy, str(onnx_path), input_names=["clip"], output_names=["logits"],
                  dynamic_axes={"clip": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)

meta = {
    "name": "violence_clip", "arch": CFG["arch"], "classes": ["non_violence", "violence"],
    "clip_fps": CFG["clip_fps"], "clip_len": CFG["clip_len"], "short_side": CFG["short_side"], "crop": CFG["crop"],
    "color": "RGB", "mean": MEAN.tolist(), "std": STD.tolist(), "layout": "NCTHW",
    "temperature": temperature, "threshold": threshold, "metrics": metrics,
    "sources": SOURCES, "trained_at": time.strftime("%Y-%m-%d %H:%M"), "smoke_test": SMOKE,
}
(out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

try:  # check the export matches PyTorch
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = torch.randn(2, 3, CFG["clip_len"], CFG["crop"], CFG["crop"])
    diff = np.abs(sess.run(None, {"clip": x.numpy()})[0] - model_cpu(x).detach().numpy()).max()
    print(f"ONNX check: max |diff| = {diff:.2e}")
except ImportError:
    print("onnxruntime not installed; skipped ONNX check")

(out_dir / "best.pt").unlink()  # the ONNX file is what Aequitas uses; keeps the download small
shutil.make_archive(str(WORK / "violence_clip"), "zip", out_dir)
print("Download:", WORK / "violence_clip.zip", "→ unzip into Aequitas/models/violence_clip/")
