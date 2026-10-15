# %% [markdown]
# # Aequitas · hazard detector (fire, smoke, weapons)
#
# Fine-tunes YOLO11 to find **fire, smoke, knives, guns and sticks/lathis/rods**. It runs alongside
# Aequitas's normal person detector, so people and bags keep using the COCO model.
#
# Training images are partly **degraded to look like CCTV** (low resolution, JPEG artefacts) because
# that is where Aequitas's detectors struggled.
#
# **Kaggle setup**
# 1. Settings → Accelerator → **GPU T4 x2** or P100. Internet **on** (installs `ultralytics`).
# 2. *Add data* (YOLO or Pascal-VOC format), for example:
#    - **D-Fire** (fire + smoke, YOLO format)
#    - a knife / pistol dataset (e.g. OD-WeaponDetection, Knife-vs-Pistol)
#    - your own **lathi / stick / rod** images labelled in CVAT or Label Studio (export YOLO)
# 3. Edit `SOURCES` (paths and, where a dataset has no `data.yaml`, its class index map), then *Run all*.
# 4. Download `/kaggle/working/hazard_detector.zip` and unzip it into Aequitas's `models/` folder.
#
# Licences: check each dataset's terms before use.

# %%
import json
import os
import random
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

SMOKE = os.environ.get("AEQUITAS_SMOKE") == "1"
WORK = Path(os.environ.get("AEQUITAS_WORK", "/kaggle/working"))
try:
    import ultralytics  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)
from ultralytics import YOLO

# %% [markdown]
# ## Configuration
# Each source is a folder of images plus labels. Class names are read from the source's `data.yaml`
# when it has one; otherwise give `index_map` (source class index → name). Names are mapped onto
# Aequitas's classes through `ALIASES`; anything unmapped is dropped.
#
# D-Fire's class order is 0 = smoke, 1 = fire (checked against the dataset page).

# %%
CLASSES = ["fire", "smoke", "knife", "gun", "stick"]
ALIASES = {
    "fire": "fire", "flame": "fire", "flames": "fire",
    "smoke": "smoke",
    "knife": "knife", "knives": "knife", "blade": "knife", "sword": "knife", "machete": "knife",
    "gun": "gun", "pistol": "gun", "handgun": "gun", "firearm": "gun", "rifle": "gun", "revolver": "gun",
    "stick": "stick", "lathi": "stick", "rod": "stick", "iron_rod": "stick", "baton": "stick", "bat": "stick",
    "club": "stick",
    "fire_and_smoke": None, "weapon": None,  # too generic for one of our classes: dropped
}
# Datasets are found automatically under /kaggle/input (Kaggle's layout changed to
# /kaggle/input/datasets/<owner>/<name>, so fixed paths break). A dataset root is any folder with a
# data.yaml / *.yaml that lists class `names`. Datasets without one need an entry in KNOWN_INDEX_MAPS
# (matched on the folder path); anything else is skipped with a warning.
KNOWN_INDEX_MAPS = {
    "smoke-fire-detection-yolo": {0: "smoke", 1: "fire"},  # D-Fire repackaged (sayedgamal99), 0 = smoke
    "d-fire": {0: "smoke", 1: "fire"},
}


def discover_sources(base: Path = Path("/kaggle/input")) -> list[dict]:
    roots = []
    for y in sorted(base.rglob("*.yaml")):
        try:
            names = (yaml.safe_load(y.read_text(errors="ignore")) or {}).get("names")
        except Exception:
            continue
        if names and not any(r in y.parents for r in roots):
            roots.append(y.parent)
    out = [{"path": str(r)} for r in roots]
    for top in sorted({p for p in base.rglob("*") if p.is_dir() and p.name in ("images", "labels")}):
        root = top.parent.parent if top.parent.name.lower() in ("train", "valid", "val", "test") else top.parent
        if any(Path(o["path"]) in [root, *root.parents] for o in out):
            continue
        key = next((k for k in KNOWN_INDEX_MAPS if k in str(root).lower()), None)
        if key:
            out.append({"path": str(root), "index_map": KNOWN_INDEX_MAPS[key]})
        else:
            print(f"WARNING: {root} has no data.yaml and no known class map: skipped")
    return out


SOURCES = discover_sources() if Path("/kaggle/input").exists() else []
CFG = dict(
    model="yolo11s.pt", imgsz=640, epochs=40, batch=-1, patience=12,
    val_frac=0.1, degrade_frac=0.3, max_negative_frac=0.25, seed=0,
    data_dir=str(WORK / "hazard_ds"), out_dir=str(WORK / "hazard_detector"),
)

# %% [markdown]
# ## Smoke-test data (only when `AEQUITAS_SMOKE=1`)

# %%
def make_synthetic(root: Path, n: int = 24) -> None:
    rng = np.random.default_rng(0)
    for split in ("train", "valid"):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        for i in range(n if split == "train" else 6):
            img = np.full((240, 320, 3), 90, np.uint8)
            lines = []
            for _ in range(int(rng.integers(1, 3))):
                c = int(rng.integers(0, 2))
                x, y = int(rng.integers(20, 240)), int(rng.integers(20, 160))
                w, h = int(rng.integers(30, 70)), int(rng.integers(30, 70))
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 80, 255) if c == 0 else (200, 200, 200), -1)
                lines.append(f"{c} {(x + w / 2) / 320:.4f} {(y + h / 2) / 240:.4f} {w / 320:.4f} {h / 240:.4f}")
            cv2.imwrite(str(root / split / "images" / f"{i}.jpg"), img)
            (root / split / "labels" / f"{i}.txt").write_text("\n".join(lines))
    (root / "data.yaml").write_text(yaml.safe_dump({"names": ["fire", "smoke"]}))


if SMOKE:
    synth = WORK / "synthetic_hazards"
    if not synth.exists():
        make_synthetic(synth)
    SOURCES = [{"path": str(synth)}]
    local = Path(os.environ.get("AEQUITAS_YOLO", "yolo11n.pt"))
    CFG.update(model=str(local), imgsz=160, epochs=1, batch=4)

random.seed(CFG["seed"])
print("sources:", [s["path"] for s in SOURCES])

# %% [markdown]
# ## 1 · Collect and merge
# Finds image/label pairs in YOLO layout (`images/…` with `labels/…` alongside) or Pascal-VOC XML,
# maps class names onto `CLASSES`, and writes one merged YOLO dataset.

# %%
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET = {n: i for i, n in enumerate(CLASSES)}


def source_names(root: Path, src: dict) -> dict[int, str]:
    if "index_map" in src:
        return {int(k): v for k, v in src["index_map"].items()}
    for y in list(root.rglob("data.yaml"))[:1] + list(root.rglob("*.yaml"))[:1]:
        names = (yaml.safe_load(y.read_text()) or {}).get("names")
        if isinstance(names, dict):
            return {int(k): v for k, v in names.items()}
        if isinstance(names, list):
            return dict(enumerate(names))
    return {}


def yolo_label_for(img: Path) -> Path | None:
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            cand = Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
            if cand.exists():
                return cand
    same = img.with_suffix(".txt")
    return same if same.exists() else None


def voc_label_for(img: Path) -> list[tuple[str, float, float, float, float]] | None:
    xml = img.with_suffix(".xml")
    if not xml.exists():
        alt = list(img.parent.parent.glob(f"Annotations/{img.stem}.xml"))
        xml = alt[0] if alt else None
    if xml is None:
        return None
    r = ET.parse(xml).getroot()
    w, h = float(r.findtext("size/width") or 0), float(r.findtext("size/height") or 0)
    if not w or not h:
        im = cv2.imread(str(img))
        h, w = im.shape[:2]
    out = []
    for o in r.findall("object"):
        b = o.find("bndbox")
        x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        out.append((o.findtext("name", ""), (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
    return out


def split_hint(p: Path) -> str | None:
    parts = {x.lower() for x in p.parts}
    if parts & {"val", "valid", "validation", "test"}:
        return "val"
    return "train" if "train" in parts else None


records, dropped = [], Counter()
for src in SOURCES:
    root = Path(src["path"])
    if not root.exists():
        print(f"WARNING: {root} not found, skipped")
        continue
    names = source_names(root, src)
    for img in root.rglob("*"):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        boxes = []
        lab = yolo_label_for(img)
        if lab is not None:
            for line in lab.read_text().splitlines():
                v = line.split()
                if len(v) >= 5:
                    name = names.get(int(float(v[0])), str(v[0]))
                    boxes.append((name, *map(float, v[1:5])))
        else:
            voc = voc_label_for(img)
            if voc is None:
                continue
            boxes = voc
        mapped = []
        for name, *xywh in boxes:
            tgt = ALIASES.get(str(name).strip().lower().replace(" ", "_"))
            if tgt is None:
                dropped[str(name)] += 1
                continue
            mapped.append((TARGET[tgt], *xywh))
        records.append({"img": img, "boxes": mapped, "split": split_hint(img.relative_to(root))})

negatives = [r for r in records if not r["boxes"]]
positives = [r for r in records if r["boxes"]]
keep_neg = int(CFG["max_negative_frac"] * len(positives))
random.shuffle(negatives)
records = positives + negatives[:keep_neg]
for r in records:
    if r["split"] is None:
        r["split"] = "val" if random.random() < CFG["val_frac"] else "train"
counts = Counter(CLASSES[b[0]] for r in records for b in r["boxes"])
print(f"{len(records)} images ({len(positives)} with boxes, {min(keep_neg, len(negatives))} background)")
print("boxes per class:", dict(counts))
if dropped:
    print("dropped labels (no alias):", dict(dropped.most_common(10)))
assert positives, "No usable labelled images: check SOURCES, index_map and ALIASES"

# %% [markdown]
# ## 2 · Write the merged dataset, degrading part of the training set
# Degradation: downscale to 288–360 px tall, upscale back, JPEG quality 25–60. Labels are unchanged
# because box coordinates are normalised.

# %%
def degrade(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    th = random.randint(288, 360)
    if h > th:
        s = th / h
        small = cv2.resize(img, (max(8, int(w * s)), th), interpolation=cv2.INTER_AREA)
        ok, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, random.randint(25, 60)])
        small = cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else small
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return img


ds = Path(CFG["data_dir"])
shutil.rmtree(ds, ignore_errors=True)
for s in ("train", "val"):
    (ds / "images" / s).mkdir(parents=True)
    (ds / "labels" / s).mkdir(parents=True)
n_degraded = 0
for k, r in enumerate(records):
    stem = f"{k:07d}"
    img = cv2.imread(str(r["img"]))
    if img is None:
        continue
    if r["split"] == "train" and random.random() < CFG["degrade_frac"]:
        img = degrade(img)
        n_degraded += 1
    cv2.imwrite(str(ds / "images" / r["split"] / f"{stem}.jpg"), img)
    (ds / "labels" / r["split"] / f"{stem}.txt").write_text(
        "\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in r["boxes"]))
data_yaml = ds / "data.yaml"
data_yaml.write_text(yaml.safe_dump({"path": str(ds), "train": "images/train", "val": "images/val",
                                     "names": dict(enumerate(CLASSES))}))
print(f"dataset at {ds}; {n_degraded} training images degraded")

# %% [markdown]
# ## 3 · Train

# %%
import torch

device = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else "cpu"
t0 = time.time()
model = YOLO(CFG["model"])
model.train(data=str(data_yaml), imgsz=CFG["imgsz"], epochs=CFG["epochs"], batch=CFG["batch"],
            patience=CFG["patience"], project=str(WORK / "runs"), name="hazard", exist_ok=True,
            cos_lr=True, close_mosaic=min(10, CFG["epochs"]), seed=CFG["seed"], plots=not SMOKE,
            device=device, workers=0 if SMOKE else 8, verbose=False)
print(f"trained in {(time.time() - t0) / 60:.1f} min")

# %% [markdown]
# ## 4 · Evaluate and export
# Per-class mAP on the validation split, then `best.pt` + ONNX + `metadata.json`.

# %%
best = WORK / "runs" / "hazard" / "weights" / "best.pt"
trained = YOLO(str(best))
val = trained.val(data=str(data_yaml), imgsz=CFG["imgsz"], device=device, plots=False, verbose=False)
per_class = {}
for i, name in enumerate(CLASSES):
    if i in list(val.box.ap_class_index):
        j = list(val.box.ap_class_index).index(i)
        per_class[name] = {"mAP50": float(val.box.ap50[j]), "mAP50-95": float(val.box.ap[j])}
metrics = {"mAP50": float(val.box.map50), "mAP50-95": float(val.box.map), "per_class": per_class,
           "boxes_per_class": dict(counts)}
print(json.dumps(metrics, indent=2))

out = Path(CFG["out_dir"])
shutil.rmtree(out, ignore_errors=True)
out.mkdir(parents=True)
shutil.copy(best, out / "hazard_detector.pt")
onnx_file = trained.export(format="onnx", imgsz=CFG["imgsz"], dynamic=False, simplify=False)
shutil.copy(onnx_file, out / "hazard_detector.onnx")
(out / "metadata.json").write_text(json.dumps({
    "name": "hazard_detector", "classes": CLASSES, "imgsz": CFG["imgsz"], "base_model": CFG["model"],
    "metrics": metrics, "sources": [s["path"] for s in SOURCES], "degrade_frac": CFG["degrade_frac"],
    "trained_at": time.strftime("%Y-%m-%d %H:%M"), "smoke_test": SMOKE,
}, indent=2))
shutil.make_archive(str(WORK / "hazard_detector"), "zip", out)
print("Download:", WORK / "hazard_detector.zip", "→ unzip into Aequitas/models/hazard_detector/")
