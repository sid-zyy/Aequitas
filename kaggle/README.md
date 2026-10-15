# Kaggle training notebooks

Two models that fix what the CAVIAR real-footage test exposed (see docs/ROADMAP.md):

| Notebook | Trains | Why | Output |
|---|---|---|---|
| `violence_clip.ipynb` | 3D ResNet (r3d_18, Kinetics pretrained) scoring a 2 s clip as violence / not | Fight rules need two people tracked close together, which fails on low-res CCTV. This model needs no tracking. | `violence_clip.zip` → `models/violence_clip/` |
| `hazard_detector.ipynb` | YOLO11s for fire, smoke, knife, gun, stick (lathi/rod), with part of the training set degraded to CCTV quality | No fire or weapon detection existed yet | `hazard_detector.zip` → `models/hazard_detector/` |

## Running on Kaggle

1. On kaggle.com: **Create → New Notebook → File → Import Notebook** and upload the `.ipynb`.
2. **Settings**: Accelerator **GPU T4 x2** (or P100), Internet **On**.
3. **Add data**: search Kaggle for the datasets and add them.
   - Violence: RWF-2000 (`Fight`/`NonFight` folders) or Real Life Violence Situations (`Violence`/`NonViolence`).
     Optional: your own hard negatives (festivals, sports, dancing, processions) in a folder named `NonViolence`.
   - Hazards: D-Fire (fire and smoke), a knife/pistol set, and your own lathi/stick/rod images labelled in
     CVAT or Label Studio (export as YOLO).
4. Check the paths in the `SOURCES` cell. They must match `/kaggle/input/<dataset-folder>`.
   **D-Fire:** confirm its class order (`index_map`) in the dataset README.
5. **Run all.** Expect about 1–2 h for the violence model and 2–4 h for the detector on a T4.
   Kaggle's weekly GPU quota is limited (check your quota page).
6. Download the zip from the **Output** panel and unzip it into Aequitas:

```
Aequitas/models/violence_clip/     violence_clip.onnx, metadata.json
Aequitas/models/hazard_detector/   hazard_detector.pt, hazard_detector.onnx, metadata.json
```

7. Turn them on in `config/settings.yaml`:

```yaml
models:
  violence_dir: models/violence_clip
  hazard_dir: models/hazard_detector
```

## Check the notebooks locally first (no GPU, about 1 minute)

```bash
set AEQUITAS_SMOKE=1
set AEQUITAS_WORK=%TEMP%\aequitas_kaggle_smoke
.venv\Scripts\python kaggle\violence_clip.py
.venv\Scripts\python kaggle\hazard_detector.py
```

Smoke mode trains on tiny synthetic clips and images, just to prove the whole notebook runs. Its
"accuracy" means nothing.

## Editing

The `.py` files are the source; the `.ipynb` files are generated from them:

```bash
.venv\Scripts\python kaggle\build_notebooks.py
```

## Licences

Most violence datasets (including RWF-2000) are research-only. D-Fire and the weapon sets have their own
terms. Check each one before use, and record what you trained on: `metadata.json` stores the source list.
