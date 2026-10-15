# Kaggle notebooks

## Run first: `ablation.ipynb`, model × precision study (no training)

3 VLMs (Gemma 4 E2B, Qwen3-VL-2B, SmolVLM2-2.2B) × FP16 / W8A8-INT8 / W4A16 on vLLM, plus MTP and
n-gram speculation on Gemma's best precision. It measures latency, stage-2 accuracy, real-clip accuracy and
memory, and outputs `ablation.zip` with `ABLATION.md` (table) and `ablation.png` (chart) for the slides.
**Steps, budget and decision rule: [docs/ABLATION_PLAN.md](../docs/ABLATION_PLAN.md) §7.** About 3–4 GPU hours on
GPU T4 x2.

## Which model is most accurate: `model_compare.ipynb` (runs in the background)

Runs our real two-stage pipeline with **gemma4:e2b, gemma4:e4b-it-qat, gemma4:12b-it-qat, qwen3-vl:4b and
qwen3-vl:8b** on SCVD (Ollama on the T4, the same backend as the demo laptop), and works out the cascade
combinations (fast model + bigger re-checker) from the saved results without extra GPU time.

1. Import `model_compare.ipynb`; **GPU T4 x2**, Internet **On**.
2. **Add Input**: "Smart-City CCTV Violence Detection Dataset (SCVD)".
3. **Save Version → Save & Run All**. About 1.5-3 h (model downloads + 60 clips per model in QUICK mode).
4. Download `model_compare.zip` → `results/model_compare/`. `COMPARE.md` + `compare.png` (accuracy vs time) for
   the slides; the winner goes into `config/settings.yaml` (`vlm.model`, or `escalation.model` for the cascade).

## Fine-tune Gemma 4 E2B on 12B's summaries: `finetune_gemma.ipynb`

12B caught 70 % of SCVD incidents, E2B 15 % (`model_compare`). 12B summarises the training clips with our real
prompts; summaries whose label agrees with the dataset label teach E2B (QLoRA, Unsloth). Stage 2 is unchanged.
Tested on the same 60 held-out SCVD clips as `model_compare`, against base E2B in the same run.

1. Import `finetune_gemma.ipynb`; **GPU T4 x2**, Internet **On**.
2. **Add Input**: SCVD (needed, it is the test set), RWF-2000, Real Life Violence Situations; optionally a
   UCF-Crime video dataset with class folders (adds arrest, fire, accident, theft).
3. **Save Version → Save & Run All**. About 3-4 h.
4. Output tab → download `finetune_gemma.zip` (FINETUNE.md, chart, LoRA adapter) → `results/finetune/`, and
   `aequitas-e2b-ollama.tar` (~3-4 GB, the model already in Ollama's format).
5. On the laptop (Ollama closed from the tray first):
   ```bash
   tar -xf %USERPROFILE%\Downloads\aequitas-e2b-ollama.tar -C %USERPROFILE%\.ollama\models
   ollama list
   ```
   `aequitas-e2b` appears. To use it set `vlm: {model: aequitas-e2b}` in `config/settings.yaml`, then rerun the
   demo set (`config/eval_demo_final.yaml`) to compare with base E2B on Bengaluru footage.
   If the notebook says the Ollama import failed, there is no tar: the LoRA adapter in the zip still works with
   vLLM (`--enable-lora`) on WSL, and the results were measured in transformers.

## Improve fight / weapon detection: train `violence_clip.ipynb` on CCTV data

Why: on the faculty's sample clips (v001-v026, w001) Gemma's summaries are too vague to label ("people
gathered in an outdoor area"): 0/5 caught. A 3D CNN that watches 2 s of motion is the second signal; in live
mode a confident score upgrades a window to `fighting` (aequitas/understand/fusion.py `fuse_window`).

1. Import `violence_clip.ipynb`; Accelerator **GPU T4 x2**, Internet **On**.
2. **Add Input** (search these names):
   - **"Smart-City CCTV Violence Detection Dataset (SCVD)"**: real CCTV with Violence / Weaponized / NonViolence;
     the faculty's clips look like this dataset.
   - **"RWF-2000"** (real CCTV fights) and/or **"Real Life Violence Situations"**.
   Every attached dataset is used; labels come from folder names. The 5 faculty clips are excluded by name.
3. **Save Version → Save & Run All**. About 1-2 h on a T4.
4. Download `violence_clip.zip` → unzip into `Aequitas/models/violence_clip/`, then in `config/settings.yaml`:
   `models: {violence_dir: models/violence_clip}`.
5. Held-out check on the faculty clips:
   `.venv/Scripts/python scripts/eval_violence.py --model models/violence_clip --videos data/footage/faculty`

## Training notebooks

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
