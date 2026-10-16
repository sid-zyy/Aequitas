# Models of Gemma 4 E2B's size, compared on SCVD (Kaggle T4, Ollama)

2026-10-16. Notebook: `kaggle/model_compare.py` with `SET = "small"`. Raw output: `COMPARE.md`, `compare.csv`,
`rows-*.json` (one row per clip), `reports/`, logs.

## Setup

- **Models:** six of about 2–4 B parameters, 4-bit in Ollama. Each fits the demo laptop's 6 GB GPU.
- **Pipeline:** our real two-stage pipeline, as in the app: the VLM summarises a 2×2 keyframe grid, then the
  same model labels the summary.
- **What is not included:** the violence model (r3d-18), which the app fuses with the VLM. So this measures the
  VLM alone.
- **Clips:** 60 SCVD clips (20 Violence, 20 Weaponized, 20 NonViolence, seed 0). The faculty clips are excluded,
  and none was in the sample (checked by file name). The faculty clips are SCVD's `v001`/`v003`/... files.
- **Clips vs the first run:** 46 of the 60 are the same clips as the first comparison (`results/model_compare`).
  Excluding v003 too shifted the draw by one file in a few folders.
- **Times:** per whole clip on a Kaggle T4, not on the laptop.

## Results

An "alert" is any label other than `none`, which is what raises a concern in the app.

| Model | Alert on incident clips (recall) | False alarms on normal clips | Balanced accuracy | Incident clips given a violent label¹ | Time per clip (p50) |
|---|---|---|---|---|---|
| **qwen3-vl:2b-instruct** | **0.70** (28/40) | 5/20 (25 %) | **0.72** | 0.38 | 2.47 s |
| gemma4:e2b (demo model) | 0.25 (10/40) | **2/20 (10 %)** | 0.57 | 0.00 | **1.82 s** |
| gemma3:4b | 1.00 | 19/20 (95 %) | 0.53 | 0.12 | 3.76 s |
| llava-phi3:3.8b | 0.97 | 20/20 (100 %) | 0.49 | 0.85 | 2.88 s |
| qwen2.5vl:3b | 0.75 | 17/20 (85 %) | 0.45 | 0.40 | 2.92 s (8/60 invalid answers) |
| granite3.2-vision:2b | – | – | – | – | skipped: 0 of 5 probe clips gave a valid answer |

¹ fighting, weapon, harassment or riot on a Violence or Weaponized clip.

The notebook's own `incident_detected_acc` column (in `COMPARE.md`) ranks gemma3:4b first. That number is
misleading: 40 of the 60 clips are incidents, so a model that flags everything scores 0.67. gemma3:4b and
llava-phi3 flag almost every normal clip too. Balanced accuracy (the mean of recall and 1 − false-alarm rate)
corrects for this.

**E2B against the first run:** recall 0.25 here vs 0.15, false alarms 2/20 vs 1/20. 40 of the 60 predictions are
identical between runs; the 14 changed clips account for part of the difference.

## What it means

- **No model of this size is clearly better on every count.**
- **Qwen3-VL 2B (instruct)** is the only one that separates incidents from normal footage. It catches 70 % of
  incident clips against E2B's 25 %. The cost is more false alarms (25 % vs 10 %) and 36 % more time per clip,
  and the faculty weight time most heavily.
- **Gemma 3 4B, LLaVA-Phi3 and Qwen2.5-VL 3B** call almost everything an incident, which is useless.
- **Gemma 4 E2B** is the fastest and the most cautious, but on its own it misses most incidents. It never gave a
  violent label to an incident clip here: it says chase, person down or none. That is why the app never lets it
  decide alone: the violence model supplies the recall (77 % of fights on 300 unseen real CCTV clips at 0.9;
  6/6 faculty clips flagged).
- **Decision for the demo:** keep gemma4:e2b. It is the fastest, has the fewest false alarms, and its weakness is
  covered by the violence model.
- **Worth testing next:** qwen3-vl:2b-instruct in the full app pipeline (with the violence model and the scene
  gate) on the 21-clip Bengaluru demo set, timed on the laptop, before considering a switch.

## Limits

- 60 clips: one clip is 2.5 points of recall, and the false-alarm figures rest on 20 normal clips.
- SCVD labels whole clips, and the scoring takes one label per clip.
- The VLM was tested alone; the app's results come from VLM + violence model.
- T4 times; laptop times differ (E2B: about 0.6 s summary + 0.4 s label per 4 s window on the RTX 4050).
