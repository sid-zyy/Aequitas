# Violence clip model (r3d_18, trained on Kaggle 2026-10-15)

`kaggle/violence_clip.ipynb` on SCVD + Real Life Violence Situations + RWF-2000 (8,058 train / 1,569 val videos).
The ONNX file (127 MB) lives in `models/violence_clip/` (not in git); `metadata.json` here is its copy.

## Validation (1,569 held-out videos from the same datasets)

Accuracy 0.950, F1 0.953, precision 0.955, recall 0.951 at threshold 0.51. The threshold and temperature were
calibrated on this same validation set, so these figures are slightly optimistic.

## Faculty clips: never seen in training (excluded by name)

| Clip | Max score (2 s) | 4 s windows ≥ 0.9 | Verdict |
|---|---|---|---|
| v001 | 0.99 | 1 of 3 | violent ✓ |
| v006 | 0.98 | 2 of 2 | violent ✓ |
| v011 | 1.00 | 4 of 4 | violent ✓ |
| v026 | 1.00 | 4 of 4 | violent ✓ |
| w001 | 1.00 | 2 of 2 | violent ✓ |

**5 of 5** (n = 5, indicative only). Gemma 4 E2B on its own: 0 of 5. A sixth faculty clip added later (v003, a street
fight) scores 0.98.

## Unseen real CCTV: Surveillance Camera Fight Dataset (2026-10-16)

300 YouTube CCTV clips (Akti et al. 2019, MIT; 150 fight / 150 noFight, ~2 s each), not in any training set.
`scripts/violence_survfight.py`: 4 fps, 2 s windows, the clip's max. Neither threshold was tuned on this data.

| Threshold | Fight recall | False alarms (noFight) | Precision | Accuracy |
|---|---|---|---|---|
| 0.51 (calibrated) | 0.93 | 67/150 (45 %) | 0.68 | 0.74 |
| 0.9 (clip model alone) | 0.77 | 21/150 (14 %) | 0.85 | 0.81 |

ROC AUC 0.89. **Out of distribution it's far weaker than the 0.95 validation accuracy.** At the calibrated
threshold it flags almost half of the normal clips, many of which show people moving close together. This is why it
must stay a second signal: it votes with the VLM, and needs 0.9 to count on its own.

## False alarms on our street footage (17 normal Bengaluru street clips, 123 four-second windows)

| Threshold | Windows flagged |
|---|---|
| 0.51 (calibrated) | 11 of 123 (9 %): busy MG Road / Brigade Road crowds, 0.72-0.92 |
| 0.9 | 2 of 123 (1.6 %) |

Other clips: punching 10/10 windows ≥ 0.51 (correct); arrest-1 and arrest-4 score high (a struggle looks
violent); fire-rescue max 0.75; accident 0.05.

## How the app uses it (aequitas/understand/fusion.py `fuse_window`)

- VLM and clip model both see violence (≥ 0.51): kept, marked "both".
- VLM named another incident (arrest, person_down, ...): its label is kept. The clip model does not turn
  an arrest into "fighting".
- Only the clip model sees violence: "fighting" only at ≥ 0.9 (`models.violence_alone_threshold`).
  **This 0.9 was chosen by looking at our demo street clips and the faculty clips, so it is a demo setting,
  not a measured accuracy.** A fair estimate needs new, unseen footage.
- A score ≥ 0.35 also sends the window to the bigger re-checking model when the cascade is on.
