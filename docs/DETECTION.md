# Detection: which person detector for triggered mode

Measured 2026-10-16 on the RTX 4050 laptop with `scripts/detector_compare.py`. Raw tables:
`data/reports/detector_compare.md` (local, not in git).

## Why this matters, and where it doesn't

- **Only triggered mode uses a detector** (`analysis.mode: triggered`: YOLO → tracker → rules → VLM). Periodic
  mode (live CCTV, uploads, bodycam: the main demo) does not, so nothing here changes its latency or accuracy.
- Triggered mode found **0 of 6 CAVIAR incidents** (docs/EVAL_CAVIAR.md) because YOLO11n misses small, high-angle
  people. docs/TRACKING.md showed the detector, not the tracker, is the bottleneck.

## What was compared

COCO-pretrained, no fine-tuning, person class only, FP16 (except RF-DETR, below), batch 1, confidence floor 0.01.

| Family | Models | Source | Licence |
|---|---|---|---|
| YOLO11 (today) | n @ 640, m @ 960 | Ultralytics | AGPL-3.0 |
| **YOLO26** (NMS-free, small-object loss) | n, s, m @ 640; s, m, l @ 960 | Ultralytics (in 8.4) | AGPL-3.0 |
| RT-DETR | l @ 640 | Ultralytics | AGPL-3.0 (Ultralytics port) |
| **RF-DETR** (DINOv2 backbone, ICLR 2026) | Nano 384, Small 512, Medium 576, Large 704 | `rfdetr` | Apache-2.0 (N/S/M/L) |
| **D-FINE** (Objects365 → COCO) | S, M, L, X | Hugging Face transformers | Apache-2.0 |
| DEIMv2 (DINOv3) | S, L | Hugging Face transformers (community ports) | Apache-2.0 |

**Two test sets:**
- **CAVIAR:** 8 clips at 4 fps, 1 048 frames, 2 035 hand-labelled people.
  - These are tiny people: median height 22 px, and 1 429 are under 30 px.
  - Metrics:
    - **AP50** and AP30 against the labels. These are threshold-free, so nothing was tuned.
    - **Recall** at the eval config's confidence (0.10), split by height.
    - **IDF1** and ID switches with our `CameraTracker` at the eval config's thresholds.
- **Faculty sample clips** (`data/footage/faculty`, 5 clips at 4 fps, 137 frames, 1280×720). These are
  medium-range CCTV with crowds, occlusion, people on the floor and a dark corridor.
  - There are no labels, so the pseudo-labels are the boxes on which D-FINE-X and RF-DETR-L agree (625 people).
  - This column measures agreement with two DETRs, not accuracy, and is biased towards DETRs. Use it as a
    sanity check only.

**Latency:** wall time per frame for the whole call (pre-processing, inference, post-processing, copy back). It's
the mean over the 137 faculty frames after warm-up, in eager PyTorch with no TensorRT.

## Results

| Detector | CAVIAR AP50 | Recall < 30 px | Recall ≥ 30 px | IDF1 | ID sw | Faculty AP50 (pseudo) | ms/frame | GPU MiB |
|---|---|---|---|---|---|---|---|---|
| **yolo11n 640 (today)** | 0.102 | 0.030 | 0.434 | 0.051 | 10 | 0.71 | **11.8** | 46 |
| yolo11m 960 | 0.241 | 0.087 | 0.624 | 0.144 | 13 | 0.50 | 16.0 | 141 |
| yolo26n 640 | 0.111 | 0.036 | 0.487 | 0.076 | 8 | 0.74 | 13.3 | 60 |
| yolo26s 640 | 0.200 | 0.118 | 0.660 | 0.132 | 23 | 0.76 | 14.1 | 69 |
| yolo26s 960 | 0.230 | 0.176 | 0.815 | 0.160 | 27 | 0.54 | 15.3 | 87 |
| yolo26m 640 | 0.290 | 0.157 | 0.756 | 0.229 | 27 | 0.72 | 19.6 | 112 |
| yolo26m 960 | 0.313 | 0.195 | 0.850 | 0.250 | 41 | 0.52 | 23.0 | 140 |
| yolo26l 960 | 0.344 | 0.195 | 0.830 | 0.252 | 32 | 0.64 | 29.6 | 160 |
| rtdetr-l 640 | 0.363 | 0.458 | 0.921 | 0.213 | 38 | 0.89 | 26.4 | 157 |
| rfdetr nano | 0.399 | 0.419 | 0.970 | 0.226 | 34 | 0.90 | 25.2 | 184 |
| **rfdetr small** | **0.425** | 0.484 | 0.969 | 0.220 | 39 | 0.93 | **25.8–28.5** | 209 |
| **rfdetr medium** | **0.475** | 0.514 | 0.965 | **0.265** | 51 | 0.94 | **31.7** (first run 58.0) | 228 |
| rfdetr large* | 0.470 | 0.491 | 0.977 | 0.264 | 38 | (ref) | 64.9 | 253 |
| d-fine-s | 0.386 | **0.726** | 0.983 | 0.133 | 68 | 0.88 | 52.6 | 100 |
| d-fine-m | 0.490 | 0.632 | 0.990 | 0.225 | 70 | 0.91 | 52.4 | 124 |
| d-fine-l | 0.488 | 0.616 | **0.993** | 0.257 | 55 | 0.91 | 63.6 | 152 |
| d-fine-x* | **0.522** | **0.754** | 0.987 | 0.195 | 64 | (ref) | 74.5 | 232 |
| deimv2-l (community port) | 0.257 | 0.477 | 0.878 | 0.125 | 68 | 0.34 | 67.4 | 134 |

\* pseudo-label source. DEIMv2-S failed to load (the community checkpoint is malformed). DEIMv2-L scored far below
its paper (56.0 COCO AP), so the port is probably wrong. Treat both as **not tested**, not as bad models.

## What the numbers say

1. **Transformer detectors (DETRs) are a different class on small people.**
   - Every DETR beats every YOLO on CAVIAR AP50: 0.36–0.52 against 0.10–0.34.
   - Under 30 px, the DETRs find 42–75 % of people; the best YOLO finds 20 %, and today's model 3 %.
   - On people of 30 px and taller, the DETRs find 92–99 %, against 43 % for today's model.
   - This fits the papers: RF-DETR and D-FINE report higher small-object AP than YOLO at equal latency
     (DEIMv2 paper, Table 3).
2. **YOLO26 is better than YOLO11 at equal size, but not enough.**
   - yolo26m at 640 roughly matches yolo11m at 960 (AP50 0.290 vs 0.241) and is faster.
   - Even yolo26l at 960 stays below RF-DETR-Nano.
3. **Today's detector is the weakest option measured.** It's 4.2× below RF-DETR-Small on AP50, and finds 16× fewer
   small people.
4. **IDF1 rises 4–5× (0.05 → 0.22–0.27), but the tracker isn't tuned for DETRs.**
   - The DETRs produce more low-confidence boxes, so they cause more ID switches at our YOLO-tuned thresholds
     (activation 0.15, `fuse_score` on).
   - Retune those on data we didn't evaluate on before reading anything into the DETR IDF1 differences.
5. **The faculty-sample column ranks the families the same way, but don't over-read it.**
   - YOLOs at 960 px score lower than at 640 px there, the opposite of CAVIAR. At 960 they find extra people
     that the pseudo-labels (both references at ≥ 0.5 confidence) don't include, and those count as false
     positives.
   - Real labels from the faculty dataset are needed.
6. **Latency today is eager-PyTorch overhead, not compute.**
   - RF-DETR Nano, Small and Medium all take 24–32 ms per call, and Nano's bare FP16 forward pass alone takes
     12.4 ms. The time goes on launching many small GPU kernels and on Python pre-processing.
   - Published TensorRT FP16 numbers on a T4 are 2.3 / 3.5 / 4.4 / 6.8 ms for N / S / M / L.
   - FP16 autocast did not help (24 → 30 ms on Nano). CUDA-graph capture fails inside RF-DETR (a CPU→GPU copy
     mid-forward).
   - D-FINE is slower than RF-DETR here (52–75 ms) mostly because of the Hugging Face image processor.
   - The papers put D-FINE-M at ~6 ms with TensorRT. With an exported runtime, D-FINE-M and RF-DETR-M should
     both fit well under 10 ms.
   - The first full run timed RF-DETR-Medium at 58 ms, and a rerun gave 31.7 ms. Laptop GPU timing varies
     between runs (clocks, heat), so quote the rerun range.
7. **GPU memory is not a constraint.** 100–250 MiB, against the ~5 GB Gemma needs on the same 6 GB GPU.

## Real violence CCTV, hand-labelled (2026-10-16)

`scripts/detector_label_check.py`. The frames are:
- 2 frames from each of the 6 faculty clips, including v003, the new street-fight clip.
- 1 middle frame from each of 30 clips of the **Surveillance Camera Fight Dataset** (Akti et al. 2019, MIT; 300
  real YouTube CCTV clips, 150 fight / 150 noFight, downloaded to `data/videos/survfight`, 86 MB). We drew
  20 fight and 10 noFight clips.

**How the labels were made:**
- Every detector's boxes at ≥ 0.25 confidence were pooled and merged, giving 501 candidates. Each was judged on
  the full frame as "one person's visible extent" or not, and people no detector boxed were counted as missed.
- **33 frames labelled, 118 people (7 missed by every detector).** 9 frames were too crowded or blurred to label
  reliably and were left out (two v001, two v006, one v011, four from the fight dataset).
- 15 frames involve judgement calls (occlusion, blur, two fighters in one box) and are flagged `uncertain`. The
  "clear only" column (18 frames, 46 people) leaves them out.
- The labels were made by hand from candidate boxes. Treat them as approximate, not as an expert ground truth.
  Labels and boxes: `results/detector_labels/` (the frame images stay local).

**Results** (recall / precision, IoU ≥ 0.5, at person confidence **0.4**, as configured for the app):

| Detector | All 118: recall | precision | at 0.25: recall | precision | Clear only (46): recall | precision | Faculty (37): recall | Fight dataset (81): recall |
|---|---|---|---|---|---|---|---|---|
| **yolo11n 640 (in use)** | 0.47 | 0.86 | 0.53 | 0.70 | 0.61 | 0.82 | 0.49 | 0.46 |
| yolo11m 640 | 0.48 | 0.90 | 0.56 | 0.84 | 0.54 | 0.89 | 0.49 | 0.48 |
| yolo11m 960 | 0.42 | 0.86 | 0.52 | 0.81 | 0.54 | 0.96 | 0.27 | 0.48 |
| yolo26m 640 | 0.53 | 0.87 | 0.60 | 0.84 | 0.65 | 0.91 | 0.41 | 0.58 |
| yolo26l 960 | 0.53 | 0.85 | 0.58 | 0.81 | 0.65 | 0.91 | 0.51 | 0.54 |
| rtdetr-l 640 | 0.67 | 0.84 | 0.75 | 0.60 | 0.76 | 0.90 | 0.81 | 0.60 |
| **rfdetr small** | **0.70** | 0.87 | 0.81 | 0.66 | **0.78** | 0.90 | **0.84** | 0.64 |
| **rfdetr medium** | **0.72** | **0.93** | 0.81 | 0.67 | **0.78** | **0.95** | 0.81 | **0.68** |
| d-fine-m | 0.69 | 0.68 | 0.85 | 0.39 | 0.74 | 0.74 | 0.78 | 0.65 |

**What it says:**
- **The ranking holds on real violence.**
  - RF-DETR finds 70–72 % of people, against 47 % for yolo11n and 48 % for yolo11m, at the same or better
    precision.
  - On the faculty clips it's 0.84 against 0.49.
  - The clear-only frames give the same order: 0.78 against 0.54–0.61.
- **YOLO11m is not better than YOLO11n here** (0.48 vs 0.47).
  - At 960 px it's worse on the faculty clips (0.27). These are 720p frames with large people, so upscaling
    doesn't help, and bigger-input YOLO drops large, blurred, oddly posed bodies.
  - YOLO11m isn't what the app uses either: `config/settings.yaml` has `yolo11n.pt` at 640.
- **Violent scenes are hard even when people are large.** Fighters grapple, fall, blur and overlap.
  - Every YOLO misses about half of them.
  - In several fights (e.g. fi092), most detectors put **one box around two grappling people**. The fight rule
    needs two separate tracks, so it can't fire then.
- **D-FINE-M** finds about as many people but has the lowest precision (0.68). Its scores are calibrated
  differently from the others, so 0.4 is too low a threshold for it.
- **Caveats:**
  - **Small sample:** 118 people, one labeller.
  - **Some bias towards these detectors:** the candidate pool is these detectors' own boxes, so people every
    detector missed are counted only when spotted by eye (7).
  - **Crowded scenes excluded:** the crowded frames, where differences are likely largest, aren't scored.

### The violence clip model on the same dataset (all 300 clips, held out from its training)

`scripts/violence_survfight.py`: frames at 4 fps, 2 s windows, max over the clip.

| Threshold | Fight recall | False alarms on noFight | Precision | Accuracy |
|---|---|---|---|---|
| 0.51 (calibrated on its validation set) | **0.93** | **67/150 (45 %)** | 0.68 | 0.74 |
| 0.9 (`models.violence_alone_threshold`) | 0.77 | 21/150 (14 %) | 0.85 | 0.81 |

- ROC AUC is 0.89, and scoring takes 193 ms per clip (CPU ONNX, decoding included).
- Neither threshold was tuned on this data.
- **On unseen real CCTV the model is much weaker than its 0.95 validation accuracy.** At 0.51 it flags almost
  half of the non-fight clips.
- The app already treats it this way:
  - Alone it needs ≥ 0.9.
  - Otherwise it only votes alongside the VLM label (`fuse_window`).
  - On this data, that 0.9 gives a 14 % false-alarm rate per clip.
- The new faculty clip v003 scores 0.98, so the model calls all 6 faculty clips violent.

## Recommendation

**Switch triggered mode to RF-DETR-Small, with Medium as a config option.**

| | yolo11n (today) | **RF-DETR-S** | RF-DETR-M |
|---|---|---|---|
| CAVIAR AP50 | 0.102 | **0.425 (4.2×)** | 0.475 (4.7×) |
| Small-person recall | 0.030 | **0.484 (16×)** | 0.514 |
| Real violence CCTV, recall / precision @0.4 | 0.47 / 0.86 | **0.70 / 0.87** | 0.72 / 0.93 |
| ms/frame (eager) | 11.8 | 26–28 | ~32 |
| Licence | AGPL | **Apache-2.0** | Apache-2.0 |

- At 4 fps a camera gives the detector 250 ms per frame. RF-DETR-S uses about 11 % of that, and the extra
  ~15 ms per frame is small next to one VLM window (1.2 s).
- **Why RF-DETR over D-FINE:**
  - D-FINE-M/X have slightly better AP and small-object recall.
  - RF-DETR is faster in this setup, ships its own lean `predict` and ONNX/TensorRT export, and returns
    `sv.Detections`, which our tracker already takes.
  - D-FINE is the alternative to measure once it runs through an exported runtime.
- **Why not YOLO26:** it's the right choice only if latency per frame has to stay under ~15 ms, and then
  yolo26s at 640 (AP50 0.200) still doubles today's accuracy for +2 ms.

**Switching between detectors** (what the research and these numbers support):
- **Per camera, by the size of the people in it.** This is the Chameleon idea (Jiang et al., SIGCOMM 2018):
  adapt each camera's configuration, because the best one depends on object size and is stable over time.
  - **Far cameras** (people < ~60 px, CAVIAR-like) need a DETR.
  - **Close cameras** (the webcam, bodycam, close indoor CCTV): on CAVIAR every model finds people of 30 px
    and taller at 0.83–0.99.
    - But the real-violence test shows YOLOs still miss about half of large people once they fight, fall or
      overlap. RF-DETR-S finds 0.84 of them on the faculty clips, against 0.49.
    - So a light YOLO is enough only for calm, close scenes. For incident footage, keep the DETR.
  - This is a per-camera `detector:` override plus a one-off calibration that measures median person height.
    It's cheap to build.
- **Per frame, cheap first and escalate to the strong model** (cascade): **not recommended now.**
  - The gap is only ~15 ms per frame.
  - A cascade would keep two models in memory and add an escalation rule we'd have to tune.
  - Its misses (people the cheap model never saw) are exactly the case we're trying to fix.

## Integrated: `detector.family: rfdetr` (2026-10-16)

- **How to turn it on:** `pip install -e .[rfdetr]`, then set `detector: {family: rfdetr, rfdetr_size: small}`.
  - `vision/detector.Detector` returns the same `sv.Detections` with COCO-80 ids for either family.
  - RF-DETR's COCO category ids are mapped: person 1 → 0, backpack 27 → 24, handbag 31 → 26, suitcase 33 → 28.
- **In use:** `config/settings.yaml` sets `family: rfdetr` since 2026-10-16. The code default stays
  `yolo`, and without the `rfdetr` package the detector falls back to YOLO with a warning.
- **What it affects:** the version tag stored with events becomes `rfdetr-small`. Only triggered mode uses a
  detector, so periodic mode is unaffected.
- **Tests:** `tests/test_detector.py`, with a fake model.

### CAVIAR rules, end to end (11 clips, `aequitas run` + `eval-report`, Gemma 4 E2B verifying, no violence model)

| Run | Config | Trigger: hits / false alarms / misses | Confirmed: hits / false alarms / misses |
|---|---|---|---|
| YOLO11n | `eval_caviar_v2.yaml` | 0 / 0 / 6 (no candidates at all) | 0 / 0 / 6 |
| **RF-DETR-S**, same thresholds | `eval_caviar_rfdetr.yaml` | **3 / 6 / 3** | **2 / 0 / 4** |
| RF-DETR-S, code-default confidences | `eval_caviar_rfdetr_default.yaml` | 2 / 3 / 4 | 0 / 0 / 6 |

- **Same thresholds:** the rules fire for the first time on CAVIAR.
  - **Confirmed:** person down on Fight_OneManDown and Rest_SlumpOnFloor, with no confirmed false alarm.
  - **Fight:** the fight rule fired on Fight_OneManDown (rule score 1.0), but Gemma rejected it ("a person is
    visible…").
  - **False alarms:** the six trigger-level false alarms are all the person-down rule on normal clips. Gemma
    rejected every one, so the trigger + VLM agreement design is doing its job.
- **Caveat on the thresholds:** the v2 thresholds (detector 0.10, person 0.15, track start 0.15) were tuned on
  these clips for YOLO11n. This compares detectors on equal settings, not an unbiased accuracy.
- **Default confidences** (0.25 / 0.4 / 0.25, never tuned on CAVIAR): triggers on 2 clips, and nothing is confirmed.
- Still missed: both fights confirmed, Rest_FallOnFloor, and the left bag.

### Bags (`scripts/bag_check.py`, CAVIAR LeftBag: the bag is 16×22 px, labelled for frames 945–1354)

| Detector | Frames with the bag found at ≥ 0.10 | at ≥ 0.25 | Best confidence |
|---|---|---|---|
| yolo11n 640 (in use) | 0 % | 0 % | – |
| yolo11m 960 | 0 % | 0 % | – |
| yolo26m 960 | 1 % | 0 % | 0.11 |
| **rfdetr small** | **65 %** | 4 % | 0.28 |
| rfdetr medium | 60 % | 6 % | 0.38 |

The rule still can't fire on this clip, for two reasons:
1. **The tracker drops the bag.** Its confidence is mostly 0.10–0.25. With `tracker.fuse_score` on, anything at
   ≤ 0.2 can't keep a track, so the bag is tracked at its spot in 0 frames (17 of 68 with `fuse_score` off,
   diagnostic only).
2. **The rule needs more time than the clip has.** `abandoned_object.unattended_s` is 30 s, but the bag is in
   the clip for only ~16 s.

Neither setting was changed on these clips. Both need data we don't evaluate on. LeftBag_PickedUp has no ground
truth locally.

## Next steps (in order)

1. ~~Decide the default for triggered mode~~: done, `config/settings.yaml` uses RF-DETR-Small.
2. **Retune on data we don't report:**
   - `tracker.activation_threshold` and `fuse_score`, and the detector / person confidences, for RF-DETR's
     calibration.
   - The person-down rule's false triggers.
   - `abandoned_object.unattended_s` for short clips.
3. **Latency:** export RF-DETR-S to ONNX and run it with ONNX Runtime GPU or TensorRT. The aim is to measure the
   2–4 ms-class numbers on the 4050.
   - TensorRT for Windows is a ~2 GB download.
   - ONNX Runtime GPU is smaller, but replaces the CPU `onnxruntime` the violence model uses.
4. **When the faculty dataset arrives:** label ~200 person boxes on it (or use its own labels) and re-run
   `detector_compare.py`. The pseudo-label column then goes away.
5. **Later:** fine-tune RF-DETR on CCTV person and bag data. It is built for fine-tuning; Ultralytics models are
   under AGPL either way.

## Caveats

- One laptop, one run per model (RF-DETR rerun for timing). CAVIAR is 384×288 from 2004, and its people are
  smaller than on most modern CCTV.
- RF-DETR ran in FP32 (its `predict` default). The others ran in FP16.
- No model was fine-tuned or threshold-tuned on these clips. AP is threshold-free, while recall and IDF1 use the
  existing CAVIAR eval config.

**Sources:** RF-DETR ([paper](https://arxiv.org/abs/2511.09554), [docs](https://rfdetr.roboflow.com/latest/));
YOLO26 ([docs](https://docs.ultralytics.com/models/yolo26/), [paper](https://arxiv.org/abs/2601.12882));
DEIMv2 / D-FINE comparison ([paper, Table 3](https://arxiv.org/abs/2509.20787)); SAHI tiling
([paper](https://arxiv.org/abs/2202.06934)); Chameleon
([Microsoft Research](https://www.microsoft.com/en-us/research/publication/chameleon-video-analytics-scale-via-adaptive-configurations-cross-camera-correlations/)).

Tiling (SAHI) was not tested. On CAVIAR's 384×288 frames, upscaling (imgsz 960) already does what tiling would. On
1080p+ faculty footage with distant people, it's the next thing to try if the DETRs still miss them, at
~4–12× the detector cost.
