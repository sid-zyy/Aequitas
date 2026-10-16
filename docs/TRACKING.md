# Tracking: ByteTrack vs DeepSORT (and BoT-SORT, OC-SORT)

Measured 2026-10-16 on the RTX 4050 laptop with `scripts/tracker_compare.py`. Every tracker gets the **same YOLO
detections**, so the differences come from association alone.

## Where Aequitas uses a tracker

Only **triggered mode** (`analysis.mode: triggered`) uses one: YOLO + ByteTrack (supervision) → rule triggers.
- **Need a stable ID over seconds to minutes:** loitering, abandoned object (owner left), tailgating (two IDs
  passing one after the other), fighting (a pair of tracks close together).
- **Need a per-frame box only:** person down, crowd count.
- **No tracker at all:** periodic mode (the live CCTV demo, the webcam, bodycam review). The skeleton prototype
  (`vision/skeleton.py`) keeps ≤ 2 people in slots with a nearest-centre match, which is enough at close range.

## What was measured

CAVIAR, 8 clips with hand-labelled tracks: Browse_WhileWaiting1, Fight_Chase, Fight_OneManDown, Fight_RunAway1,
LeftBag, Meet_Crowd, Rest_InChair, Walk1 (41 labelled objects, 384×288, high angle). Detector conf 0.10 and
track start threshold 0.15, as in `config/eval_caviar_v2.yaml`. Lost-track buffer 15 s for every tracker. A
match needs IoU ≥ 0.5. The time column covers the tracker only, on frames with detections: DeepSORT's appearance
model runs once per detection.

**IDF1** (how long the right ID stays on the right person) is the metric that matters for the rules above.

| Setup | Tracker | IDF1 | MOTA | ID switches | Recall | Precision | ms/frame |
|---|---|---|---|---|---|---|---|
| YOLO11n 640, 4 fps (today) | ByteTrack (supervision, Aequitas) | 0.052 | −0.071 | 9 | 0.041 | 0.276 | 0.24 |
| | ByteTrack (Ultralytics) | 0.050 | −0.111 | 11 | 0.043 | 0.225 | 0.23 |
| | BoT-SORT (camera-motion comp., no ReID) | 0.050 | −0.115 | 11 | 0.044 | 0.223 | 2.75 |
| | OC-SORT | 0.056 | −0.093 | 11 | 0.044 | 0.250 | 0.35 |
| | DeepSORT (MobileNetV2 ReID) | 0.063 | −0.450 | 53 | 0.099 | 0.159 | 32.2 |
| YOLO11m 960, 4 fps | ByteTrack (supervision, Aequitas) | 0.141 | 0.049 | 13 | 0.116 | 0.655 | 0.33 |
| | ByteTrack (Ultralytics) | 0.143 | 0.037 | 13 | 0.118 | 0.614 | 0.31 |
| | BoT-SORT (no ReID) | 0.160 | 0.044 | 12 | 0.121 | 0.628 | 2.89 |
| | OC-SORT | 0.164 | 0.044 | 11 | 0.120 | 0.630 | 0.44 |
| | DeepSORT | 0.154 | −0.035 | 58 | 0.173 | 0.490 | 46.0 |
| YOLO11m 960, 12 fps | ByteTrack (supervision, Aequitas) | 0.192 | 0.077 | 49 | 0.163 | 0.677 | 0.39 |
| | ByteTrack (Ultralytics) | 0.201 | 0.073 | 48 | 0.167 | 0.661 | 0.31 |
| | BoT-SORT (no ReID) | 0.205 | 0.080 | 41 | 0.173 | 0.668 | 3.03 |
| | OC-SORT | 0.198 | 0.075 | 37 | 0.163 | 0.664 | 0.47 |
| | DeepSORT | 0.192 | −0.006 | 167 | 0.236 | 0.523 | 30.1 |

## What the numbers say

1. **The detector is the bottleneck, not the tracker.**
   - YOLO11n at 640 px finds very few CAVIAR people. On two clips checked by hand, only 2 % (Walk1) and 33 %
     (Fight_OneManDown) of labelled boxes had a detection at IoU ≥ 0.5.
   - It also keeps a fixed false "person" in the left of the frame in every clip.
   - Switching to YOLO11m at 960 px raised every tracker's IDF1 2.7–3.3×, and precision from ~0.25 to ~0.65.
   - The choice of tracker moves IDF1 by at most 0.02 within any setup.
   - The detector's own extra latency was not timed here. Measure it before switching.
2. **The frame rate matters next.** From 4 to 12 fps, IDF1 rose about 35 % (0.14 → 0.19–0.20). Motion models
   predict better when frames are closer together. The cost is 3× the detector calls.
3. **Among the IoU/motion trackers there's no clear winner.**
   - OC-SORT and BoT-SORT have the fewest ID switches (11–12 at 4 fps, 37–41 at 12 fps, against ByteTrack's
     13 / 48–49).
   - The IDF1 differences (≤ 0.024) are small for 41 objects in one run. Treat them as a tie.
4. **DeepSORT is a poor fit.**
   - **ID switches:** 4–5× more than every other tracker (53–167). Its appearance embeddings of small,
     high-angle 384×288 people are unreliable, so it re-links the wrong people.
   - **Recall:** it does find more boxes (it keeps tracks alive on appearance), but with lower precision and
     the worst MOTA.
   - **Latency:** 30–46 ms per frame, about 100× ByteTrack, because each detection needs a CNN embedding.
     With latency weighted most heavily, that's per camera, per frame.
   - **Guardrails:** it stores an appearance vector per person. That's re-identification by appearance,
     close to what the guardrails rule out (no identification of people) if those vectors are ever compared
     across cameras or kept.

## Fit and tradeoffs

| | ByteTrack | OC-SORT | BoT-SORT | DeepSORT |
|---|---|---|---|---|
| Association | IoU + Kalman, second pass with low-score detections | ByteTrack + observation-centric motion recovery | ByteTrack + camera-motion compensation (+ optional ReID) | Kalman + appearance (CNN embedding) |
| Cost here | 0.2–0.4 ms | 0.4 ms | 2.8–3 ms (optical flow for camera motion) | 30–46 ms + embedder weights |
| Strong when | weak detections (low-res CCTV): the second pass keeps low-confidence people | non-linear motion, short occlusions | **moving camera** (bodycam, PTZ) | long occlusions with distinct-looking people at good resolution |
| Weak when | long occlusion (only the lost buffer) | same | the camera is static: the motion compensation costs time for no gain (here: no gain) | small/low-res people, uniforms (police, school), crowds: lookalikes swap IDs |
| Aequitas fit | **fixed CCTV, today** | fixed CCTV, a drop-in alternative | **bodycam** (Part 2) if tracking is ever needed there | not recommended |

## Recommendation

- **Keep a ByteTrack-family tracker.** Don't adopt DeepSORT: it's slower, has more ID switches, and raises a
  privacy issue.
- **Migrate off `supervision.ByteTrack`.** It is deprecated since supervision 0.28 and **removed in 0.31**
  (warning seen in this run; we're on 0.30.9).
  - Ultralytics' `BYTETracker` or `OCSORT` (already installed, same numbers) can replace it.
  - Alternatively, pin `supervision<0.31` until then.
  - The `activation_threshold − 0.1` workaround in `vision/detector.py` is specific to supervision. Drop it
    when switching.
- **Spend the latency budget on detection, not association.**
  - YOLO11m at 960 px tripled IDF1 on CAVIAR. Time it on the 4050 first.
  - Run triggered-mode detection above 4 fps only where the rules need IDs.
- **For bodycam tracking later,** use BoT-SORT for its camera-motion compensation, without ReID.

**Caveats:**
- **Scope:** one dataset, 8 clips, one run, default DeepSORT settings (n_init 2), thresholds from our CAVIAR
  config.
- **Absolute scores are low** because CAVIAR's people are tiny for a COCO detector. The comparison between
  trackers is the useful part, not the absolute IDF1.
