# First real-footage test: CAVIAR (2026-10-15)

**Footage:** 11 clips from the CAVIAR INRIA lobby set (CC BY-SA; EC Funded CAVIAR project/IST 2001 37540).
384×288, 25 fps, 17–58 s, one high wide-angle fixed camera.
Download with `scripts/get_caviar.sh`; expected labels are in `config/caviar_expected.yaml`.
Scoring is clip-level, with `aequitas eval-report`.

**Machine:** CPU only (Intel i9-13900H, Iris Xe). VLM: gemma4:e2b.

## Results

| Run | Config | Hits | False alarms | Misses | Precision | Recall |
|---|---|---|---|---|---|---|
| Baseline | defaults (`eval_caviar.yaml`) | 0 | 0 | 6 | – | 0.00 |
| Loosened thresholds | `eval_caviar_lowres.yaml` | 0 | 0 | 6 | – | 0.00 |
| A: tracker + person-down fixes | `eval_caviar_v2.yaml` | 0 | 0 | 6 | – | 0.00 |
| B: VLM review only, every 4 s window, all events | `eval_caviar_periodic.yaml` | 3 | 8 | 3 | 0.27 | 0.50 |
| B: VLM review, `confirmed` only | | 0 | 2 | 6 | 0.00 | 0.00 |
| C (2026-10-16, GPU): A with **RF-DETR-S** instead of YOLO11n, triggers | `eval_caviar_rfdetr.yaml` | 3 | 6 | 3 | 0.33 | 0.50 |
| C: `confirmed` only | | 2 | 0 | 4 | 1.00 | 0.33 |
| A re-run 2026-10-16 (GPU, YOLO11n) | `eval_caviar_v2.yaml` | 0 | 0 | 6 | – | 0.00 |

Run C changes only the detector (docs/DETECTION.md). The A thresholds were tuned on these clips for YOLO11n, so C is a
detector comparison on equal settings, not an unbiased accuracy. With untuned default confidences RF-DETR-S gives 2
trigger hits and 0 confirmed.

The loosened thresholds were tuned on these same clips, so even a better number here would be optimistic.

## What went wrong, and why

**Rules (baseline and run A): the person detector is the bottleneck.**
- YOLO11n gives people in this footage median confidence 0.16–0.27. The default filter (≥ 0.4) dropped 79–100% of them.
- supervision's ByteTrack only starts tracks above its threshold **+ 0.1**, so low-confidence people were never
  tracked. *Fixed:* the configured value is now the real start threshold (since 2026-10-16 the tracker is
  Ultralytics' BYTETracker, which has no offset; see docs/TRACKING.md).
- In `Rest_FallOnFloor`, the person who falls is **not detected while walking**. They first appear already on
  the floor, so no "upright → down" transition exists to detect.
  - Person-down now also handles ID switches (upright at the same spot) and overhead perspective (height
    collapse). Unit tests cover both. But it cannot recover a person the detector never saw.
- A static object is detected as a person (~97 px, confidence 0.2–0.4) throughout several clips.
- Fight pairs are rarely tracked together, and motion energy stays at 0.1–0.8× baseline against a 3× threshold.
- **No bags were detected** in either LeftBag clip, so abandoned-object detection was impossible.
- Loitering needs 60 s of dwell and every clip is shorter, so it could not fire.

**VLM review alone (run B): biased towards "person down".**
- Gemma 4 E2B answered `person_down` for **every clip**: walking, sitting, meeting, browsing.
- Small people in a 2×2 grid from a high angle seem to look "on the ground" to a 2B model.
- It never recognised either fight.
- Repeating the same wrong answer in consecutive windows confirmed 2 false alarms. So window agreement does not
  protect against a **systematic** bias.
- About 20–29 s per window on this CPU.

## What this changes

> **Follow-up 2026-10-16** (docs/TRACKING.md, against CAVIAR's ground-truth tracks):
> - The weak detector is confirmed as the bottleneck. YOLO11n at 640 px matches 2–33 % of labelled people
>   and keeps a fixed false "person" on the left of the frame.
> - YOLO11m at 960 px tripled tracking quality (IDF1 0.05 → 0.14–0.16), whichever tracker is used.
> - Its latency on the 4050 is not timed yet.
> - Periodic mode (VLM review every window, with the violence clip model as the second signal) became the
>   main mode (ARCHITECTURE.md §4).

1. **The trigger + VLM agreement design holds up:** the VLM alone would have flooded operators with false falls.
   But the trigger side needs a much better person detector on low-res CCTV.
2. **Next experiments**
   - Larger off-the-shelf detectors (YOLO11s/m, larger `imgsz`) on CAVIAR before any training. This is cheap and
     may fix person recall.
   - The Kaggle violence clip classifier (`kaggle/violence_clip.ipynb`) for fights. It needs no tracking.
   - Hazard detector for fire and weapons (`kaggle/hazard_detector.ipynb`).
3. **Part 2 (bodycam) cannot rely on Gemma 4 E2B window review alone.** Before relying on it, re-test with:
   - `gemma4:e4b` on the GPU
   - a prompt that asks first whether anyone is lying down, with a "people standing or walking" counter-example
   - single frames at higher resolution instead of a 2×2 grid
4. Re-run this evaluation after each change. It takes about 5 minutes for rules and 10 for VLM review on this CPU.
