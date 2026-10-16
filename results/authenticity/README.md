# AI-generated check: frame detectors measured on real and AI video

2026-10-16. Script: `scripts/eval_authenticity.py`. Per-clip scores: `scores/*.json`; summaries: `summary.json`
and `summary_crop15.json`.

## Why

The citizen portal's check ran `Organika/sdxl-detector` (an image detector trained on SDXL stills) on 8 sampled
frames and showed the median as "AI-generated %". It gave real stock video 100 %, including the live demo clip,
and the officer screens showed "100 % ?".

## Data

| Set | Clips | Source |
|---|---|---|
| real | 36 | everything under `data/footage` except `faculty/` (Pexels stock in the locality folders, `incidents/`, `staged_candidates/`, `demo/`) |
| real | 44 | `data/videos/survfight`, every 7th clip of `fight/` and `noFight/` (real YouTube CCTV, MIT licence) |
| AI | 40 | `bitmind/klingai-videos` (Apache-2.0): the first 20 files under 5 MB from `batch_000` and from `batch_005` |
| AI | 10 | `bitmind/bm-video-benchmarks`, split `20250821_132430` (81 MB tar; no licence stated, used only for local evaluation): CogVideoX-5b ×5, HunyuanVideo ×2, AnimateDiff-Lightning ×2, Mochi-1 ×1 |
| faculty | 6 | `data/footage/faculty`: **reported only**. These are the held-out test, never used to choose a model or a threshold |

The clips live in `data/videos/ai_kling/` and `data/videos/ai_bmbench/` (gitignored). No Sora or Veo clips: the
sources I found came only as 0.9–2 GB zips, and `bitmind/CogVideoX1.5-5B-I2V_0to782` is empty.

Score = median over 8 frames of P(AI), using the same frame sampling as the app.

## Results (real n=80, AI n=50)

| Detector (licence, size) | AUC | ≥ 0.5: FPR / TPR | ≥ 0.85: FPR / TPR | Real clips at ≥ 99.5 % | Kling ≥ 0.85 | Open models ≥ 0.85 |
|---|---|---|---|---|---|---|
| Organika/sdxl-detector (CC-BY-NC-3.0, 347 MB), **old default** | 0.57 | 48 % / 58 % | 38 % / 50 % | 22 / 80 | 24 / 40 | 1 / 10 |
| umm-maybe/AI-image-detector (CC-BY-4.0, 348 MB) | 0.63 | 30 % / 42 % | 20 % / 20 % | 4 / 80 | 9 / 40 | 1 / 10 |
| Ateeqq/ai-vs-human-image-detector (Apache-2.0, 372 MB) | 0.86 | 42 % / 88 % | 39 % / 86 % | 28 / 80 | 40 / 40 | 3 / 10 |
| **haywoodsloan/ai-image-detector-deploy** (Apache-2.0, 781 MB), **new default** | 0.88 | 2 % / 70 % | **1 % / 68 %** | **0 / 80** | 34 / 40 | 0 / 10 |

FPR = share of real clips at or above the threshold; TPR = share of AI clips at or above it.

At a target false-positive rate of at most 5 %, a threshold picked on alternate halves of the data and tested on
the other half (2-fold) gives: haywoodsloan 5 % FPR / 72 % TPR, Ateeqq 8 % / 64 %, Organika 8 % / 12 %,
umm-maybe 6 % / 0 %.

Faculty clips (all real; not used for any choice):

| | v001 | v003 | v006 | v011 | v026 | w001 |
|---|---|---|---|---|---|---|
| Organika | 0.06 | 0.02 | 0.55 | 0.00 | 0.00 | 0.44 |
| umm-maybe | 0.49 | 0.04 | 0.08 | 0.66 | 0.05 | 0.98 |
| Ateeqq | 0.99 | 0.00 | 0.43 | 0.00 | 1.00 | 1.00 |
| haywoodsloan | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 |

haywoodsloan put all six below 0.85, so all six read "inconclusive". The other three detectors would have given
some faculty clips 98–100 %.

**Watermark check.** Kling clips carry a "KLING AI" logo in the bottom-right corner. Re-scoring with the bottom
15 % of every frame cropped (`--crop-bottom 0.15`) left haywoodsloan's Kling scores unchanged (median 1.00, AUC
0.87), so it isn't just reading the logo. Cropping raised its real FPR at 0.85 to 4 % (3/80).

**Speed** (RTX 4050, with Ollama holding about 4.7 GB): 8 frames take 0.33 s on the GPU (peak 852 MB VRAM) and
3.2 s on the CPU. Loading the model takes about 10 s on first use.

## What changed in the app

- Default detector: `haywoodsloan/ai-image-detector-deploy`. It runs on the GPU only when ≥ 1.5 GB is free, so it
  never squeezes the VLM; otherwise it runs on the CPU.
- When only the frame detector speaks (no C2PA, no AI-tool metadata, no camera metadata), the result has **no
  percentage** (`ai_likelihood_pct: null`, `basis: "frames"`):
  - median ≥ 0.85 (`frame_flag_at`, the old `likely_ai_at`, not tuned): verdict `uncertain`, "Frame check:
    AI-like frames, needs a human look (in our test this flagged 1 of 80 real clips and 34 of 50 AI clips)";
  - otherwise verdict `inconclusive`, "Frame check: inconclusive". **Changed the same day at the owner's request:**
    this case now reads verdict `no_ai_signs`, "Frame check: no AI signs found, treated as real", shown as
    **Real**. The owner prefers missed fakes to real videos flagged. Given the limits below (0/10 open-model clips
    caught), "Real" here means "no AI signs found", not "verified real".
  The raw score is kept in `frame_score` and `frames.per_frame` for audit, not shown as a likelihood.
- Percentages remain only for real evidence: C2PA declares AI (99 %), AI-tool metadata (90 %), phone camera
  metadata (10 %).
- The officer queue and report screens read the new `basis` field. Results stored before this change (no `basis`,
  first reason "Frame detector: …") also show "inconclusive", so old "100 %" rows disappear without a re-run.
- Demo clip `data/footage/demo/flyover-attack.mp4`: 0.03, shown as "inconclusive" (was "100 % ?").

## Limits

- **It catches Kling, not open models.** 34/40 Kling clips, but 0/10 from CogVideoX, HunyuanVideo, Mochi and
  AnimateDiff. Sora, Veo and Runway were not tested. "Inconclusive" never means "real".
- The sets are small (80 real, 50 AI), so 1/80 FPR has a wide confidence interval (roughly 0–7 %). The
  "1 of 80 / 34 of 50" note shown to officers is an in-sample figure from this set.
- The one flagged real clip is `incidents/arrest-4.mp4` (0.96, 6 of 8 frames ≥ 0.95). Real footage can still be
  flagged, so the frame check alone can never say "likely AI" and a human decides.
- The real set is mostly stock and YouTube CCTV. Phone video forwarded through WhatsApp (heavy re-encoding) is
  not in it.
- The AI clips were picked by file order and size (< 5 MB), not at random.
