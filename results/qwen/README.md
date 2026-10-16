# Switching the demo model to Qwen3-VL 2B: every presentation number re-measured (2026-10-17)

The owner chose `qwen3-vl:2b-instruct` over `gemma4:e2b` after the same-size comparison
(results/model_compare_small/README.md) and asked for every number in the presentation to be re-run with it.

**Setup:** RTX 4050 laptop (6 GB), Ollama, temperature 0. The Ollama web server (port 8000) was running but idle.
Runs marked "busy" overlapped a model download and are not used for timing.

## Decisions made during the re-run (owner away, "make the best decision")

| Decision | Why (measured) |
|---|---|
| **Ollama, no speculative decoding for Qwen** (owner's choice) | Ollama only speculates with an MTP drafter shipped inside the model, and Qwen3-VL has none. llama.cpp refuses a draft model when an image projector is loaded. The Gemma MTP measurement (docs/SPECULATIVE.md) stays as the requirement-3 evidence. |
| **Previous-window context off** (`analysis.label_context: false`) | On the 72 saved windows, Qwen with context: fire 6/10 → 1/10, flagged 33 → 29 of 39, false alarms 2 → 4 of 33. Fight windows unchanged. With Gemma, context had helped. |
| **4-bit (Q4_K_M), not 8-bit or 16-bit** | See *Quantization*. 8-bit was better on the 72 windows. End to end, 8-bit scored 8/5/2 three times against 4-bit's 9/2/1 and 10/2/0, called fire-rescue "vandalism" in every run, and was 6–10 % slower. 16-bit doesn't fit the GPU. |
| **`repeat_penalty: 1.1`** (new setting `vlm.repeat_penalty`) | At temperature 0, Qwen repeated a sentence until the token cap in 3 of 6 chat answers to the demo questions, including "Who is the attacker?". With 1.1: 0 of 6, shorter answers, chat 0.62 → 0.43 s. On the 72 windows: 4 → 0 broken outputs, 127 → 108 s total, p95 2.86 → 1.62 s, about the same accuracy. 1.2 still looped once, and presence penalty 1.5 was slower. |
| **Live CCTV: Electronic City's Neeladri Road camera now shows a car break-in (theft)** | Qwen confirms theft on it. The camera showed fire-rescue, a second fire next to Whitefield's. |
| **Fusion rule left alone** | Qwen often labels a faculty fight "person_down" ("a man is pushed to the ground"), and the fusion keeps the VLM's specific label over the violence model's "fighting". Changing the rule to fix that would be tuning on the held-out faculty clips. |

## Quantization (faculty requirement 4): 4-bit vs 8-bit vs 16-bit on the laptop

72 saved windows (`scripts/bench_variants.py --variants base`, `data/bench_windows`), the same frames for each
precision, run back to back.

| Precision (Ollama tag) | GPU memory | Fits 6 GB? | Fight / arrest / fire right | Incident windows flagged | False alarms (normal) | Broken outputs | Time / window p50 · p95 | Total, 72 windows |
|---|---|---|---|---|---|---|---|---|
| **4-bit Q4_K_M** (`2b-instruct`) | 1.85 GB | yes | 7/17 · 11/12 · 6/10 | 33/39 | 2/33 | 4 | **1.60 · 2.86 s** | **127 s** |
| 8-bit Q8_0 (`2b-instruct-q8_0`) | 2.53 GB | yes | 9/17 · 12/12 · 4/10 | 34/39 | 0/33 | 0 | 1.74 · 1.95 s | 135 s |
| 16-bit BF16 (`2b-instruct-bf16`) | 4.71 GB needed, 3.41 GB on GPU | **no**: 23 of 29 layers on GPU, the rest on CPU | 6/17 · 12/12 · 4/10 | 33/39 | 0/33 | 1 | 9.64 · 10.3 s | 707 s |

End to end on the 21 demo clips (final setup, confirmed hits / false alarms / misses): 4-bit 9/2/1 and 10/2/0;
8-bit 8/5/2 in all three runs (`data/eval/demo_q8_*`). Most of 8-bit's extra false alarms are wrong labels on
faculty fights (harassment, person_down). Those clips are not used for the choice. On the other clips, 8-bit adds
a "vandalism" label on fire-rescue in every run.

## The numbers in the deck: Gemma → Qwen, shipped setup

The shipped setup is 4-bit, context off, `repeat_penalty: 1.1`. The deck uses its numbers. Runs:
`data/eval/*_rp`, `data/reports/*q4rp*` and `bench_text_qwen_rp.txt`. The middle column is the same setup
before the repeat penalty was added.

| Deck item | Gemma 4 E2B (4-bit, MTP 3, context on) | Qwen, no repeat penalty | **Qwen, shipped** |
|---|---|---|---|
| Faculty clips, wall time, warm / cold | 25 s / 40 s | 29 s / 47 s | **28 s / 42 s** |
| Faculty clips flagged as incidents | 6/6 | 6/6 | 6/6 |
| Faculty: right label confirmed | 4/6 | 3/6, 2/6 | 2/6, 3/6 |
| Faculty: the VLM's own label is violent | 0/6 | 2/6 | 4/6 |
| Window: summarise / classify / both (p50, faculty warm) | 0.58 / 0.32 / 0.91 s | 0.77 / 0.32 / 1.06 s | 0.69 / 0.34 / 1.01 s |
| 21 demo clips, confirmed hits / false alarms / misses | 7/1/3 (later runs 6/2/4 to 8/1/2) | 9/2/1, 10/2/0 | 9/2/1, 8/3/2 |
| 21 demo clips, wall time | 1 min 48 s | 2 min 33 s, 2 min 40 s | 2 min 24 s (the repeat run, 3 min 48 s, had its VLM time jump 120 → 194 s with no setting change, so other GPU load; not used for timing) |
| Clip votes (replay of the saved windows) | 5/10 → 7/10 | 8/10, 7/10 → 10/10 | 7/10, 7/10 → 10/10, no new false alarms |
| Scene gate, VLM time on 21 clips | 113.9 → 81.7 s (−28 %) | 136.5 → 122.8 s (−10 %) | 136.8 → 120.0 s (−12 %), 13 normal windows skipped, no incident window |
| 72 saved windows: flagged / false alarms / broken outputs / total time | 25/39 (context) / 0/33 / – / – | 33/39 / 2/33 / 4 / 127 s | 34/39 / 3/33 / 0 / 108 s |
| Stage 2, 57 labelled summaries | 0.98 | 0.91 | 0.91 (incident P/R 0.94 / 1.0) |
| Chat: question → search filter, dev 28 + held-out 20 | 48/48 | 48/48 | 47/48 (one held-out camera wrong) |
| Chat answer, p50 of 12 | 0.45 s (MTP) / 0.58 s | 0.62 s | **0.43 s** |
| Chat answers that loop to the token cap (6 demo questions) | – | 3/6 | 0/6 |
| Bodycam recording summary, p50 of 6 | 1.6 s | 2.3 s | 2.8 s |
| Translation Hindi / Kannada → English | 0.31 / 0.34 s, 6/6 kept every fact | 0.33 / 0.33 s, 1/6 | 0.34 / 0.30 s, **2/6 kept every fact** |
| Incident clips for the Live CCTV cameras | 4 of 10 candidates | 9 of 10 | 8 of 10. All four used on cameras pass (fire, crashed car, vandalism, car break-in) |

**Translations, checked by eye** (no repeat penalty; the shipped run kept Hindi 2 and 3 right, and still lost the
phone snatch, the knife, MG Road and the pedestrian fleeing):
- Hindi 1: "surrounded him, snatched his phone and pushed him" became "seized him and hit him with a punch".
- Hindi 2: "this evening" became "this morning at 5:00 PM".
- Kannada 1: "MG Road" became "M.J. road".
- Kannada 2: "knife" (chaku) became "chakka".
- Kannada 3: "fled" was dropped and an injury was invented.

The 8-bit model did a little better (`data/reports/bench_text_q8.txt`) but still dropped the knife. **Recommendation for the owner:**
use Gemma for translation (`chat.model`), or show the original text next to the translation. Before deciding, measure
whether running both models on the 6 GB GPU makes Ollama swap them.

**Speculative decoding** is not available for Qwen in Ollama (no drafter), so the deck keeps the Gemma MTP
measurement, labelled as Gemma's.

## Not re-run (model-independent)
- Violence model: 90 ms on the CPU; 77 % of fights at 14 % false alarms on 300 unseen clips.
- SigLIP gate: 44 ms.
- RF-DETR vs YOLO.
- The Kaggle speed ladder: it already used Qwen3-VL 2B, on vLLM FP16.
- The model comparisons (results/model_compare*, as measured).
