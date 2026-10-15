# Ablation results

Files here come from `kaggle/ablation.ipynb` (unzipped `ablation.zip`). See docs/ABLATION_PLAN.md.

## Run 1 (2026-10-15, Kaggle T4 x2, vLLM 0.31.0): what happened

Only **Qwen3-VL-2B FP16** ran end to end. The other 8 configs failed for setup reasons, not model reasons:

| Failure | Cause (from the log) | Fixed in the notebook for run 2 |
|---|---|---|
| All quantizations | llm-compressor's environment had a `transformers` too old for Qwen3-VL / Gemma 4 | `transformers` upgraded after llm-compressor |
| SmolVLM FP16 did not start | its image processor failed while vLLM profiled it (likely the missing `num2words` package) | `num2words` installed |
| Gemma FP16 did not start | the cause was cut off: the notebook kept the start of errors, not the end | full logs in `logs/`, errors keep their end; automatic FP32 retry over both T4s |

### Qwen3-VL-2B FP16 (the only full row)

| Measure | Value |
|---|---|
| One window (summary + label), alone / 8 at once | 2.27 s / 7.93 s |
| Summary first word / full | 0.57 s / 2.10 s |
| Label | 0.16 s |
| Chat first word / full answer | 0.06 s / 0.94 s |
| Dev-set label macro-F1 (48 hand-written summaries) | 0.95 |
| **Real clips, fighting vs none (n = 10)** | **0.5 accuracy, 0 of 5 fights found** |

Why the real clips failed: the summaries mostly described the attacks ("being attacked by another person",
"one man is being held by the other... falls"), but **Qwen as the labeller** chose person_down / chase /
suspicious. The **same summaries labelled by Gemma 4 E2B** (local check, Ollama): **9/10** correct, and
Gemma scores 98 % on the dev set. The labeller choice matters more than the prompt; the label prompt now also
has priority rules (attack beats fall; sports are none), checked to cause no regression.

### Latency levers (Qwen FP16)

- Smaller tiles: 224 px vs 448 px = 1.62 s vs 1.97 s alone, **3.05 s vs 6.99 s with 8 at once**, accuracy not worse.
- Concurrency 1 -> 8: throughput 0.47 -> 1.05 windows/s, but each window waits longer (2.2 -> 7.4 s). The T4 is
  compute-bound.
- A 64-token summary cap truncates the JSON (invalid output, retries): 128 stays.

### Time saved (80 s test video)

The table in ABLATION.md says 59.6 s -> 5.3 s (11x), but steps 0 and 1 were the **same setup** (59.6 s vs
31.6 s): the first run after the server started included warm-up. **The honest figure is 31.6 s -> 5.3 s (~6x)**,
and most of the last step comes from skipping static windows on a test video that is mostly quiet stretches.
Run 2 warms the server up after every start. One window was dropped at the end of the video (fixed: the last
window now waits for room too).
