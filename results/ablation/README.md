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

## Run 3 (2026-10-15, Kaggle T4 x2, vLLM 0.31): `run3/`

This run used the **older imported notebook** (its quantizer environment had no `torchvision`), so every
4-/8-bit config failed again with "VideoProcessor requires the Torchvision library" (Gemma's shows up as
"Could not import module 'Gemma4Processor'", the same missing package). The repo notebook already installs
it: re-import `kaggle/ablation.ipynb` from the repo before the next run.

**Gemma 4 on vLLM, root cause found** (`run3/logs/vllm-gemma-*.log`): while vLLM profiles CUDA-graph memory,
a Triton kernel asks for 96 KB of shared memory per block; the T4 has 64 KB
(`OutOfResources: out of resource: shared memory, Required: 98304, Hardware limit: 65536`). It fails the
same way in FP16 and FP32, so it is the GPU generation, not the precision. Fix in the notebook: Gemma starts
with `--enforce-eager` (no CUDA graphs), then eager + FlexAttention, then FP32 over both T4s; a start only
counts if the warm-up requests succeed. The laptop's RTX 4050 (Ada, 100 KB shared memory) is not affected.

| Config | Window alone / 8 at once | Summary full | Label | Chat first word | Dev F1 | Real clips (n=10) |
|---|---|---|---|---|---|---|
| Qwen3-VL-2B FP16 | 1.94 s / 7.29 s | 1.77 s | 0.18 s | 0.05 s | 0.93 | acc 0.7, precision 1.0, recall 0.4 |
| SmolVLM2-2.2B FP16 | 4.19 s / 18.38 s | 4.04 s | 0.15 s | 0.04 s | 0.09 | acc 0.6, precision 0.67, recall 0.4 |

SmolVLM is 2× slower and its labels are near random on the dev set (macro-F1 0.09): ruled out.

Time-saved ladder (Qwen FP16, 20 windows of real video, warm server):

| Step | Total | Saved vs baseline |
|---|---|---|
| 0 baseline: 1 at a time, long summary | 54.1 s | |
| 1 same config again (quantization not applied: no quantized model ran) | 33.1 s | 21.0 s * |
| 2 + 8 windows in flight (vLLM batching) | 20.1 s | 34.0 s |
| 3 + short capped summary | 15.0 s | 39.1 s |
| 4 + skip static windows | 5.4 s | 48.7 s (10× faster) |

\* Step 1 changed nothing, yet ran 21 s faster: run-to-run variance on a shared Kaggle GPU is that large on
the first step. Honest figure for the levers we actually changed: **33.1 s → 5.4 s (6.1×)**.
