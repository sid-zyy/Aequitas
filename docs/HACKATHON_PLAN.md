# Hackathon plan: faculty requirements (2026-10-15, status updated 2026-10-16)

## What the faculty asked for

| Requirement | How Aequitas meets it | Status (2026-10-16) |
|---|---|---|
| Three tasks: **summarization → classification of the summary → chatbot** | Three explicit stages with their own prompts, model endpoints, timings and metrics (below; docs/ARCHITECTURE.md §2). | **Done** |
| **vLLM** server | Every LLM/VLM call can go through vLLM's OpenAI-compatible API (`vlm.backend: openai`). Ollama is the dev fallback and what the laptop runs today. vLLM runs in **WSL2 Ubuntu**. | Scripts done, run on Kaggle; **WSL not installed on the laptop yet** |
| **Speculative decoding** and **multi-token prediction (MTP)** | Gemma 4's MTP drafter (`-assistant` checkpoint) through vLLM's `--speculative-config`; **n-gram** speculation for chat answers, which copy retrieved summaries. | `serve.sh` profiles done; **speed-up not measured yet** |
| **4-bit and 8-bit quantization**, studied and implemented | BF16 baseline, **W8A8-INT8**, **FP8** and **W4A16** made with `llm-compressor` (`scripts/quantize/`). Each benchmarked for latency **and** accuracy. | Script done; **no successful run yet** (Kaggle environment issues, fixed) |
| **React** frontend, **FastAPI** backend, no Jupyter | React (Vite + TypeScript) served by FastAPI. Training and study code are `.py`; the `.ipynb` files are generated from them only for Kaggle. | **Done** |
| Judged on **approach**, **end-to-end demo**, **latency (total execution time)** | Every stage is timed and stored; camera tiles, bodycam and report pages show summary and label time and the total. `aequitas bench` produces the latency report across quantization × speculation. | Timing in the UI done; **latency report pending the vLLM runs** |
| The faculty dataset | `aequitas eval-dataset` (folder-per-label or a manifest; categories configurable). | Adapter done; **no dataset is coming** (2026-10-16): the 6 sample clips are the faculty test |

**Model:** Gemma 4 E2B (decided). The ablation is re-run at the end for the presentation's quantization and
speculation tables.

The dataset arrives soon. Everything below is built so the dataset only needs an adapter and labels.

## The GPU laptop (recorded 2026-10-15)

| Fact | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| VRAM | **6 GB** (6141 MiB) |
| Compute capability | **8.9** (Ada): FP8 is supported |
| Driver | 592.82 (CUDA 13.1) |

**What 6 GB means.** Gemma 4 E2B has 5.1 B parameters in total (2.3 B effective; the rest are
per-layer embeddings), so its **BF16 weights are ~10 GB and do not fit** in VRAM. vLLM's recipe lists
E2B/E4B for 24 GB+ GPUs. Consequences:
- The BF16 baseline can only run with `--cpu-offload-gb` (slow). Measure it anyway: it is the
  accuracy reference, and its latency shows what quantization buys.
- Quantization is what makes the model fit here at all: FP8 (~5 GB of weights) is borderline once the
  KV cache and the Windows desktop are counted, and W4A16 is the likely default. This is a real finding
  for the quantization report, not a footnote.
- Whether the per-layer embeddings are quantized decides most of the size. Check it per recipe.
- E4B will not fit unquantized; try it only at 4 bit.

**Verified model IDs (HF, Apache 2.0, not gated):** `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`.
MTP drafters: `google/gemma-4-E2B-it-assistant`, `google/gemma-4-E4B-it-assistant`.

**Correction to the earlier plan:** Gemma 4 MTP in vLLM *does* use a separate checkpoint. The
`-assistant` model is passed as the speculator:
`--speculative-config '{"method":"mtp","model":"google/gemma-4-E2B-it-assistant","num_speculative_tokens":1}'`.
Older vLLM releases treat it as a generic draft model and fail on multimodal targets
(vllm-project/vllm#42005). If the log shows `method='draft_model'`, upgrade vLLM.
`scripts/vllm/serve.sh` builds these flags.

## Pipeline v2: the three tasks

```
video ──► segments ──► [1] SUMMARIZE (VLM) ──► summary text ──► [2] CLASSIFY (LLM, constrained) ──► label + confidence
              ▲                                                                                      │
   YOLO triggers or fixed windows                                                              event store
                                                                                                     │
                                         [3] CHATBOT: question → filters + retrieval → grounded streamed answer
```

1. **Summarization.** The VLM gets a segment (a 2×2 key-frame grid or a short frame sequence) and returns
   a 1–2 sentence factual summary. Output is short, so latency is dominated by image prefill.
   Requests run **concurrently**, and vLLM's continuous batching processes them together.
2. **Classification of the summary.** A text-only call constrained by vLLM structured outputs
   (`{"structured_outputs": {"choice": [...categories, "none"]}}`). The output is one label, a few tokens,
   so it is very fast. Confidence comes from the label's token log-probabilities.
   - Option B, once the dataset arrives: a small fine-tuned text classifier (millisecond latency).
     Compare both on accuracy and time.
   - The rule trigger, where one exists, is still combined with the label (agreement → confirmed). On
     CAVIAR, the VLM alone was strongly biased, so this check stays.
3. **Chatbot.** Unchanged grounding (filters + `[#id]` references), now **streamed** token by token, with
   n-gram speculation, because answers quote the retrieved summaries.

## Latency plan

"Total execution time" means video in → summaries, labels and a ready chatbot.

| Lever | Expected effect | Where |
|---|---|---|
| Concurrent requests + vLLM continuous batching | Biggest win: N segments run as one batch instead of one after another | `vlm.concurrency` |
| W4A16 / W8A8 / FP8 weights | Less memory traffic, faster decode, bigger batches | `scripts/quantize/` |
| MTP (Gemma 4 drafter) | Fewer decode steps (Google reports up to ~3×; measure ours) | `scripts/vllm/serve.sh SPEC=mtp` |
| n-gram speculation for chat | Fast when the answer copies retrieved text | `SPEC=ngram` |
| Automatic prefix caching | The shared system prompt is computed once | vLLM default |
| Constrained classification | Output of a few tokens | stage 2 |
| Image token budget | Fewer image tokens → faster prefill | `vlm.tile_px`, max pixels |
| Overlapping stages | Decode, detect and summarize run in parallel threads | pipeline |

**Measured and reported:**
- per stage: decode/detect, summarize, classify, database write
- per request: TTFT, time per output token, tokens/s
- per video: total wall time
- the speculative-decoding acceptance rate (vLLM metrics)

Everything runs as a matrix of {BF16, W8A8, W4A16, FP8} × {no spec, MTP, n-gram} × concurrency {1, 4, 16}.

## Quantization study (what we will show)

| Scheme | What is quantized | Calibration | Notes |
|---|---|---|---|
| BF16 | none | none | Baseline accuracy and latency |
| W8A8-INT8 | weights and activations to int8 | ~256–512 samples (SmoothQuant + GPTQ) | Uses int8 tensor cores; usually small accuracy loss |
| FP8 (block or dynamic) | weights (and activations) to FP8 | none for block/dynamic | Needs an FP8-capable GPU (Ada/Hopper). For Gemma 4, keep the vision tower, embeddings and lm_head in BF16 (llm-compressor's Gemma 4 recipe). |
| W4A16 (GPTQ or AWQ) | weights to 4 bit, activations BF16 | ~256–512 samples | Smallest memory footprint; the most accuracy risk on small models |

Accuracy is measured with the faculty dataset's labels: summary quality and classification accuracy/F1
per scheme. Latency is measured with `aequitas bench`.

## Build order

1. ~~**Pipeline v2:** separate summarize and classify stages, concurrent VLM workers, per-stage timings.~~ Done.
2. ~~**`aequitas bench`:** latency harness for any OpenAI-compatible server.~~ Done.
3. ~~**vLLM scripts:** WSL install, `serve.sh` with quantization and speculation profiles.~~ Written (`scripts/vllm/`).
4. ~~**Quantization scripts:** llm-compressor recipes for FP8, W8A8 and W4A16.~~ Written (`scripts/quantize/`).
5. ~~**React frontend:** upload, live per-stage latency, summaries and labels, streamed chat.~~ Done.
6. ~~**Dataset adapter + evaluation.**~~ Done (`eval-dataset`); runs when the dataset arrives.
7. **GPU day (next):** install WSL + vLLM on the laptop, run the benchmark matrix (FP8 / W8A8 / W4A16 × none /
   MTP / n-gram × concurrency 1/4/16), write `docs/QUANTIZATION.md`.
8. **At the end:** re-run the Kaggle ablation for the presentation; demo script with total time on screen; final
   report.
