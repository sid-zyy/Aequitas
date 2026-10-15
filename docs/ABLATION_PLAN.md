# Ablation plan: 3 models × 16/8/4-bit (cheap, for the presentation)

**Goal.** Choose the VLM and precision before fine-tuning or adding more features. Show the faculty a
small, honest table and one chart: **latency vs accuracy vs memory** for each model × precision. This
covers the quantization-study requirement and justifies the model we pick.

**Budget.** $0, using Kaggle's free GPUs. About 3–4 GPU hours in one session. Kaggle gives ~30 GPU
hours per week, so this uses about a tenth of the weekly quota.

## 1. The matrix

| | 16-bit | 8-bit | 4-bit |
|---|---|---|---|
| **Gemma 4 E2B-it** (`google/gemma-4-E2B-it`; 5.1 B total / 2.3 B effective; the faculty's candidate, has MTP) | FP16 | W8A8-INT8 | W4A16 (GPTQ) |
| **Qwen3-VL-2B-Instruct** (`Qwen/Qwen3-VL-2B-Instruct`; ~4.3 GB in BF16) | FP16 | W8A8-INT8 | W4A16 (GPTQ) |
| **SmolVLM2-2.2B-Instruct** (`HuggingFaceTB/SmolVLM2-2.2B-Instruct`; named in the problem statement) | FP16 | W8A8-INT8 | W4A16 (GPTQ) |

That is 9 runs, plus 2 extra for the best Gemma row: **+MTP** (the `-assistant` drafter) and **+n-gram**
speculation. These are the speculative-decoding numbers the faculty asked for.

**Why these schemes on Kaggle.** Kaggle's free GPU is a T4 (Turing). vLLM supports GPTQ/Marlin W4A16 and
INT8 W8A8 on Turing, but **not FP8 and not BF16** (vLLM quantization hardware table). So:
- 16-bit = FP16 on Kaggle.
- **FP8 runs on our RTX 4050 laptop (Ada)**, as one extra column for the chosen model only.
- If Gemma misbehaves in FP16 (overflow → empty or garbage text), that is a finding. Re-run its
  16-bit row in FP32 on both T4s (`--tensor-parallel-size 2`) and say so in the table.

## 2. What is measured (identical for all runs)

| Metric | How | Data |
|---|---|---|
| **Stage 1 latency**: summarize TTFT, TPOT, total; throughput at c=1 and c=8 | `aequitas bench --tasks summarize classify --concurrency 1 8 --requests 16` (unique image per request) | synthetic grids (latency only) |
| **Stage 2 latency + accuracy**: accuracy, macro-F1, incident precision/recall | `aequitas eval-classify` | `config/classify_eval.yaml` (48 labelled summaries) |
| **End-to-end on real footage**: summary → label, fighting vs none | stage 1 + 2 on 2×2 grids from 20 clips (10 Violence, 10 NonViolence) | Real Life Violence Situations (Kaggle) |
| **Quantization damage** to summaries | similarity (ROUGE-L) of each quantized summary to the same model's 16-bit summary of the same grid | the same 20 grids |
| **Memory** | checkpoint size on disk; GPU memory used by vLLM (`nvidia-smi`) | n/a |
| **Speculation** | acceptance rate from vLLM `/metrics` | bench run |

Every configuration runs the same requests in the same order. Ten configs at ~8 minutes each (load +
bench + eval), plus ~1 h to quantize, fits one Kaggle session.

## 3. Procedure (Kaggle notebook `kaggle/ablation.ipynb`)

1. Quantize each model to W8A8 and W4A16 with `scripts/quantize/quantize.py`. Use 128 calibration
   samples (enough for a demo, and it halves the time). Keep the vision towers, embeddings and lm_head
   in 16-bit.
2. For each of the 9 configs: start `vllm serve` → wait for `/health` → run bench + eval-classify + real
   clips → record GPU memory → stop the server.
3. Gemma's best config: repeat with `SPEC=mtp` (`NSPEC=1,2`) and `SPEC=ngram`.
4. Write `ablation/results.csv`, `ablation/ABLATION.md` (the table) and `ablation/ablation.png` (chart),
   then zip `ablation/` for download. Quantized checkpoints are optional outputs, because they are
   large. The laptop needs only the chosen one.

## 4. How we pick the model (decide before looking at the numbers)

1. Discard configs with <95 % valid outputs or with stage-2 macro-F1 more than **3 points below that
   model's own 16-bit run**.
2. It must fit the laptop: GPU memory ≤ ~5 GB, or work with a small `--cpu-offload-gb`.
3. Among the rest, pick the **lowest end-to-end time per window** (summarize + classify at c=8),
   because latency is weighted most.
4. Ties: higher incident recall (missing an incident is worse than a false alarm a human dismisses).

## 5. Honesty rules for the slide

- The 48-summary set is **hand-written by us**: it compares configs, it does not show real-world
  accuracy. Label it "development set".
- The 20 real clips are a small binary check (fighting vs none). Report n=20 next to every number.
- T4 latencies are not laptop latencies. Re-measure the chosen config on the RTX 4050 and show both.
- Report failures (a config that would not load, FP16 overflow) in the table, not in footnotes.

## 6. After the ablation

1. Re-run the chosen config on the laptop: bench + FP8 column + MTP. These are the demo numbers.
2. Faculty dataset arrives → the same `eval-classify` / real-clip evaluation on it.
3. Fine-tuning, if the chosen model is still weak on our categories:
   - **cheapest:** a small text classifier for stage 2 (ms latency), trained on summaries + labels
   - **then:** LoRA on the VLM for summaries on police footage

   Both run on Kaggle. Decide with the faculty dataset, not the 48-item dev set.
4. Then the remaining faculty features (React live dashboard, bodycam panel) on the chosen model.

## 7. Exact Kaggle steps

1. kaggle.com → **Create → New Notebook → File → Import Notebook** → upload `kaggle/ablation.ipynb`.
2. Right panel → **Session options**: Accelerator **GPU T4 x2**, Internet **On**, Persistence: Files.
3. **Add Input** → search "Real Life Violence Situations" → add it. Optional; without it the
   real-clip column is skipped.
4. (Optional, faster downloads) **Add-ons → Secrets** → `HF_TOKEN` = a Hugging Face read token.
5. **Run all.** The first cell prints the GPU, and the plan cell prints the configs and an estimated
   time. Leave the tab open or use **Save Version → Save & Run All (Commit)** so it runs unattended
   (counts against GPU quota, max 12 h).
6. When it finishes: **Output** panel → download `ablation.zip` → unzip into `Aequitas/data/ablation/`
   → put `ABLATION.md` and `ablation.png` on the slide.
7. Out of quota or time? Set `MODELS = ["gemma"]` and `PRECISIONS = ["fp16", "w4a16"]` in the config
   cell: the minimum that still shows a 16-vs-4-bit comparison (~1 h).
