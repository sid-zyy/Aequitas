# Speculative decoding with MTP, measured without vLLM (2026-10-16)

> **2026-10-17:** the demo model is now `qwen3-vl:2b-instruct` (results/qwen/README.md). In Ollama, speculative
> decoding only works through an MTP drafter shipped with the model; Qwen3-VL has none, and llama.cpp refuses a
> draft model when an image projector is loaded. So the Qwen demo runs without speculation, and this page remains
> the measurement of speculative decoding, made on Gemma 4 E2B.

vLLM can't hold Gemma 4 E2B on the laptop's 6 GB GPU, so speculative decoding was measured in
**Ollama**, which the live demo actually uses. Ollama's `gemma4:e2b` ships Gemma 4's **MTP drafter** as a separate
98 MB layer (`application/vnd.ollama.image.draft`, the `-assistant` checkpoint) next to the 3.5 GB Q4_K_M model.
At each step the drafter proposes `draft_num_predict` tokens, and the main model checks all of them in one forward
pass and keeps the ones it agrees with. The model's default is 3.

Setting: `vlm.draft_num_predict` in `config/settings.yaml` (null = the model's default; 0 = off), sent per request
by `OllamaClient`. Benchmark: `scripts/bench_spec.py`; raw output in `results/speculative/`.

## Setup
- RTX 4050 Laptop (6 GB), Ollama 0.40.4, gemma4:e2b Q4_K_M, temperature 0, `think` off. Nothing else running on the GPU.
- The app's own prompts and client, three workloads:
  - **stage 1**, summary: the 72 saved 2×2 window grids (`data/bench_windows`: fights, arrests, fire, normal streets). Image in, about 36 tokens of JSON out.
  - **stage 2**, label: one-word JSON from that summary.
  - **chat**: a grounded answer over 10 stored summaries, streamed as in the app (12 questions).
- Each setting ran every input twice: settings in order 0→6, then 6→0, so warm-up and heat affect all of them alike.
  The results below are medians over both rounds: 144 windows and 24 chat answers per setting.

## Results

| drafter tokens | stage 1 s | stage 1 decode tok/s | stage 2 s | stage 2 decode tok/s | chat s | chat decode tok/s | **window total s** |
|---|---|---|---|---|---|---|---|
| 0 (off) | 0.875 | 82 | 0.402 | 61 | 0.58 | 89 | **1.276** |
| 1 | 0.815 | 98 | 0.376 | 78 | 0.47 | 125 | 1.193 |
| 2 | 0.782 | 107 | 0.368 | 80 | 0.46 | 150 | 1.154 |
| **3 (default)** | 0.784 | 106 | 0.366 | 88 | **0.45** | **165** | **1.155** |
| 4 | 0.815 | 98 | 0.384 | 78 | 0.46 | 160 | 1.206 |
| 6 | 0.781 | 84 | 0.397 | 69 | 0.44 | 133 | 1.168 |

With 3 drafted tokens against none:
- **Decoding is 1.3× faster for summaries and 1.85× faster for chat answers** (82 → 106 and 89 → 165 tokens/s).
- **Time per window: 1.28 s → 1.16 s, 9.5 % less.** Stage 1 writes only about 36 tokens, and most of its time is
  spent reading the image (prefill), which speculation doesn't touch. So the end-to-end gain is much smaller than
  the decode speed-up.
- **Chat answers: 0.58 s → 0.45 s, 22 % less.** They're longer and quote the stored summaries, which the drafter
  predicts well.
- 2 and 3 tie. Past 3, more of each draft gets rejected and decoding slows again (6: 84 tok/s for summaries).

## Does it change the answers?

Without the drafter, both rounds gave identical outputs (144/144 summaries), so the run is deterministic. With the
drafter, the **wording** of about 28 % of summaries and 17 % of chat answers changed (draft 3: 104/144 and 20/24
identical). Speculative decoding with temperature 0 should not change outputs. Ours does, most likely because
checking a batch of drafted tokens uses slightly different 4-bit GPU arithmetic than one token at a time. That's
enough to flip a near-tie between two words, and the rest of the sentence then follows a different path.

**The labels didn't move in any way that matters:** 142/144 identical. The right answers were 44/72 with the
drafter off and 44–45/72 at every setting. False alarms on normal street windows were 0/33 at every setting.

## What we use, and what wasn't tested
- **The app keeps the default, 3 drafted tokens.** It was already on, because Ollama enables the drafter by default.
  `config/settings.yaml` now says so explicitly.
- **n-gram speculation** (drafting from the prompt itself, useful for chat answers that quote summaries) isn't
  available in Ollama. It's in the vLLM profiles (`scripts/vllm/serve.sh SPEC=ngram`) but not measured, because vLLM
  doesn't fit this GPU.
- **Acceptance rate** isn't reported by Ollama. The decode tokens/s above is the measurable effect.
- These are single requests (one camera window at a time). Batched serving isn't covered.
