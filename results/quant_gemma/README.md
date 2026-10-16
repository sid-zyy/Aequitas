# Gemma 4 E2B at four precisions (Kaggle T4, Ollama, 2026-10-16)

Notebook: `kaggle/quant_gemma.py`. Each tag was pulled, warmed up on a random image, run through
`scripts/bench_variants.py --variants context` on the 72 saved windows (`kaggle/bench_windows.zip`: 39 incident
windows, 33 normal), then removed. Raw output: `QUANT.md`, `summary.json`, `rows-*.jsonl` (one row per window).
`ollama.log` and `commands.log` are kept locally and gitignored.

## Results

GPU memory is read from the llama-server log (model weights on CUDA + vision encoder 941 MiB + KV and compute
buffers). `ollama ps` sizes are listed for completeness but don't match the log (it reports 330 MB for
Q4_K_M), so they aren't used.

| Tag | Weights | GPU memory (log) | `ollama ps` | Incident windows flagged | False alarms | Fight / arrest / fire label right | Same label as BF16 | Time / window p50 · p95 |
|---|---|---|---|---|---|---|---|---|
| `e2b-it-bf16` | BF16 | ~5.5 GB (+0.75 GB pinned host) | 4.8 GB | 20/39 | 0/33 | 0/17 · 9/12 · 5/10 | – | 4.11 · 4.33 s |
| `e2b-it-q8_0` | Q8_0 | ~3.5 GB | 2.7 GB | 20/39 | 0/33 | 0/17 · 8/12 · 5/10 | 69/72 | 2.06 · 2.19 s |
| `e2b-it-q4_K_M` | Q4_K_M | ~2.7 GB (incl. MTP drafter) | 330 MB | 20/39 | 0/33 | **2/17 · 8/12 · 7/10** | 65/72 | **1.84** · 1.98 s ⚠ |
| `e2b-it-qat` | Q4_0, quantization-aware trained | ~2.5 GB | 1.7 GB | 18/39 | 0/33 | 0/17 · 6/12 · 6/10 | 65/72 | 1.89 · 2.05 s |

The per-layer embeddings stay in CPU-mapped memory at every precision (1.8–4.5 GB), as usual for Gemma's E models.

## Read this before quoting the numbers

- **⚠ Only Q4_K_M ran with the MTP drafter** (`--spec-type draft-mtp --spec-draft-n-max 3` in its llama-server
  command; the other three started without it). Part of its lead over Q8_0 is speculative decoding. QAT (also
  4-bit, no drafter) at 1.89 s is the fair 4-bit vs 8-bit comparison: **8 % faster than Q8_0**.
- **Quantization cost no detection here.** BF16, Q8_0 and Q4_K_M flag the same 20/39 incident windows with 0/33
  false alarms. QAT flags 2 fewer.
- **Labels differ a little.** Q4_K_M is the only one that names any fight (2/17) and gets the most fires right
  (7/10). With 17 fight windows and 10 fire windows, one window is 6–10 points, so this is not a clear win.
- **BF16 is 2.2× slower** than Q8_0 on the T4, and at ~5.5 GB it wouldn't leave room for the SigLIP gate on the
  laptop's 6 GB. Not tried on the laptop.
- **Times are on a Kaggle T4**, not the RTX 4050. The laptop's Q4_K_M time on the same windows is 1.16 s
  (docs/SPECULATIVE.md).
- The 72 windows include the six faculty clips; nothing was tuned in this run.

## Decision

Keep Q4_K_M with the MTP drafter: no detection lost against BF16, the smallest on the GPU after QAT, the fastest,
and the best fight/fire labels of the four. Q8_0 also fits the laptop (~3.5 GB) and agrees with BF16 more often
(69/72), but is slower and no more accurate on these windows.
