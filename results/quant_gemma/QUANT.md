# Gemma 4 E2B quantization on a Kaggle T4 (Ollama, 72 saved windows, variant `context`)

| Tag | Loaded size (ollama ps) | Fight / arrest / fire right | Incident windows flagged | False alarms (normal) | Broken outputs | Time / window p50 · p95 | Total |
|---|---|---|---|---|---|---|---|
| `gemma4:e2b-it-q4_K_M` | 330 MB 100% | 2/17 · 8/12 · 7/10 | 20/39 | 0/33 | 0 | 1.84 · 1.98 s | 133 s |
| `gemma4:e2b-it-q8_0` | 2.7 GB 100% | 0/17 · 8/12 · 5/10 | 20/39 | 0/33 | 0 | 2.06 · 2.19 s | 148 s |
| `gemma4:e2b-it-bf16` | 4.8 GB 100% | 0/17 · 9/12 · 5/10 | 20/39 | 0/33 | 0 | 4.11 · 4.33 s | 295 s |
| `gemma4:e2b-it-qat` | 1.7 GB 100% | 0/17 · 6/12 · 6/10 | 18/39 | 0/33 | 0 | 1.89 · 2.05 s | 138 s |
