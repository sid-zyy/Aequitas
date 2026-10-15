# Bench levers-concurrency

Model `m` at `http://127.0.0.1:8000`, 2026-10-15T17:11:22, 8 requests per level.

| task | conc. | ok | wall s | req/s | out tok/s | total p50 ms | TTFT p50 ms | TPOT p50 ms | spec. accept |
|---|---|---|---|---|---|---|---|---|---|
| summarize | 1 | 8/8 | 14.9 | 0.536 | 36.8 | 1852.4 | 515.6 | 19.3 | - |
| summarize | 4 | 8/8 | 8.5 | 0.942 | 68.9 | 3631.4 | 1052.8 | 35.0 | - |
| summarize | 8 | 8/8 | 6.9 | 1.166 | 69.2 | 5503.4 | 3092.4 | 45.6 | - |
| classify | 1 | 8/8 | 1.4 | 5.725 | 20.8 | 168.8 | 132.6 | 9.6 | - |
| classify | 4 | 8/8 | 1.0 | 7.91 | 29.7 | 433.3 | 251.8 | 15.4 | - |
| classify | 8 | 8/8 | 0.9 | 8.878 | 32.2 | 844.1 | 551.6 | 56.1 | - |
