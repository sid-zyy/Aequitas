# Bench levers-concurrency

Model `m` at `http://127.0.0.1:8000`, 2026-10-15T16:07:36, 8 requests per level.

| task | conc. | ok | wall s | req/s | out tok/s | total p50 ms | TTFT p50 ms | TPOT p50 ms | spec. accept |
|---|---|---|---|---|---|---|---|---|---|
| summarize | 1 | 8/8 | 17.1 | 0.468 | 34.9 | 2092.6 | 564.5 | 20.5 | - |
| summarize | 4 | 8/8 | 8.7 | 0.925 | 64.1 | 4207.4 | 1252.8 | 38.9 | - |
| summarize | 8 | 8/8 | 7.6 | 1.046 | 73.6 | 6691.1 | 3550.6 | 46.9 | - |
| classify | 1 | 8/8 | 1.2 | 6.461 | 21.8 | 152.1 | 116.9 | 8.3 | - |
| classify | 4 | 8/8 | 0.8 | 9.734 | 34.1 | 383.1 | 236.3 | 44.9 | - |
| classify | 8 | 8/8 | 0.7 | 10.92 | 34.1 | 682.4 | 545.0 | 43.6 | - |
