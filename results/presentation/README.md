# Numbers measured for the presentation (2026-10-16, RTX 4050 laptop, Ollama gemma4:e2b Q4_K_M, MTP draft 3)

## The six faculty clips, end to end (`config/eval_faculty.yaml`, `config/faculty_cameras.yaml`)

Current setup: periodic 4 s windows, previous-window context, SigLIP scene gate, violence clip votes.
Run twice; both runs gave identical labels. Scored with `eval-report --expected config/faculty_expected.yaml`.

| Clip | Expected | Result |
|---|---|---|
| v001 | fighting | fighting, confirmed |
| v003 (3 s, one window) | fighting | person_down, unverified (missed) |
| v006 | fighting | fighting, confirmed |
| v011 | fighting | fighting, confirmed |
| v026 | fighting | fighting, confirmed |
| w001 (weapon clip) | fighting or weapon | chase, confirmed (false alarm: wrong label) |

Confirmed: 4 hits, 1 false alarm, 1 miss (precision 0.80, recall 0.80). The fighting labels come from the
violence clip model's votes; Gemma's own summaries describe people standing or sitting (Gemma alone: 0/5).

| | Run 1 (cold: model loads) | Run 2 (warm) |
|---|---|---|
| Wall time, 6 clips (38.8 s of footage) | 40 s | **25 s** |
| Summarize p50 / p95 | 858 / 10 554 ms | 579 / 674 ms |
| Classify p50 | 423 ms | 317 ms |
| Both stages p50 | 1 301 ms | 906 ms |
| Window end to end p50 (incl. queue) | 19.1 s | 5.1 s |

## Per-call costs of the second signals (scratch script, faculty frames)

| Model | Device | Time |
|---|---|---|
| Violence clip model (r3d_18 ONNX, 8 frames) | CPU (onnxruntime has no CUDA provider here) | 90 ms per score (+19 ms preprocessing), every 0.5 s per camera |
| SigLIP scene gate, 4 frames | GPU | 44 ms (CPU: 448 ms). The earlier 17 ms figure was measured alone; this run shared the GPU with Ollama |

## Text tasks (`aequitas.understand.recording_summary`, `aequitas.translate`)

| Task | Inputs | p50 |
|---|---|---|
| Bodycam whole-recording summary | 6 recordings, 2–20 windows | 1.6 s (1.1–2.6 s) |
| Citizen report translation, Hindi → English | 6 reports | 0.31 s |
| Citizen report translation, Kannada → English | 6 reports | 0.34 s |

Translations were checked by eye for the 6 sample reports: facts, places and times were kept.
