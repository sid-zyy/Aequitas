# Minerva handover (written 2026-10-15, for the GPU laptop)

This file is the single starting point. It covers everything decided so far, what exists, what has been
measured, and the complete plan. Details live in the linked docs. **§6 is the working checklist.**

> **⚠️ The hackathon faculty's requirements (R5 below) come first, always.**
> - three stages: summarization → classification of the summary → chatbot
> - vLLM
> - speculative decoding + multi-token prediction
> - 4-bit and 8-bit quantization study
> - React + FastAPI, no Jupyter
> - the faculty dataset
>
> **Judging:** approach, end-to-end demo, and **latency (total execution time), weighted most heavily**.
> Product work (Parts 1–3) only happens once these are covered. CLAUDE.md repeats this at the top.

---

## 1. What Minerva is

An open-source, local-only system that takes video (files, RTSP, webcam), finds and summarises incidents,
classifies them, stores them with key frames, and answers questions about them in a chatbot. It is being
built for a **hackathon** (the Honeywell problem statement, judged by faculty) and shaped into a
**police-oriented product for India**.

## 2. Every requirement so far, and where it came from

| # | Source | Requirement |
|---|---|---|
| R1 | Honeywell problem statement (PDF) | Ingest RTSP/webcam/file with frame sampling. Detect and track (e.g. YOLO). Summarise with an open-source VLM (Gemma, Qwen-VL, LLaVA, SmolVLM). Classify into person down, fighting, tailgating, crowd incident, suspicious activity, with confidence. Store timestamp, camera ID, category, summary and key frame. Natural-language chatbot ("Show all tailgating events from camera 2 today"). **Open-source only, no paid cloud APIs. One consumer GPU or laptop; quantized models allowed.** Deliver a working prototype plus an accuracy and latency report. |
| R2 | Team meeting slide | **Gemma 4 is the candidate VLM.** YOLO tracking is **optional**. Multi-camera. The chatbot is model-independent ("any local open-source LLM"). **Grounded response = summary + event references.** |
| R3 | Owner: product split | **Part 1 General** (supporting: everyday video, e.g. fire). **Part 2 Police bodycam review** (major: mark key timestamps for officers). **Part 3 CCTV public order** (major: riots, stone pelting). Indian context. **Inference for every frame or timestamp.** |
| R4 | Owner: decisions | Bodycam audio languages: **English, Hindi, Kannada**. **No footage partner**: use public datasets, staged recordings and synthetic data. **A GPU laptop** is the target machine. Kaggle may be used for training. |
| **R5 ⚠️** | **Hackathon faculty: TOP PRIORITY, never drop** | **Three tasks: (1) summarization, (2) classification of the summarized info, (3) chatbot.** Serve models with **vLLM**. Use **speculative decoding** and **multi-token prediction** for faster inference. **Study and implement 4-bit and 8-bit quantization.** **React** frontend and **FastAPI** backend, **no Jupyter notebooks**. **The faculty will provide a dataset soon.** Judged on **approach**, **end-to-end demonstration**, and **latency (total execution time)**. |
| R6 | Owner + design | Guardrails: no face recognition or identification, no profiling by religion, caste, party or attire, peaceful gatherings are not incidents, a human confirms incidents. See CLAUDE.md and docs/POLICE_EDITION.md §8. |

**Priority order:** R5 (hackathon deliverables) first, then R3/R4 product depth.

## 3. What exists today

Repo: https://github.com/sid-zyy/Minerva. 59 tests pass (`pytest`).

| Area | Status | Where |
|---|---|---|
| Ingest (file/RTSP/webcam, 4 fps sampling, per-camera thread, ring buffer) | Done | `minerva/ingest/` |
| YOLO11n + ByteTrack, pose on crops, motion energy | Done; **weak on low-res CCTV** (see §4) | `minerva/vision/` |
| Rules: person down, fighting, tailgating, overcrowding, loitering, abandoned object | Done, unit-tested; **fail on CAVIAR** (§4) | `minerva/triggers/` |
| VLM check: 2×2 key-frame grid → category + summary → fusion (confirmed/disputed/rejected) | Done (category and summary in one call; **R5 needs these as two stages**) | `minerva/understand/` |
| Periodic mode (no YOLO, VLM reviews every window) | Done | `understand/periodic.py`, `analysis.mode` |
| Model-independent clients: Ollama + any local OpenAI-compatible server (**vLLM-ready**), non-local hosts refused | Done | `understand/vlm_client.py` |
| SQLite store, FTS5, uploads table, per-stage timings table | Done | `minerva/store/` |
| Chat: LLM → structured filter (reconciled with keywords) → SQL → `[#id]` grounded answer + optional cited summary | Done (not streamed yet) | `minerva/chat/` |
| Web app: FastAPI API + **vanilla JS** frontend (upload, jobs, chat, event log, overview) | Done; **R5 needs React** | `minerva/web/` |
| CLI: `init-db`, `run`, `ask`, `web`, `spike`, `stats`, `eval-report` | Done | `minerva/cli.py` |
| CAVIAR evaluation (11 clips, expected labels, scorer) | Done | `config/caviar_*.yaml`, `scripts/get_caviar.sh`, docs/EVAL_CAVIAR.md |
| Trained-model loaders (violence clip ONNX, hazard YOLO) + fight-from-clip-score path | Done, tested with fakes | `vision/violence.py`, `vision/hazards.py` |
| Training scripts: violence clip classifier (r3d_18), hazard detector (YOLO11 fire/smoke/knife/gun/stick) | Written, smoke-tested on synthetic data, **never trained on real data** | `kaggle/*.py` (the `.ipynb` copies are only for Kaggle; R5 says no Jupyter, so run the `.py` directly on the GPU) |
| **Pipeline v2 (separate summarize → classify), vLLM serving, speculative decoding/MTP, quantization, latency bench, React** | **Not started** | Plan in docs/HACKATHON_PLAN.md and §6 below |

**Docs:** DESIGN.md (original design + M0 numbers), HACKATHON_PLAN.md (R5 architecture), ROADMAP.md
(three parts), POLICE_EDITION.md (Part 3 taxonomy, datasets, guardrails), EVAL_CAVIAR.md (real-footage results).

## 4. What has been measured (dev laptop, CPU only, i9-13900H, no NVIDIA GPU)

| Thing | Result |
|---|---|
| YOLO11n per frame | 43–82 ms |
| gemma4:e2b, uncached call, 896 px 2×2 grid | ~16 s (qwen2.5vl:3b: ~44 s). Gemma is the default. |
| Chat question → filter | ~6 s |
| CAVIAR, rules (3 configs) | **0/6** expected incidents found, 0 false alarms |
| CAVIAR, Gemma E2B window review only | Said `person_down` on **11/11 clips**; precision 0.27, recall 0.50; never recognised a fight |

**Root causes (EVAL_CAVIAR.md):**
- YOLO11n person confidence is 0.16–0.27 on 384×288 footage, and it misses the faller while they walk.
- It detects no bags in the left-bag clips.
- Fight pairs are never tracked together.
- The small VLM is biased towards "lying on the ground" for small, high-angle people.

**Not yet tried:** larger YOLO (11s/11m, bigger `imgsz`), `gemma4:e4b`, any GPU numbers.

## 5. Set up the GPU laptop

1. Clone and install (Windows):
   ```bash
   git clone https://github.com/sid-zyy/Minerva.git && cd Minerva
   python -m venv .venv
   ```
   Install the **CUDA build of PyTorch** first (pick the command for your CUDA version at pytorch.org),
   then:
   ```bash
   .venv\Scripts\python -m pip install -e .[dev] onnx
   .venv\Scripts\python -m pytest
   ```
2. Ollama (dev fallback): `ollama pull gemma4:e2b` (~4.6 GB), and later `gemma4:e4b`.
3. Smoke-check the app: `python -m minerva web`, upload a clip, ask a question.
4. **vLLM goes in WSL2 Ubuntu** (vLLM needs Linux + an NVIDIA GPU):
   - an NVIDIA Windows driver with WSL support
   - in Ubuntu: Python venv + `pip install vllm`
   - a Hugging Face token: Gemma models are gated, so accept the licence on HF first
   - **Look up the exact HF model IDs** for Gemma 4 E2B/E4B instruct and their **MTP drafter** models. Don't guess them.
   - Serve on `0.0.0.0:8000` (or another port).
   - In Minerva's `config/settings.yaml`, set `vlm.backend: openai`, `vlm.host: http://127.0.0.1:<port>`,
     `vlm.model: <served name>`. WSL2 forwards localhost by default.
5. Record GPU facts in HACKATHON_PLAN.md: model name, VRAM, compute capability. **FP8 needs compute ≥ 8.9 (Ada/Hopper).**

## 6. The plan (checklist, in order)

Rules for every phase: tests pass before committing; commit locally; **ask the owner before pushing**;
record numbers in the docs.

### Phase 0 · GPU baseline (half a day)
- [ ] Setup (§5). Run `minerva spike --models gemma4:e2b gemma4:e4b` with CUDA PyTorch. Record the results in DESIGN.md §4.3.
- [ ] Re-run CAVIAR with **YOLO11s and YOLO11m** at `imgsz` 640 and 960. Does person recall fix the rules?
  Record the results in EVAL_CAVIAR.md. (The weights are small and come from Ultralytics.)

### Phase 1 · Pipeline v2: the three faculty tasks (R5)
- [ ] **Stage 1, Summarize:** the VLM returns only `{summary, visible_evidence}`. This is a new prompt and
  model in `understand/prompts.py`. Keep the "describe only what is visible" rule.
- [ ] **Stage 2, Classify the summary:** a text-only call, constrained to the category list + `none`.
  - vLLM: `extra_body={"structured_outputs": {"choice": [...]}}` plus `logprobs` for confidence.
  - Ollama: JSON-schema enum, no confidence.
  - Add `classify_text()` to `LlmClient`.
  - Fusion (`understand/fusion.py`) and `WindowAggregator` take the classifier's label instead of the VLM's.
  - Later, train a small text classifier on the faculty dataset and compare accuracy and latency.
- [ ] **Stage 3, Chatbot:** add a streaming endpoint (SSE) for answers. Keep the grounding checks.
- [ ] **Concurrency:** `vlm.concurrency` worker threads, so vLLM batches requests. In periodic mode, keep
  each camera's windows in order (the aggregator assumes it).
- [ ] **Timings:** split `vlm_ms` into `summarize_ms` + `classify_ms`. Add an upload job `started_at` so the
  **total execution time** is exact. Add a small schema migration (`ALTER TABLE … ADD COLUMN` if missing).
- [ ] Tests: fake clients, stage separation, fusion with classifier labels, ordering under concurrency.

### Phase 2 · vLLM, speculative decoding, MTP, latency bench (R5)
- [ ] `scripts/vllm/install_wsl.sh`, plus `scripts/vllm/serve.sh` with env profiles:
  `MODEL=… QUANT=bf16|fp8|w8a8|w4a16 SPEC=none|mtp|ngram`, using
  `--speculative-config '{"method":"mtp","num_speculative_tokens":N}'` (Gemma 4 MTP, no separate draft),
  or `{"method":"ngram","num_speculative_tokens":N,"prompt_lookup_max":K}`. Check the flag names against the
  installed vLLM version's docs.
- [ ] `minerva bench`: for any OpenAI-compatible server, stream requests and measure TTFT, time per output
  token, tokens/s, total time, at concurrency 1/4/16, for all three tasks. Use a **unique image per request**
  (cache trap). Read the speculative acceptance rate from vLLM's `/metrics`. Write a JSON + markdown report
  to `data/reports/`.
- [ ] Turn on automatic prefix caching (shared system prompts) and set an image pixel budget. Measure both.

### Phase 3 · Quantization study (R5)
- [ ] `scripts/quantize/quantize.py` with llm-compressor recipes:
  - **FP8** (block/dynamic, no calibration; for Gemma 4, keep the vision tower, embeddings and lm_head in BF16)
  - **W8A8-INT8** (SmoothQuant + GPTQ, ~256–512 calibration samples)
  - **W4A16** (GPTQ or AWQ)

  Check the current llm-compressor API and its Gemma 4 / multimodal examples first.
- [ ] Benchmark the matrix {BF16, W8A8, FP8 (if supported), W4A16} × {no spec, MTP, n-gram}:
  - latency, from Phase 2's bench
  - accuracy: classification accuracy/F1 and summary quality on the faculty dataset, or CAVIAR plus a small
    hand-labelled set until it arrives
- [ ] Write `docs/QUANTIZATION.md`: method, results table, chosen config and why. The faculty judge the approach, so explain it.

### Phase 4 · React frontend (R5)
- [ ] `frontend/`: Vite + React + TypeScript. Port the existing pages (Overview, Analyze video, Ask, Event
  log, event detail with key frames and video at the event offset) from `minerva/web/static/`.
  - The API stays the same (`/api/*`, documented at `/api/docs`).
  - Add a **live latency panel**: per-stage timings and the total time per video.
  - Add **streamed chat**.
- [ ] FastAPI serves the built `frontend/dist`. Then delete the vanilla `static/` UI. Keep `npm run dev`
  with an API proxy for development.

### Phase 5 · Faculty dataset (R5): when it arrives
- [ ] Write an adapter for its format, and an `eval` command for all three tasks:
  - summarization: ROUGE-L / BERTScore against references if provided, otherwise a rubric
  - classification: accuracy, macro-F1, confusion matrix
  - chatbot: a question set with expected event IDs or answers
- [ ] If its categories differ from Minerva's five, make the category list configurable. It is currently a
  hard-coded enum in `types.py`, the DB `CHECK` constraint, prompts and the UI.

### Phase 6 · Accuracy work
- [ ] Train the **violence clip classifier** and **hazard detector** on the GPU, running `kaggle/*.py` directly
  (they detect CUDA). Datasets: RWF-2000, Real Life Violence Situations; D-Fire (**check its class order**),
  a knife/pistol set. Lathi/stick images must be self-labelled.
- [ ] Turn them on via `models.violence_dir` / `models.hazard_dir`, then re-run CAVIAR.
- [ ] VLM bias: try `gemma4:e4b`, single higher-resolution frames instead of a grid, and a prompt that asks
  first whether anyone is lying down.

### Phase 7 · Product parts (R3/R4), after the hackathon deliverables
- [ ] Profiles (General / Bodycam / CCTV) and a **per-frame inference table + timeline UI** (ROADMAP.md step 2).
- [ ] Part 1: fire/smoke from the hazard model.
- [ ] Part 2, bodycam:
  - audio pipeline: faster-whisper for EN/HI, plus an Indic ASR model (AI4Bharat) for Kannada after checking
    its licence; an audio-event classifier; a phrase list in EN/HI/KN including code-mixed speech
  - key-moment fusion, a review UI, report export, blurring of bystander faces and plates
  - staged footage recorded by the team, with consent forms
- [ ] Part 3, CCTV riots: violence clip model, stone-pelting (throwing pose + flow), crowd density and crush
  risk, an Indian test set. There is **no real Indian riot footage**, so the report must say accuracy on real
  riots is unmeasured.

### Phase 8 · Demo and report
- [ ] A scripted end-to-end demo: upload → live stages → labels → chat questions. It must show total
  execution time on screen.
- [ ] A final report: architecture, quantization and speculation study, latency table (the faculty weight it
  heavily), accuracy, and limitations.

## 7. Open items and waiting on
- **Faculty dataset:** format and categories are unknown.
- GPU model and VRAM: decides whether FP8 is possible and how big the models can be.
- D-Fire class order, and the exact HF IDs for Gemma 4 models and MTP drafters: verify, don't assume.
- One local commit on the dev laptop may not be pushed yet. Check `git log origin/main..` on the dev laptop if something seems missing.
