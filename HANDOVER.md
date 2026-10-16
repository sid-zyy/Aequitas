# Aequitas handover (updated 2026-10-16)

The single starting point for a new session, human or Claude. It covers what is decided, what exists, what is
measured, what is running, and what to do next. **How the system works:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).** §6 is the working checklist.

> **⚠️ The hackathon faculty's requirements (R5) come first, always.**
> - three stages: summarization → classification of the summary → chatbot
> - vLLM serving
> - speculative decoding + multi-token prediction, with the speed-up measured
> - 4-bit and 8-bit quantization study
> - React + FastAPI, no Jupyter
> - the faculty dataset
>
> **Judging:** approach, end-to-end demo, and **latency (total execution time), weighted most heavily**.
> CLAUDE.md repeats this at the top.

---

## 1. What Aequitas is

An open-source, local-only system that takes video (files, RTSP, webcam, bodycam recordings, citizen uploads),
summarises and labels what happens, stores events with key frames and timings, and answers questions about
them in a grounded chatbot. Built for a **hackathon** (the Honeywell problem statement, judged by faculty) and
shaped into a **police product for India**: live CCTV, bodycam review, citizen reports.

## 2. Requirements and where they came from

| # | Source | Requirement |
|---|---|---|
| R1 | Honeywell problem statement (PDF) | Ingest RTSP/webcam/file with frame sampling. Detect and track (e.g. YOLO). Summarise with an open-source VLM. Classify into person down, fighting, tailgating, crowd incident, suspicious activity, with confidence. Store timestamp, camera ID, category, summary, key frame. Natural-language chatbot. **Open-source only, no paid cloud APIs, one consumer GPU.** A working prototype plus an accuracy and latency report. |
| R2 | Team meeting | Gemma 4 is the candidate VLM. YOLO is optional. Multi-camera. The chatbot is model-independent. Grounded response = summary + event references. |
| R3 | Owner: product split | Part 1 General (supporting). **Part 2 Police bodycam review** (major). **Part 3 CCTV public order** (major). Indian context. Inference for every frame or timestamp. |
| R4 | Owner: decisions | Bodycam audio: English, Hindi, Kannada. No footage partner: public datasets, staged recordings, synthetic data. A GPU laptop is the target. Kaggle may be used for training. |
| **R5 ⚠️** | **Hackathon faculty: TOP PRIORITY** | **(1) summarization, (2) classification of the summarized info, (3) chatbot.** **vLLM.** **Speculative decoding + MTP.** **4-bit and 8-bit quantization.** **React + FastAPI, no Jupyter.** **The faculty dataset.** Judged on **approach**, **end-to-end demo**, **latency (total execution time)**. |
| R6 | Owner + design | Guardrails: no face recognition or identification; no profiling by religion, caste, party or attire; peaceful gatherings are not incidents; a human confirms incidents (docs/POLICE_EDITION.md §8). |

### Decisions made
- **The model is Gemma 4 E2B** (`gemma4:e2b`). It's the default since 2026-10-15, it fits the 6 GB GPU, and the
  fine-tune distils 12B into it. The Kaggle ablation's automatic pick (Qwen3-VL-2B) only reflects that Gemma
  can't start on a T4 (docs/ARCHITECTURE.md §10). **The ablation is re-run at the end only, for the
  quantisation and speculation tables in the presentation.**
- **Periodic mode is the main mode** (live CCTV, uploads, bodycam). On low-res CCTV the rules found 0 of 6
  CAVIAR incidents, because YOLO11n misses small people.
- **Second signal:** the violence clip model (5/5 faculty fights; Gemma alone 0/5) plus agreement across
  windows.
- **Tracking:** stay with a ByteTrack-family tracker; DeepSORT was rejected (docs/TRACKING.md).

## 3. What exists today

Repo: https://github.com/sid-zyy/Aequitas. **187 tests pass** (`pytest`; synthetic data, no model downloads).

| Area | Status | Where |
|---|---|---|
| Ingest: file / RTSP / webcam / **browser webcam push**, 4 fps, ring buffer | Done | `aequitas/ingest/` |
| **Three separate stages** with their own timings | Done | `understand/stages.py`, `prompts.py`, `vlm_client.py` |
| Stage 2: configurable categories (14 police labels + none), previous-window context, vLLM `choice` decoding + logprob confidence | Done | `taxonomy.py`, `vlm_client.py` |
| Stage 3: chat with LLM form + keyword reconcile, grounded `[#id]` answers, **SSE streaming** | Done | `aequitas/chat/`, `/api/ask/stream` |
| Periodic mode with blank / static skips, **SigLIP scene gate**, concurrency, in-order results | Done | `pipeline.py`, `understand/gate.py` |
| Triggered mode: YOLO11n + ByteTrack + 6 rules + fusion | Done; weak on low-res CCTV | `vision/`, `triggers/` |
| Violence clip model (r3d_18 ONNX), clip votes | Done, in use (`models/violence_clip/`, not in git) | `vision/violence.py` |
| Bodycam review: timeline, officer confirm/dismiss (audited), whole-recording summary, printable report | Done | `web/bodycam.py`, `understand/recording_summary.py` |
| Citizen reports + AI-generated likelihood + sha256 | Done | `reports.py`, `authenticity.py` |
| Police auth (scrypt, sessions, lockout, audit) | Done | `auth.py` |
| **React frontend** (Live CCTV, Active requests, Bodycam, Reports, Citizen portal, Analyst chat) served by FastAPI | Done | `frontend/` → `aequitas/web/frontend_dist/` |
| vLLM serve profiles (FP8 / quantised path × none / MTP / n-gram), WSL install script | Written; **run on Kaggle only** | `scripts/vllm/` |
| Quantisation (FP8 / W8A8 / W4A16, llm-compressor) | Written; **no successful run yet** | `scripts/quantize/quantize.py` |
| `aequitas bench` (TTFT, TPOT, tokens/s, acceptance rate) | Done | `eval/bench.py` |
| **Faculty dataset adapter** (`eval-dataset`: folder or manifest → all stages → accuracy, F1, summary similarity, time) | Done, waiting for the dataset | `eval/dataset.py` |
| Gemma fine-tune (12B → E2B distillation, QLoRA) | Run 1 failed at stage 5; run 2 ready | `kaggle/finetune_gemma.py` |
| Skeleton action model for the webcam (YOLO pose + STGCN++ NTU-120) | **Prototype, not wired in** | `vision/skeleton.py`, `scripts/skeleton_check.py` |
| Tracker comparison on CAVIAR ground truth | Done | `scripts/tracker_compare.py`, docs/TRACKING.md |
| Detector comparison (YOLO11/26, RT-DETR, RF-DETR, D-FINE, DEIMv2) on CAVIAR + faculty samples | Done; not integrated | `scripts/detector_compare.py`, docs/DETECTION.md |

## 4. Key numbers (for the slides)

All on the RTX 4050 laptop with Ollama unless noted. Details are in the linked files.

| Measure | Value | Source |
|---|---|---|
| One window, both stages (warm p50) | 1.2 s (summarize 0.8 + classify 0.38) | docs/DESIGN.md §4.4 |
| End to end, 21 demo clips | 1 min 48 s; 7/10 incidents confirmed, 1 false alarm (±2 run-to-run) | results/improvements/README.md §5 |
| SigLIP gate | 17 ms/window; −28 % Gemma time; 0/39 incident windows skipped (threshold set on those windows) | results/improvements/README.md |
| Previous-window context | flagged 22 → 25/39, 0 false alarms | same |
| Violence clip model | val acc 0.95; 6/6 faculty clips; **unseen real CCTV (300 clips): recall 0.93 with 45 % false alarms at 0.51, 0.77 / 14 % at 0.9**, AUC 0.89 | results/violence_clip/README.md |
| Stage 2 (57 summaries, regression set) | accuracy 0.983, macro-F1 0.991, 1.45 s p50 | results/chat_eval/README.md |
| Chat parsing | 28/28 dev, 20/20 held out; Gemma ~0.7 s/question | same |
| Webcam false alarms (stage 2, saved windows) | 57/120 → 6/120 labels | commit e0c2791 |
| SCVD incident recall (Kaggle T4) | E2B 0.15 (1.8 s/clip), E4B 0.175, 12B 0.70 (5.1 s/clip) | kaggle/model_compare |
| Time-saved ladder, 80 s video (Kaggle T4, vLLM, Qwen FP16) | 33.1 s → 5.4 s (6.1×) | results/ablation/README.md |
| Trackers on CAVIAR | tracker choice ≤ 0.02 IDF1; YOLO11n → 11m@960 3×; DeepSORT 4–5× ID switches, 30–46 ms/frame | docs/TRACKING.md |
| YOLO11n / pose crop | 12.7 ms / 17.6 ms per frame | docs/DESIGN.md §4.4 |
| Detectors on CAVIAR (AP50, person) | YOLO11n 0.102 · YOLO26l@960 0.344 · RF-DETR-S 0.425 (~27 ms) · D-FINE-X 0.522 (75 ms) | docs/DETECTION.md |
| Detectors on real violence CCTV (118 hand-labelled people, recall/precision @0.4) | YOLO11n 0.47/0.86 · YOLO11m 0.48/0.90 · RF-DETR-S 0.70/0.87 · RF-DETR-M 0.72/0.93 | docs/DETECTION.md |

## 5. Machine setup

**GPU laptop:** RTX 4050 Laptop, 6 GB, compute 8.9, so FP8 works. Gemma 4 E2B in BF16 is ~10 GB and does
not fit unquantised.

1. Clone, then `python -m venv .venv`. Install the CUDA build of PyTorch, then
   `.venv\Scripts\python -m pip install -e .[dev,detector]`.
2. `ollama pull gemma4:e2b` (~4.6 GB). Warm it before a demo: the first call loads the model, ~107 s.
3. Violence model: unzip `violence_clip.zip` into `models/violence_clip/` (127 MB, not in git).
4. `python scripts/get_footage.py` (demo clips), `bash scripts/get_caviar.sh` (CAVIAR evaluation).
5. Frontend: the built app is committed (`aequitas/web/frontend_dist/`). After UI changes:
   `cd frontend && npm run build`. Use `npm run dev` for hot reload against the API.
6. **vLLM needs WSL2, which is not installed.** You (the owner) install it, with admin rights and a reboot:
   `wsl --install -d Ubuntu`, then `bash scripts/vllm/install_wsl.sh` inside Ubuntu. Then point
   `vlm: {backend: openai, host: http://127.0.0.1:8000, model: gemma4}` at it, using another port if the web
   app is on 8000.
   - HF model IDs (verified): `google/gemma-4-E2B-it`, MTP drafter `google/gemma-4-E2B-it-assistant`.

## 6. The plan (checklist)

Rules for every phase: tests pass before committing; commit locally; **ask the owner before pushing**; record
numbers in the docs.

### Faculty requirements (R5): do these first
- [x] Three separate stages, timed separately; streamed chat
- [x] React frontend + FastAPI; no notebooks in the deliverable (`kaggle/*.ipynb` are build artefacts of
  `.py` files, used only on Kaggle)
- [x] Per-stage timings and total time per upload shown in the UI
- [x] Faculty dataset adapter (`aequitas eval-dataset`)
- [ ] **vLLM on the laptop** (needs WSL, above): Gemma 4 E2B FP8, then `aequitas bench` at concurrency
  1/4/16 and the demo set end to end
- [ ] **MTP and n-gram speculation measured** on the laptop: speed-up and acceptance rate (`serve.sh SPEC=mtp`,
  `NSPEC` sweep)
- [ ] **Quantisation study:** W8A8 and W4A16 checkpoints (`scripts/quantize`) + FP8, latency × accuracy
  (stage-2 eval, demo set, SCVD subset). Write `docs/QUANTIZATION.md`.
- [ ] **Ablation re-run at the end** (Kaggle, re-import `kaggle/ablation.ipynb`; the torchvision fix is in
  the repo) → tables and charts for the presentation
- [ ] Faculty dataset: run `eval-dataset` when it arrives; make its categories the label set if they differ
- [ ] Demo script: upload → live stages → labels → chat, **total execution time on screen**; final report and
  latency slide

### Accuracy (in progress)
- [ ] Fine-tune run 2 on Kaggle (§7); evaluate base vs fine-tuned (results/chat_eval/README.md, demo set,
  bodycam clips arrest-1 and fire-rescue)
- [ ] Webcam: record a test clip (`scripts/skeleton_check.py --record 20`: wave, thumbs up, sit still, a staged
  push with a second person). Wire the skeleton model in as a webcam second signal only if it separates these.
- [x] RF-DETR integrated behind `detector.family: rfdetr` (`pip install -e .[rfdetr]`). CAVIAR rules: YOLO11n 0/6 →
  RF-DETR-S 3/6 triggered, 2/6 confirmed, 0 confirmed false alarms (same YOLO-tuned thresholds; docs/DETECTION.md)
- [ ] **Owner decides** whether `config/settings.yaml` switches to `family: rfdetr`; then retune tracker/confidence
  thresholds on non-test data, and export to ONNX/TensorRT for latency
- [ ] Abandoned object on CAVIAR: RF-DETR sees the 16×22 px bag (YOLO never does) but at 0.10–0.25 confidence, which
  `fuse_score` drops; the bag is also present only ~16 s against `unattended_s: 30`
- [ ] Riot data (XD-Violence riot + peaceful crowds) for a later fine-tune; the owner downloads and uploads it
- [ ] Hazard detector training (fire/smoke/weapons; D-Fire class order still to verify)

### Product (after the faculty items)
- [ ] Bodycam: speech transcript (Whisper small ~0.5 GB, ask before downloading; Indic ASR for Kannada after a
  licence check), blurring bystander faces and number plates
- [ ] Per-frame inference table and timeline lanes (docs/ROADMAP.md)
- [ ] Staged bodycam footage with consent forms

## 7. Running elsewhere and waiting on

- **Fine-tune, run 1** died at stage 5. `ollama pull` got a 503 after 3.5 h of teacher work, and only 96 of
  1,654 teacher summaries were kept: 12B's self-check rejected 84 %, mostly over "number of people" and
  clothing.
- **Fine-tune, run 2** is pushed. Changes:
  - pulls retry and happen early
  - the self-check fails only on invented actions or things
  - UCF-Crime keeps arrest and fire, and drops theft, vandalism and accident

  Re-import `kaggle/finetune_gemma.ipynb`. Inputs are listed in kaggle/README.md.
- **Tracker migration: done** (merged 2026-10-16). Tracking runs on Ultralytics' `BYTETracker` (OC-SORT via
  `tracker.algorithm`). CAVIAR scores match the old supervision tracker (docs/TRACKING.md).
- **Faculty dataset:** format and categories still unknown.
- **Open question for the owner:** test events #149–151 (demo-webcam, 03:38–03:39 on 2026-10-16) were created
  by Claude in the real `data/aequitas.db` while testing. Delete them?

## 8. Rules (from CLAUDE.md and the owner)

- Ask before every push and before downloads over ~1 GB.
- Report failures and worse numbers. Never present thresholds tuned on a test set as accuracy.
- Guardrails (R6) are design, not options.
- Don't change Windows system or privacy settings; tell the owner what to change.
- Test servers use their own `data_dir`, never the real `data/aequitas.db`.
- Ollama caches identical prompts and images: benchmarks need a different image per call.
- On Windows, scripts should call `127.0.0.1`, not `localhost`: urllib's IPv6 fallback adds ~2 s per request.
