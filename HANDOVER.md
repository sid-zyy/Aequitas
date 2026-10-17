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
| **R5 ⚠️** | **Hackathon faculty: TOP PRIORITY** | **(1) summarization, (2) classification of the summarized info, (3) chatbot.** **Serving (vLLM optional; Ollama in use).** **Speculative decoding + MTP.** **4-bit and 8-bit quantization.** **React + FastAPI, no Jupyter.** **The faculty dataset.** Judged on **approach**, **end-to-end demo**, **latency (total execution time)**. |
| R6 | Owner + design | Guardrails: no face recognition or identification; no profiling by religion, caste, party or attire; peaceful gatherings are not incidents; a human confirms incidents (docs/POLICE_EDITION.md §8). |

### Decisions made
- **The model is Gemma 4 E2B, 4-bit, with MTP** (`gemma4:e2b`, back from Qwen on 2026-10-17). Qwen3-VL 2B
  (`qwen3-vl:2b-instruct`) was the demo model for one day; every deck number was measured with both
  (results/qwen/README.md), and the deck shows the head-to-head ("Why Gemma" slide).
  - **Why Gemma:** faster end to end (21 demo clips 1 min 48 s vs 2 min 24 s; 0.91 vs 1.01 s per window),
    speculative decoding works (MTP drafter; Qwen has none), Hindi/Kannada translation 6/6 vs 2/6, fewer false
    alarms. The violence model covers Gemma's weak recall.
  - **Cost:** Qwen confirms 1–2 more of the 10 demo incidents (8–9 vs 6–8), and alone it finds far more on
    SCVD (70 % vs 25 %).
  - **To switch to Qwen:** `vlm.model: qwen3-vl:2b-instruct`, `analysis.label_context: false`,
    `vlm.repeat_penalty: 1.1`, `vlm.draft_num_predict: null`.
  - **Quantization evidence:** Qwen on the laptop (results/qwen/README.md) and Gemma E2B at BF16 / Q8_0 / Q4_K_M / QAT
    on a Kaggle T4 (results/quant_gemma/README.md). E2B Q8_0 (~3.5 GB on the GPU) would also fit; Q4_K_M is kept.
  - **Qwen3-VL 4B/8B in results/model_compare** returned 60/60 empty summaries: Ollama's default tags are the
    thinking builds. That run is not a verdict on those models.
  - **The fine-tune (12B → E2B)** applies to Gemma only.
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
| Per-video review + chat: citizen-report videos get the bodycam timeline (summary, key moments with keyframes, confirm/dismiss); a chat answers only from that video's analysis, citizen's words marked unverified, cited times become seek buttons | Done | `chat/video_qa.py`, `components/VideoReview.tsx` |
| Police auth (scrypt, sessions, lockout, audit) | Done | `auth.py` |
| **React frontend** (Live CCTV, Active requests, Bodycam, Reports, Citizen portal, Analyst chat) served by FastAPI | Done | `frontend/` → `aequitas/web/frontend_dist/` |
| vLLM serve profiles (FP8 / quantised path × none / MTP / n-gram), WSL install script | Written; **run on Kaggle only** | `scripts/vllm/` |
| Quantisation (FP8 / W8A8 / W4A16, llm-compressor) | Written; **no successful run yet** | `scripts/quantize/quantize.py` |
| `aequitas bench` (TTFT, TPOT, tokens/s, acceptance rate) | Done | `eval/bench.py` |
| **Faculty dataset adapter** (`eval-dataset`: folder or manifest → all stages → accuracy, F1, summary similarity, time) | Done; no faculty dataset is coming (owner, 2026-10-16), only the 6 sample clips | `eval/dataset.py` |
| Gemma fine-tune (12B → E2B distillation, QLoRA) | Run 2 done: SCVD incident recall 0.20 → 0.60 (in-domain); not yet served or tested on Bengaluru clips | `kaggle/finetune_gemma.py`, results/finetune/README.md |
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
| Same-size VLMs on SCVD (Kaggle T4, VLM alone) | balanced acc.: qwen3-vl:2b-instruct 0.72 (recall 0.70, 25 % false alarms, 2.5 s), E2B 0.57 (0.25, 10 %, 1.8 s); gemma3:4b, llava-phi3, qwen2.5vl:3b flag almost everything; granite3.2-vision no valid output. E2B kept | results/model_compare_small/README.md |
| Time-saved ladder, 80 s video (Kaggle T4, vLLM, Qwen FP16) | 33.1 s → 5.4 s (6.1×) | results/ablation/README.md |
| Trackers on CAVIAR | tracker choice ≤ 0.02 IDF1; YOLO11n → 11m@960 3×; DeepSORT 4–5× ID switches, 30–46 ms/frame | docs/TRACKING.md |
| YOLO11n / pose crop | 12.7 ms / 17.6 ms per frame | docs/DESIGN.md §4.4 |
| Detectors on CAVIAR (AP50, person) | YOLO11n 0.102 · YOLO26l@960 0.344 · RF-DETR-S 0.425 (~27 ms) · D-FINE-X 0.522 (75 ms) | docs/DETECTION.md |
| Detectors on real violence CCTV (118 hand-labelled people, recall/precision @0.4) | YOLO11n 0.47/0.86 · YOLO11m 0.48/0.90 · RF-DETR-S 0.70/0.87 · RF-DETR-M 0.72/0.93 | docs/DETECTION.md |

## 5. Machine setup

**GPU laptop:** RTX 4050 Laptop, 6 GB, compute 8.9, so FP8 works. Gemma 4 E2B in BF16 is ~10 GB and does
not fit unquantised.

1. Clone, then `python -m venv .venv`. Install the CUDA build of PyTorch, then
   `.venv\Scripts\python -m pip install -e .[dev,detector,rfdetr]`.
2. `ollama pull gemma4:e2b` (~4.6 GB; the demo model). `qwen3-vl:2b-instruct` (~1.9 GB) is the alternative. Warm the model before a demo: the first call loads it.
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
- [x] **Serving decided: Ollama** (owner, 2026-10-16: vLLM was only an option). vLLM 0.31 was installed in WSL but
  can't fit Gemma 4 E2B in 6 GB; the vLLM scripts stay for a bigger GPU
- [x] **MTP measured in Ollama** (docs/SPECULATIVE.md, `scripts/bench_spec.py`): gemma4:e2b's drafter at 3 tokens,
  window 1.28 → 1.16 s, chat 0.58 → 0.45 s, labels unchanged. n-gram needs vLLM: not measured
- [x] **Quantisation study, measured on the laptop** (Qwen3-VL 2B, results/qwen/README.md): 4-bit 1.9 GB and
  1.60 s/window; 8-bit 2.5 GB and 1.74 s/window, and no more accurate end to end; 16-bit doesn't fit (4.7 GB,
  23/29 layers on the GPU, 9.6 s/window). 4-bit is used.
- [x] **Gemma E2B quantisation, Kaggle T4** (results/quant_gemma/README.md): BF16, Q8_0 and Q4_K_M all flag 20/39
  incident windows with 0/33 false alarms (QAT 18/39); BF16 4.11 s, Q8_0 2.06 s, QAT 1.89 s, Q4_K_M 1.84 s per window.
  Only the Q4_K_M run had the MTP drafter on; QAT vs Q8_0 (both without) is the fair 4-bit vs 8-bit gap: 8 % faster.
- [ ] **Ablation re-run at the end** (Kaggle, re-import `kaggle/ablation.ipynb`; the torchvision fix is in
  the repo) → tables and charts for the presentation
- [x] Faculty dataset: none is coming (owner, 2026-10-16). The 6 sample clips are the faculty test, spread one per
  locality on the Live page (config/bangalore.yaml)
- [ ] Demo script: upload → live stages → labels → chat, **total execution time on screen**; final report and
  latency slide

### Accuracy (in progress)
- [x] Fine-tune run 2 on Kaggle: results/finetune/README.md (in-domain gain; Ollama import failed, timed in
  transformers; notebook now falls back to `--quantize int4`)
- [ ] Serve the fine-tuned adapter (vLLM `--enable-lora`, results/finetune/lora/) and evaluate base vs fine-tuned
  on the demo set (`config/eval_demo_final.yaml`), bodycam clips arrest-1 and fire-rescue, and chat_eval
- [ ] Webcam: record a test clip (`scripts/skeleton_check.py --record 20`: wave, thumbs up, sit still, a staged
  push with a second person). Wire the skeleton model in as a webcam second signal only if it separates these.
- [x] RF-DETR integrated behind `detector.family: rfdetr` (`pip install -e .[rfdetr]`). CAVIAR rules: YOLO11n 0/6 →
  RF-DETR-S 3/6 triggered, 2/6 confirmed, 0 confirmed false alarms (same YOLO-tuned thresholds; docs/DETECTION.md)
- [x] `config/settings.yaml` uses `family: rfdetr` since 2026-10-16 (owner's decision; falls back to YOLO if `rfdetr` isn't
  installed). Code default stays `yolo`
- [ ] Retune tracker/confidence
  thresholds on non-test data, and export to ONNX/TensorRT for latency
- [ ] Abandoned object on CAVIAR: RF-DETR sees the 16×22 px bag (YOLO never does) but at 0.10–0.25 confidence, which
  `fuse_score` drops; the bag is also present only ~16 s against `unattended_s: 30`
- [ ] Riot data (XD-Violence riot + peaceful crowds) for a later fine-tune; the owner downloads and uploads it
- [ ] Hazard detector training (fire/smoke/weapons; D-Fire class order still to verify)

### Product (after the faculty items)
- [x] **Production features** (2026-10-17, docs/ARCHITECTURE.md §12.1): tamper-evident audit log (HMAC hash chain,
  admin Audit log page), per-incident evidence package (ZIP with original video, manifest, SHA256SUMS, draft
  certificate), System health page. Tests: tests/test_production.py
- [ ] Bodycam: speech transcript (Whisper small ~0.5 GB, ask before downloading; Indic ASR for Kannada after a
  licence check), blurring bystander faces and number plates
- [ ] Per-frame inference table and timeline lanes (docs/ROADMAP.md)
- [ ] Staged bodycam footage with consent forms

## 7. Running elsewhere and waiting on

- **Fine-tune, run 1** died at stage 5. `ollama pull` got a 503 after 3.5 h of teacher work, and only 96 of
  1,654 teacher summaries were kept: 12B's self-check rejected 84 %, mostly over "number of people" and
  clothing.
- **Fine-tune, run 2** finished (results/finetune/README.md): 290 teacher summaries kept; on the 60 SCVD test
  clips incident recall 0.20 → 0.60, precision 0.62 → 0.86. In-domain only, weapon 0/20 labelled weapon,
  fire worse, and the Ollama import failed on `--quantize q4_K_M`, so its time (7.7 s, transformers) is not
  comparable. Run 2's changes were:
  - pulls retry and happen early
  - the self-check fails only on invented actions or things
  - UCF-Crime keeps arrest and fire, and drops theft, vandalism and accident

  Re-import `kaggle/finetune_gemma.ipynb`. Inputs are listed in kaggle/README.md.
- **Tracker migration: done** (merged 2026-10-16). Tracking runs on Ultralytics' `BYTETracker` (OC-SORT via
  `tracker.algorithm`). CAVIAR scores match the old supervision tracker (docs/TRACKING.md).
- **Faculty dataset:** none is coming (owner, 2026-10-16); the 6 sample clips are all there is.
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
