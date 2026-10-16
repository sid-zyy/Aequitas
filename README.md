# Aequitas

Open-source, local-only video understanding for public safety: video in → **(1) summary → (2) label → (3) a
grounded chatbot**, with every stage timed. It runs on one laptop GPU. No cloud API is called, and non-local
model hosts are refused unless `allow_remote: true`.

- **How it works:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Project state and plan:** [HANDOVER.md](HANDOVER.md)
- **Faculty requirements:** [docs/HACKATHON_PLAN.md](docs/HACKATHON_PLAN.md)

**In short:**
- **Ingest:** files, RTSP, a webcam (OpenCV or the browser), uploads. Sampled at 4 fps.
- **Cheap checks first:** blank and static windows are skipped, and a SigLIP scene gate skips clearly normal
  ones.
- **Stage 1:** Gemma 4 E2B writes a short factual summary of a 2×2 key-frame grid.
- **Stage 2:** a text-only call labels the summary, constrained to the category list (14 police categories +
  `none`, configurable).
- **Second signal:** a violence clip model, and agreement across windows, decide what is *confirmed*. A human
  still confirms every incident.
- **Stage 3:** the chatbot turns questions into filters and SQL (the LLM never writes SQL) and answers with
  `[#id]` references to stored events, streamed.
- **Serving:** Ollama on the laptop today. Any OpenAI-compatible server (vLLM, llama.cpp) by config.
  `scripts/vllm/serve.sh` adds FP8 / quantised checkpoints and MTP / n-gram speculation.

## Setup (fresh clone)

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -e .[dev,detector]   # detector: AI-likelihood check + scene gate
ollama pull gemma4:e2b                      # the vision model the app uses (4.6 GB)
python scripts/get_footage.py               # demo street + incident clips into data/footage
```

On an NVIDIA machine, install the CUDA build of PyTorch first (pytorch.org), then the line above.
`device: auto` in `config/settings.yaml` picks the GPU.

Recommended: the trained **violence clip model** (127 MB, not in git). Unzip `violence_clip.zip` (from
`kaggle/violence_clip.ipynb`, or ask the team) into `models/violence_clip/`. Without it the app still runs, but it
misses fights that Gemma describes vaguely (results/violence_clip/README.md).

**vLLM** runs in WSL2 Ubuntu: `wsl --install -d Ubuntu`, then `bash scripts/vllm/install_wsl.sh` inside it, then
`scripts/vllm/serve.sh`. Point `vlm:` in `config/settings.yaml` at it (`backend: openai`).

## Web app

```bash
.venv\Scripts\python -m aequitas web      # http://127.0.0.1:8000
```

**Sign in with `admin` / `admin`.** On first start with an empty database the app creates this demo account and
writes it to the audit log. It's for a laptop demo only: the server listens on 127.0.0.1. Make real accounts with
`.venv\Scripts\python -m aequitas create-user --username <name> --role admin` and set `auth.demo_admin: false`.

- **Live CCTV:** pick a Bengaluru locality. Each camera tile shows the latest summary and label, the summary and
  label time, and how far behind live it is. The laptop webcam can be one of the cameras.
- **Active requests:** open citizen reports, longest waiting first, with the video, what the AI found and the
  AI-generated likelihood.
- **Bodycam review:** upload an officer's recording. You get a 4 s timeline of summaries, key moments to confirm
  or dismiss (audited), a whole-recording overview, a printable report marked as an AI draft, and the total and
  per-stage time.
- **All reports:** the citizen-report archive.
- **Citizen portal** (`/report`): public upload with a tracking code.
- **Analyst chat:** streamed, grounded answers.

The React app lives in `frontend/` (Vite + TypeScript); its build in `aequitas/web/frontend_dist/` is committed.
After UI changes run `cd frontend && npm run build`. `npm run dev` gives hot reload with the API proxied. The API
is documented at `/api/docs`.

## Command line

```bash
.venv\Scripts\python -m aequitas spike                    # detector / VLM latency and memory on this machine
.venv\Scripts\python -m aequitas init-db                  # register cameras (config/cameras.yaml)
.venv\Scripts\python -m aequitas run                      # process camera streams
.venv\Scripts\python -m aequitas ask "Show all fighting events from camera 2 today"
.venv\Scripts\python -m aequitas stats                    # per-stage latency of recorded events
.venv\Scripts\python -m aequitas bench --host http://127.0.0.1:8001 --model gemma4 --concurrency 1 4 16   # vLLM latency
.venv\Scripts\python -m aequitas eval-classify            # stage 2 on labelled summaries
.venv\Scripts\python -m aequitas eval-chat                # chatbot filter parsing
.venv\Scripts\python -m aequitas eval-dataset --data <folder> [--manifest <csv/json>] [--label-map <yaml>]   # faculty dataset
.venv\Scripts\python -m aequitas --settings config/eval_demo_final.yaml --cameras config/demo_cameras.yaml run
.venv\Scripts\python -m aequitas --settings config/eval_demo_final.yaml eval-report --expected config/demo_expected.yaml
```

Check `--help` on each command for its options. To simulate a live RTSP camera, run
[MediaMTX](https://github.com/bluenviron/mediamtx) and loop a file into it:
`ffmpeg -re -stream_loop -1 -i data/videos/cam1.mp4 -c copy -f rtsp rtsp://127.0.0.1:8554/cam1`.

## Tests

```bash
.venv\Scripts\python -m pytest
```

Tests use synthetic data and need no models or video.

## Layout

```
aequitas/ingest      frame sources (file/RTSP/webcam/browser push), ring buffer
aequitas/vision      YOLO + ByteTrack, pose, motion, violence clip model, hazards, skeleton (prototype)
aequitas/triggers    one rule per category + the engine (triggered mode)
aequitas/understand  prompts, model clients, stages 1-2, key-frame grid, SigLIP gate, fusion, window aggregation
aequitas/store       SQLite schema, EventStore, QueryFilter -> SQL
aequitas/chat        question -> filter (LLM + keywords), grounded streamed answers
aequitas/web         FastAPI server, upload jobs, live localities, bodycam review
aequitas/eval        spike, bench, stage-2 / chat / dataset evaluation
frontend/            React app
scripts/             vLLM, quantization, benchmarks and evaluation scripts
kaggle/              training and study scripts (.py; notebooks built from them for Kaggle only)
docs/                architecture, plans, evaluations
```
