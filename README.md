# Minerva

Open-source live video event summarization, classification and conversational querying.
All inference is local: YOLO11n + ByteTrack for detection and tracking, rule-based triggers,
a quantized VLM in Ollama for verification and summaries, SQLite for storage, Gradio for chat.

Design and event definitions: [docs/DESIGN.md](docs/DESIGN.md).

**Two analysis modes** (`analysis.mode` in `config/settings.yaml`):
- `triggered` (default): YOLO tracking and rules decide when the VLM looks, so it runs a few times per event.
- `periodic`: no YOLO; the VLM reviews every window of sampled frames. Simpler, but one VLM call per window per camera.

**Models are independently replaceable.** The VLM (`vlm:`) and chat LLM (`chat:`) can each point at Ollama
or at any local OpenAI-compatible server (llama.cpp, vLLM, LM Studio). Gemma 4 (`gemma4:e2b`, `gemma4:e4b`)
and Qwen2.5-VL-3B are the tested candidates. Non-local hosts are refused unless `allow_remote: true`.

**Grounded answers.** Every listed event has an `[#id]` reference. The optional LLM summary is kept only if it
cites retrieved events and nothing else.

## Setup

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -e .[dev]
ollama pull qwen2.5vl:3b
```

On an NVIDIA machine, install the CUDA build of PyTorch first (see pytorch.org), then the line above.
`device: auto` in `config/settings.yaml` picks the GPU automatically.

## Use

```bash
# M0: measure detector and VLM speed and memory on this machine
.venv\Scripts\python -m minerva spike

# Register cameras, then process streams (put videos at the paths in config/cameras.yaml)
.venv\Scripts\python -m minerva init-db
.venv\Scripts\python -m minerva run
.venv\Scripts\python -m minerva run --no-vlm --camera cam1   # trigger-only, one camera

# Ask questions
.venv\Scripts\python -m minerva ask "Show all tailgating events from camera 2 today"
.venv\Scripts\python -m minerva ui        # http://127.0.0.1:7860

# Latency summary of recorded events
.venv\Scripts\python -m minerva stats
```

### Simulating a live RTSP camera

Run [MediaMTX](https://github.com/bluenviron/mediamtx), then loop a file into it with FFmpeg:

```bash
ffmpeg -re -stream_loop -1 -i data/videos/cam1.mp4 -c copy -f rtsp rtsp://127.0.0.1:8554/cam1
```

and set `source: rtsp://127.0.0.1:8554/cam1` in `config/cameras.yaml`.

## Tests

```bash
.venv\Scripts\python -m pytest
```

The trigger rules are tested on synthetic tracks, so tests need no models or video.

## Layout

```
minerva/ingest      frame sources (RTSP/webcam/file) and the per-camera ring buffer
minerva/vision      YOLO detector, ByteTrack, pose on crops, motion energy
minerva/triggers    one rule per category + the engine (cooldown, episodes)
minerva/understand  keyframe grid, VLM prompt/client, fusion
minerva/store       SQLite schema, EventStore, QueryFilter -> SQL
minerva/chat        question -> QueryFilter (LLM, keyword fallback), templated answers
minerva/ui          Gradio app
minerva/eval        M0 spike, latency report
```
