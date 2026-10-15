# Aequitas — Refined Design (v0.2)

Live video event summarization, classification and conversational querying.
Refines `Video_Event_System_Problem_Statement_and_Plan.pdf`. This doc is the contract
the code is built against; change it here first, then in code.

**Hardware:** the target GPU is not settled. The development laptop has no NVIDIA GPU
(Intel Iris Xe, 32 GB RAM), so everything also runs on CPU. `device: auto` picks CUDA when present.
Only §4.2 (model sizes) depends on the card.

**Implementation note:** trigger scores are computed from the strength of the evidence when the
rule fires, not from duration (a duration score would always equal the sustain time). The exact
formulas are in `aequitas/triggers/*.py`; the score formulas in §2 are superseded.

## 0. Requirements added at the team meeting (2026-10-15)

The team's workflow slide extends the problem statement. It is binding alongside the PDF.

| Slide requirement | How Aequitas meets it |
|---|---|
| Capture: RTSP, webcam or file; single or multiple cameras | `FrameSource` per camera, one thread each (§4.1). |
| Prepare: sample frames; **YOLO tracking is optional**; camera ID and timestamp | Two analysis modes (`analysis.mode`): **`triggered`** (YOLO + rules gate the VLM, the default) and **`periodic`** (no YOLO: the VLM reviews every sampled window). See §3.5. |
| Understand: **Gemma 4 is the candidate VLM**; summarize and categorise | Gemma 4 (`gemma4:e2b` / `gemma4:e4b`, quantized in Ollama) is a first-class VLM option and part of the M0 comparison with Qwen2.5-VL-3B. Quantized models are allowed. |
| Structure: summary, category, timestamp, camera ID, key-frame reference | `events` table (§5). |
| Persist: event database plus key-frame storage | SQLite + JPEGs under `data/keyframes/`. |
| Chatbot: **any local open-source LLM**; models are independently replaceable | VLM and chat LLM are configured separately. Backends: Ollama or any **local** OpenAI-compatible server (llama.cpp, vLLM, LM Studio). Non-local hosts are refused unless explicitly allowed, so no paid cloud API is used by accident. |
| Retrieve: filter by time, camera, category | `QueryFilter` → parameterised SQL (§6). |
| **Grounded response: summary + event references** | Each listed event carries an `[#id]` reference. An optional LLM summary may cite only IDs from the retrieved rows; if it cites anything else it is discarded and the template answer is used. |

---

## 1. Pipeline at a glance

```
Source (RTSP / webcam / file)
  └─ Ingest thread ── 4 fps, latest-frame queue, 12 s ring buffer per camera
       └─ Detect + track (YOLO11n + ByteTrack) ── every sampled frame
            └─ Trigger engine (rules per category, per-camera zones)
                 └─ Candidate event ──► VLM queue (bounded, 1 worker)
                                         └─ Qwen2.5-VL-3B (4-bit, Ollama) on a 2×2 keyframe grid
                                              └─ Fusion → status + confidence
                                                   └─ SQLite + keyframe JPEG
Chat UI (Gradio) ─► same VLM as text LLM ─► QueryFilter JSON ─► SQL ─► results + keyframes
```

**Change from the original plan:** sample at **4 fps**, not 1–2. ByteTrack loses and swaps IDs
at 1 fps when people move fast, and fighting needs motion between frames. YOLO11n costs
about 5–10 ms per frame on GPU, so 4 fps per camera is cheap. The VLM still runs only on triggers.

---

## 2. Event definitions

Every category has a **trigger rule** (cheap, runs on every frame), a **trigger score**
`s ∈ [0,1]` (how far past the threshold the evidence is), and an explicit **out of scope** list.
All thresholds live in `config/cameras.yaml` and get tuned on the validation split (§6).

Common rules:
- A rule fires only after its condition holds for the **sustain time** (stops one-frame flicker).
- **Cooldown:** the same (camera, category, track set) cannot fire again for 30 s. Instead it
  extends the open event's `end_ts`.
- Only `person` tracks with confidence ≥ 0.4 and height ≥ 40 px count (too small means too noisy).

### 2.1 Person down
| | |
|---|---|
| **Signal** | A person track goes from **upright** to **horizontal** and stays horizontal. |
| **Upright** | Box h/w ≥ 1.4 at some point in the last 3 s. |
| **Horizontal** | Torso angle > 60° from vertical, measured with **YOLO11n-pose run only on that track's crop** (shoulder to hip midpoints). Falls back to box w/h ≥ 1.2 if keypoints are low-confidence. |
| **Sustain** | Horizontal for ≥ 3 s, and the box centroid moves < 0.3 × box width during that time. |
| **Score** | `s = min(1, horizontal_duration / 10 s) × keypoint_conf` |
| **Out of scope** | Sitting, crouching, bending to pick up (torso stays < 60°); people already lying when the track starts (no upright→down transition). |

Why pose: aspect ratio alone breaks on high, angled CCTV views where a standing person can look wide.
Running pose only on candidate crops keeps cost near zero.

### 2.2 Fighting
| | |
|---|---|
| **Signal** | Two or more people close together with high, erratic motion. |
| **Proximity** | Centre distance < 0.8 × mean box height of the pair, or box IoU > 0. |
| **Motion** | Mean absolute frame difference inside the union box, normalised by the camera's rolling 60 s baseline, ≥ 3.0. Also requires box-centre jitter (direction changes) on ≥ 2 tracks. |
| **Sustain** | ≥ 1.5 s (6 frames). |
| **Score** | `s = min(1, motion_ratio / 6) × min(1, duration / 4 s)` |
| **Out of scope** | Hugging, dancing, sports. The rule will fire on these; the VLM check is the filter. This is the category where the VLM matters most. |

### 2.3 Tailgating (video-only definition)
The original plan says "behind one authorization". There is **no badge or door signal** in a
video-only system, so the operational definition is:

| | |
|---|---|
| **Signal** | Two or more distinct person tracks cross a camera's configured **entry line** in the **entering direction** within Δt seconds of each other. |
| **Δt** | 2.0 s default, per camera. |
| **Sustain** | n/a (crossing is an instant event). |
| **Score** | `s = 1 − (gap / Δt)`: closer crossings score higher. |
| **Hook** | `AccessEvent(camera_id, ts)` interface. If badge data exists later, tailgating becomes "crossings > authorizations within the window". Stubbed for now. |
| **Out of scope** | Cameras with no entry line configured; exits (wrong direction); a single person crossing twice. |

Report must state this limitation: groups who legitimately enter together will be flagged.

### 2.4 Crowd incident
Narrowed to two concrete subtypes:

| Subtype | Rule | Sustain | Score |
|---|---|---|---|
| `overcrowding` | Person count inside the camera's `crowd_zone` ≥ `max_people` | 10 s | `min(1, (count − max) / max + 0.5)` |
| `dispersal` (stretch) | ≥ 5 tracks, and mean track speed jumps to ≥ 3× the 60 s median within 2 s, with velocity vectors pointing away from a shared centre | 2 s | `min(1, speed_ratio / 6)` |

**Out of scope:** queues, normal rush-hour density below `max_people`.

### 2.5 Suspicious activity
Only these two subtypes. Nothing else lands here, and the VLM cannot invent new ones.

| Subtype | Rule | Score |
|---|---|---|
| `loitering` | One track stays inside a `watch_zone` for ≥ 60 s with net displacement < 2 × box height over the last 30 s. | `min(1, dwell / 180 s)` |
| `abandoned_object` | A `backpack`, `handbag` or `suitcase` box stays still (IoU ≥ 0.7 with its first position) for ≥ 30 s, **and** no person box centre is within 1.5 × person height of it for those 30 s, **and** the person nearest it when it first appeared has left the frame or moved away. | `min(1, unattended_time / 90 s)` |

ByteTrack `track_buffer` is raised to 60 frames (15 s at 4 fps) so short occlusions don't reset dwell timers.

### 2.6 Category list (fixed enum)
```
person_down | fighting | tailgating | crowd_incident | suspicious_activity
```
The VLM may additionally answer `none`. Subtypes are stored in a separate column.

---

## 3. Fusion: from trigger + VLM to a stored event

### 3.1 What the VLM sees
- The ring buffer gives frames from the **onset** of the condition to the **trigger time**.
- Pick 4 frames: onset, ⅓, ⅔, trigger. Draw boxes and track IDs of the tracks involved on each
  frame and tile them into **one 2×2 grid, 896×896**. One image keeps token count and latency down.
- For tailgating, the entry line is drawn too.

### 3.2 Prompt contract
The prompt lists the 5 categories with their **definitions from §2 (one line each)**, plus `none`.
It asks: *which category best describes the highlighted people, if any?* It does **not** tell the VLM
which rule fired, to avoid anchoring it.

Output is forced to this JSON schema (Ollama `format=`):
```json
{
  "category": "person_down | fighting | tailgating | crowd_incident | suspicious_activity | none",
  "summary": "1–2 sentences, describe only what is visible",
  "visible_evidence": "short phrase"
}
```
The VLM's self-reported confidence is not requested; it is poorly calibrated.

### 3.3 Fusion rule
| VLM answer | `status` | `confidence` |
|---|---|---|
| Same category as trigger | `confirmed` | `0.6 + 0.4·s` |
| Different category | `disputed` | `0.4·s` (category stays = trigger's; VLM's choice saved in `vlm_category`) |
| `none` | `rejected` | `0.2·s` |

All three are stored. The chat UI shows only `confirmed` by default. Keeping rejected rows makes
false-positive analysis and the trigger-vs-VLM ablation possible (§6).
The coefficients are starting values; they get re-fit on the validation split.

### 3.4 Back-pressure
VLM queue holds at most 8 candidates per camera. When full, drop the candidate with the lowest
trigger score and log it in `dropped_candidates`. Dropped counts go in the latency report.

### 3.5 Periodic mode (no YOLO)
- Each camera's stream is cut into **windows** (`analysis.window_s`, default 4 s). Each window becomes
  a 2×2 grid without boxes, and the VLM gets the same category list plus `none`.
- A new window is only queued when the camera has no window still waiting. If the VLM is slower than
  real time, windows simply get longer instead of piling up. The effective interval is logged.
- Consecutive windows with the same category merge into one event, with `end_ts` extended.
- There is no second signal, so agreement **across windows** replaces agreement with a rule:

| Agreeing consecutive windows | `status` | `confidence` |
|---|---|---|
| 1 | `unverified` | 0.4 |
| ≥ `confirm_votes` (default 2) | `confirmed` | 0.7, +0.05 per further window, capped at 0.9 |

- `unverified` is a fifth status, used only in periodic mode. The chat hides it by default, like
  `disputed` and `rejected`.
- Trade-off: this costs one VLM call per window per camera, which is why `triggered` stays the default.
  On this laptop's CPU (~4.5 s per call) periodic mode keeps up with about one camera.

---

## 4. Stack

### 4.1 Libraries
| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.11 | |
| Video I/O | `opencv-python` (FFmpeg backend) | One `FrameSource` class covers RTSP, webcam and file. |
| Simulated live stream | **MediaMTX** + `ffmpeg -re -stream_loop -1` | Turns any test video into a real RTSP feed for demos. |
| Detection | Ultralytics **YOLO11n** (COCO) | Classes used: person, backpack, handbag, suitcase. AGPL-3.0, fine for an open-source project. |
| Pose (crops only) | Ultralytics **YOLO11n-pose** | Person-down only. |
| Tracking | `supervision` ByteTrack, one tracker per camera and class group (people, bags) | Lets one detector serve all cameras. Pinned `<0.31`, where `sv.ByteTrack` is removed. |
| Zones and lines | Own geometry in `triggers/base.py` | Small, dependency-free, and unit-tested. |
| VLM runtime | **Ollama** | Open-source local server. Supports JSON-schema structured output, runs on Windows, keeps the VLM in a separate process. |
| VLM model | **`qwen2.5vl:3b`** (Q4) | Named in the plan, good at structured output. **Challenger:** `qwen3-vl:4b` or `qwen3-vl:2b`; whichever wins the M0 spike becomes the default. **Fallback:** SmolVLM2-2.2B. |
| Chat LLM | **The same Ollama model, text-only** | Satisfies "one heavy model on the GPU". |
| Validation | `pydantic` v2 | VLM output and chat filters are parsed into models; invalid output is retried once, then the event is stored as `disputed` with `summary = null`. |
| Storage | `sqlite3` (stdlib), WAL mode, FTS5 | No ORM. |
| Semantic search (stretch) | `sentence-transformers/all-MiniLM-L6-v2` on CPU + NumPy cosine | Prototype event counts are small; no vector DB needed. |
| UI | **Gradio** | Chat panel and keyframe gallery in one app. |
| Metrics | `pynvml` (VRAM), `time.perf_counter_ns` per stage | |
| Config | YAML + pydantic | |

Versions are pinned in `requirements.lock` at setup time; nothing above relies on unreleased features.

### 4.2 VRAM budget (6 GB, to be measured in M0)
| Item | Estimate |
|---|---|
| CUDA context, detector process | ~0.4 GB |
| YOLO11n + YOLO11n-pose, FP16 | ~0.3 GB |
| Ollama CUDA context | ~0.4 GB |
| qwen2.5vl:3b Q4 weights + vision encoder | ~3.2 GB |
| KV cache (4k context, 1 image) | ~0.5 GB |
| **Total** | **~4.8 GB**, leaving ~1 GB of headroom |

If M0 measures above 5.5 GB: lower Ollama `num_ctx` to 2048, then drop to `qwen3-vl:2b` or SmolVLM2.

### 4.3 M0 results on the dev laptop (CPU only, 2026-10-15)
| Stage | Measured |
|---|---|
| YOLO11n, 640 px, torch CPU | 45–82 ms per frame (p50 across runs), so 4 fps per camera is comfortable |
| YOLO11n-pose on one crop | 16–36 ms |
| qwen2.5vl:3b, 896 px 2×2 grid, **uncached** | ~42–46 s per call; 3/3 valid JSON |
| Chat question → filter (text only) | ~6 s |

**Gemma 4 E2B vs Qwen2.5-VL-3B** (same 896 px grid, unique image per call, CPU):

| Model | Download | Uncached call (p50) | Cold start | Valid JSON | Static scene answered `none` |
|---|---|---|---|---|---|
| qwen2.5vl:3b | 3.2 GB | ~44 s | ~40 s | 8/8 | 6 of 12 calls across runs (others "tailgating") |
| **gemma4:e2b** | 4.6 GB | **~16 s** | ~16 s | 8/8 | **8 of 8** |

Gemma 4 E2B is about 2.7× faster on this CPU and made no false calls on the static scene, so it is now the
default VLM. One still scene is not an accuracy benchmark: the M5 evaluation on labelled clips decides.

Lessons:
- **Ollama caches identical prompts and images.** A repeated image returns in ~4 s, which is misleading.
  The spike now uses a different image for every call.
- At ~45 s per call on CPU, triggered mode is the only practical mode here. Periodic windows stretch to
  ~45 s. A discrete GPU is expected to bring this down by an order of magnitude (to be measured on the target card).
- Qwen2.5-VL-3B labelled a static scene as "tailgating" in 2 of 3 calls. The trigger + VLM agreement
  rule turned that into `disputed`, which is the intended safety net.

---

## 5. Storage schema (frozen contract)

Times are **UTC epoch milliseconds** (INTEGER). The UI and chatbot convert to and from local time.
For file sources, `ts = file_start_time + frame_pts`. `file_start_time` comes from config or defaults to ingest start.

```sql
CREATE TABLE cameras (
  camera_id     TEXT PRIMARY KEY,          -- 'cam2'
  name          TEXT NOT NULL,             -- 'Lobby entrance'
  aliases       TEXT NOT NULL DEFAULT '[]',-- JSON: ["entrance","camera 2","front door"]
  source_uri    TEXT NOT NULL,
  zones         TEXT NOT NULL DEFAULT '{}' -- JSON: entry_line, crowd_zone, watch_zones
);

CREATE TABLE events (
  event_id        INTEGER PRIMARY KEY,
  event_uid       TEXT NOT NULL UNIQUE,            -- uuid4, used in file names
  camera_id       TEXT NOT NULL REFERENCES cameras(camera_id),
  start_ts        INTEGER NOT NULL,                -- "timestamp" in the problem statement
  end_ts          INTEGER NOT NULL,
  category        TEXT NOT NULL CHECK (category IN
                    ('person_down','fighting','tailgating','crowd_incident','suspicious_activity')),
  subtype         TEXT,                            -- overcrowding | dispersal | loitering | abandoned_object
  status          TEXT NOT NULL CHECK (status IN ('confirmed','disputed','rejected')),
  confidence      REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
  trigger_score   REAL NOT NULL,
  vlm_category    TEXT,                            -- raw VLM answer, incl. 'none'
  summary         TEXT,
  zone            TEXT,                            -- zone name the event happened in, if any
  track_ids       TEXT NOT NULL DEFAULT '[]',      -- JSON array
  keyframe_path   TEXT NOT NULL,                   -- data/keyframes/<cam>/<yyyy-mm-dd>/<uid>.jpg (the 2x2 grid)
  clip_path       TEXT,                            -- optional short MP4
  model_versions  TEXT NOT NULL,                   -- JSON: detector, vlm, rules version
  created_at      INTEGER NOT NULL
);
CREATE INDEX ix_events_cam_time ON events(camera_id, start_ts);
CREATE INDEX ix_events_cat_time ON events(category, start_ts);

CREATE VIRTUAL TABLE events_fts USING fts5(summary, content='events', content_rowid='event_id');

CREATE TABLE event_timings (
  event_id   INTEGER PRIMARY KEY REFERENCES events(event_id),
  detect_ms  REAL, trigger_ms REAL, queue_ms REAL, vlm_ms REAL, db_ms REAL,
  e2e_ms     REAL            -- condition onset -> row committed
);

CREATE TABLE dropped_candidates (
  camera_id TEXT, ts INTEGER, category TEXT, trigger_score REAL, reason TEXT
);
```

`zone` and camera `aliases` exist so questions like "anything near the entrance" resolve without
semantic search.

---

## 6. Conversational querying

The LLM never writes SQL. It fills this model, which code turns into a parameterised query:

```python
class QueryFilter(BaseModel):
    categories: list[Category] = []        # empty = all
    camera_ids: list[str] = []             # resolved against cameras.name/aliases
    start: datetime | None = None          # local time, converted to UTC epoch ms
    end: datetime | None = None
    statuses: list[Status] = ["confirmed"]
    min_confidence: float = 0.0
    text: str | None = None                # goes to FTS5 (later: embeddings)
    sort: Literal["newest", "oldest", "confidence"] = "newest"
    limit: int = Field(20, le=100)
    intent: Literal["list", "count"] = "list"
```

- The prompt includes **current local time, timezone**, the camera list with aliases, and the enum.
  "Today", "this morning" (06:00–12:00) and "last hour" are resolved by the LLM, then sanity-checked
  in code (start ≤ end ≤ now).
- Unknown camera names are reported back ("I don't know a camera called X") rather than dropped silently.
- Answers are built from a template plus the rows (count, then the list with time, camera, category,
  summary and thumbnail). The LLM does not paraphrase results, so it cannot hallucinate events.

---

## 7. Evaluation

### 7.1 Data
| Category | Candidate sources (licence and availability to confirm before use) |
|---|---|
| Fighting | RWF-2000; UCF-Crime "Fighting" |
| Person down | UR Fall Detection; Le2i Fall; plus self-recorded |
| Crowd | UMN unusual crowd activity (dispersal); self-recorded or staged overcrowding |
| Suspicious | PETS 2006 (abandoned luggage); self-recorded loitering |
| Tailgating | **Self-recorded only** (doorway, 20+ clips: single entry, pairs at 0.5–3 s gaps) |
| Negatives | UCF-Crime "Normal" videos, several hours total |

Split per category: **validation** (tune thresholds and fusion) and **test** (report only, touched once).

### 7.2 Metrics
- **Event-level precision and recall per category.** A prediction matches a ground-truth event if
  same camera and category, and the intervals overlap or are within ±3 s.
- **False positives per hour** on negative footage. This is the main operational number.
- **Ablation:** trigger-only vs trigger + VLM (shows what the VLM adds; uses the stored `rejected` rows).
- **Latency:** p50/p95 per stage from `event_timings`; end-to-end onset → DB row; chat question → answer.
- **Resources:** peak and steady VRAM (`pynvml`), sustainable fps per camera, max cameras before
  the VLM queue starts dropping.
- **Chat accuracy:** 40 hand-written questions with expected `QueryFilter`s; exact-match rate per field.

---

## 8. Project layout

```
aequitas/
  ingest/       frame_source.py, ring_buffer.py
  vision/       detector.py, tracker.py, pose.py
  triggers/     base.py, person_down.py, fighting.py, tailgating.py, crowd.py, suspicious.py
  understand/   keyframes.py, vlm_client.py, prompts.py, fusion.py
  store/        schema.sql, db.py
  chat/         query_filter.py, parser.py, answer.py
  ui/           app.py
  eval/         datasets.py, match.py, report.py, vram_probe.py
config/         cameras.yaml, thresholds.yaml
data/           keyframes/, clips/, aequitas.db
tests/
```

## 9. Milestones
| # | Milestone | Done when |
|---|---|---|
| M0 | **VRAM and latency spike** | YOLO11n tracking at 4 fps + qwen2.5vl:3b and qwen3-vl (2b/4b) answering the §3.2 schema on a 2×2 grid; peak VRAM and VLM latency recorded; default VLM chosen. |
| M1 | Ingest + detect/track | File, webcam and MediaMTX RTSP all produce tracked frames with camera ID and timestamps. |
| M2 | Storage + triggers | All 5 rules emit candidates into SQLite (no VLM yet), with unit tests on synthetic tracks. |
| M3 | VLM + fusion | Candidates get summaries and status; timings recorded. |
| M4 | Chat | Gradio app answers the example questions with keyframes. |
| M5 | Evaluation | Test-split report: per-category P/R, FP/hour, ablation, latency, VRAM. |

## 10. Open decisions (need input)
1. **Actual GPU and VRAM?** Only §4.2 depends on it.
2. **Local timezone** for "today" / "this morning" (default: the system timezone).
3. **Any badge/door data available?** If yes, tailgating switches to the authorization-count definition.
4. **How many simultaneous cameras** the demo needs (affects VLM queue sizing; default is 2).
