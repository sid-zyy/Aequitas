# Aequitas roadmap: three parts on one core

Aequitas is split into three **analysis profiles** that share one engine. The user picks a profile when
uploading a video or adding a camera. Each profile sets the event categories, the models, the default
analysis mode and the timeline lanes in the UI.

| | Part 1 · General | Part 2 · Bodycam review | Part 3 · CCTV public order |
|---|---|---|---|
| Priority | Supporting (demo and baseline) | **Major** | **Major** |
| Footage | Everyday video: phones, shops, homes, offices | Police body-worn and dash cameras (first-person, moving) | Fixed CCTV and drones over streets and gatherings |
| User | Anyone reviewing a clip | An officer or supervisor reviewing an incident | Control-room operators; post-incident review |
| Core question | "What notable things happen in this video?" | "Where are the key moments I need to review or report?" | "Where and when did violence, arson or crowd danger occur?" |
| Main output | Event list + summaries | **Key-moment index** with timestamps, a review report, redacted export | **Labelled timeline** per camera + incident list + alerts |
| Default mode | Triggered (YOLO + rules) | **VLM review + audio** (moving camera breaks fixed-camera rules) | Triggered + per-frame inference |

## Shared core (built once, used by all three)

| Piece | Status |
|---|---|
| Ingest (file / RTSP / webcam), sampling, ring buffer | Done |
| YOLO detection + ByteTrack, pose on crops, motion energy | Done |
| VLM verification and summaries (Gemma 4 E2B default, any local model) | Done |
| Event store, grounded chat, web app (upload, chat, event log) | Done |
| **Per-frame inference table + timeline UI** (an answer for any timestamp) | **Next**, needed by all three parts |
| **Profiles** (category set + models + prompt + lanes per profile) | **Next** |
| Clip-level evaluation (`aequitas eval-report`) | Done; CAVIAR baseline running |

---

## Part 1 · General (supporting)

Shows that the system works on ordinary videos. It gives everyone a quick demo and a baseline.

**Detects:** fire and smoke, person down/fall, fighting, abandoned bag, loitering, overcrowding,
plus a VLM summary of each event.

**Gap today:** no fire/smoke model. Plan:
1. Fine-tune YOLO11n/s on **D-Fire** (21.5k images, fire and smoke boxes, YOLO format) on Kaggle and export to ONNX/OpenVINO.
2. Run it as a second detector head in Tier 1. Add a `fire_smoke` rule (persistent over N frames, area growing) and a timeline lane.
3. Test footage: CAVIAR (done) plus free-licence stock clips of fires, kitchens, candles and sunsets.
   The last three are hard negatives.

**Done when:** a demo video with a fire, a fall and a left bag produces correct, summarised events in the app.

---

## Part 2 · Police bodycam review (major)

An officer uploads body-worn or dash-camera footage and gets a **key-moment index**: a timeline of
moments worth reviewing, each with a timestamp, a label, a short description and a thumbnail. They
can confirm, edit or dismiss each one, add notes, and export a report.

**Key-moment types:**

| Moment | Signals |
|---|---|
| Use of force / physical struggle | VLM window review + close-range person + violent motion |
| Weapon visible (firearm, knife, lathi/rod) | Detector (Part 3 weapon model) |
| Person on the ground / injured | Pose + VLM |
| Foot pursuit / officer running | Ego-motion from optical flow (strong camera shake + forward motion) |
| Raised voices, shouting, screaming | **Audio** event classifier (PANNs/YAMNet AudioSet classes: Shout, Screaming) |
| Gunshot, explosion, glass breaking, siren | Audio event classifier |
| Commands and key phrases ("stop", "hands up", Hindi and regional phrases) | **Speech-to-text** (faster-whisper, local) + phrase list |
| Vehicle stop / vehicle interaction | Detector (car, motorbike, auto) + VLM |
| Arrest / handcuffing | VLM window review |
| Camera covered / dark / no signal | Frame brightness and variance |

**Why a different mode:** the camera moves with the officer, so the fixed-camera rules (zones, entry
lines, "stationary bag") don't apply. This part relies on **VLM review of every window** plus **audio**.
On a GPU that is fast enough. On this CPU laptop it runs at about 16 s per 4 s window, so about
4× slower than real time.

**New pieces to build:**
1. Audio pipeline: extract the audio track (PyAV or ffmpeg), run faster-whisper (transcript with
   timestamps) and an audio-event classifier, and store both on the timeline.
2. Ego-motion estimate (global optical flow) for running and struggle detection.
3. Key-moment fusion: merge video, audio and speech signals into moments, with confidence and reasons.
4. Review UI: a video player with the key-moment list and timeline; confirm, edit or dismiss each moment;
   notes; export as a PDF/CSV report.
5. **Redacted export:** blur faces and number plates of bystanders in copies shared outside the department.

**Data:** there is no good public bodycam dataset with a clear licence. Two options:
- Staged footage: team members wear a chest-mounted phone in scripted scenarios (cheap and quick).
- Partner police footage under an MoU (real conditions, Indian languages).

**Done when:** a 10-minute staged bodycam video produces a key-moment index that a reviewer agrees with,
measured as precision and recall of moments within ±3 s.

---

## Part 3 · CCTV public order: riots, stone pelting (major)

Labels fixed-camera footage of streets and gatherings, with an inference for **every frame**, and raises
incidents. The full design is in [POLICE_EDITION.md](POLICE_EDITION.md).

**Incidents:** violent clash, stone pelting / projectile throwing, arson, smoke or gas, vandalism,
visible weapons, crowd-crush risk, panic dispersal, cordon breach, person down, and police use of force
(for accountability). A peaceful gathering is context only and never an incident.

**New pieces to build:**
1. Kaggle training:
   - a detector for weapons (stick/rod/lathi, knife, firearm), fire and smoke
   - a short-clip violence classifier (RWF-2000, XD-Violence Riot and Fighting, SCVD, plus Indian festival
     hard negatives)
   - a crowd-density model (JHU-CROWD++)
2. A throwing-motion detector for stone pelting (overhead-arm pose + fast small objects in optical flow).
3. Crowd-crush and dispersal rules using camera calibration (people per m²).
4. An Indian held-out test set and a false-alarm budget per camera-hour.

**Done when:** on held-out footage, per-category precision and recall are reported, along with the
false-alarm rate on long normal footage (rallies, festivals, markets).

---

## Next steps (in order)

> **Status 2026-10-16:** the hackathon faculty requirements come first (HANDOVER.md §6). Of the steps below:
> - Done: the CAVIAR baseline (step 1); the bodycam review UI and report export (step 5, without redaction);
>   the violence clip classifier (step 7)
> - Not trained yet: the hazard detector (steps 3 and 7)
> - Not started: profiles, the per-frame table, audio, staged footage, stone pelting
>
> How the built system works: [ARCHITECTURE.md](ARCHITECTURE.md).

| # | Step | Part | Size |
|---|---|---|---|
| 1 | Finish the **CAVIAR baseline**, fix what it exposes (rules on real low-res footage) | Core / 1 | Small, in progress |
| 2 | **Profiles** + **per-frame inference table** + **timeline UI** | Core | Medium |
| 3 | Kaggle notebook: **fire/smoke detector** on D-Fire → Part 1 complete | 1 | Small |
| 4 | **Audio pipeline** (faster-whisper + Indic ASR for Kannada, audio events, EN/HI/KN phrase list) and key-moment fusion | 2 | Medium |
| 5 | **Bodycam review UI** + report export + redaction | 2 | Medium |
| 6 | Record **staged bodycam** test footage (team, chest-mounted phone, EN/HI/KN dialogue, consent forms) | 2 | Small |
| 7 | Kaggle notebooks: **weapon/fire detector**, **violence clip classifier**, **crowd density** | 3 (reused by 1, 2) | Large |
| 8 | Stone-pelting and crowd-crush logic, Indian test set, evaluation report | 3 | Large |

Steps 3, 4 and 6 can run in parallel once step 2 lands.

## Decisions (answered 2026-10-15)

1. **Languages: English, Hindi and Kannada.** Speech-to-text and the phrase list must cover all three,
   including code-mixed speech (Hinglish, Kannada-English) and phrases written in Latin script as well as
   Devanagari and Kannada script.
   - Start with **faster-whisper** (Whisper large-v3 class): good for English and Hindi, weaker for Kannada.
   - Evaluate AI4Bharat's open Indic ASR models (IndicWhisper / IndicConformer) for Kannada. Check their licences first.
   - Run language ID per segment and route each segment to the best model.
   - Command and phrase list (e.g. "stop", "ruko", "nillu", "hands up", "haath upar") lives in a reviewable YAML file.
2. **No footage partner.** All training and test data comes from public licensed datasets, **staged
   recordings**, and synthetic data. This means:
   - Bodycam (Part 2): staged scenarios recorded by the team with chest-mounted phones, in English, Hindi
     and Kannada, with signed consent from everyone filmed.
   - CCTV riots (Part 3): public datasets (RWF-2000, XD-Violence, SCVD, UCF-Crime) plus staged mock clashes
     with volunteers (foam props for stones and sticks) and CC-licensed public footage where the licence
     is clear. There is still no real Indian riot footage, and **the report must say so**: accuracy on
     real Indian riots is unmeasured until such data exists.
3. **GPU (arrived: RTX 4050 Laptop 6 GB; results in DESIGN.md §4.4).** The original plan:
   - Install the CUDA build of PyTorch.
   - Re-run `aequitas spike` (including `gemma4:e4b` and YOLO11s/m) and record the results in DESIGN.md §4.3.
   - Raise the defaults (larger detector, per-frame inference at 8 fps, VLM review in real time).
   - Until then, development and Kaggle training continue on CPU and Kaggle GPUs.

## Guardrails (all parts)

These are unchanged from [POLICE_EDITION.md §8](POLICE_EDITION.md):
- No face recognition or identification.
- No profiling by religion, caste, party or attire.
- Human confirmation of every incident and key moment.
- An audit log of every search.
- Retention limits.
- Bodycam exports are redacted by default.
