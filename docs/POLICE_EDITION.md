# Aequitas for police and public-order CCTV (India) — proposal

Status: proposal, not yet built. Extends [DESIGN.md](DESIGN.md).

## 1. Scope

**Goal:** help a control room see and log *incidents* during public-order situations (clashes, arson,
stampede risk, injuries), with an inference available for **every sampled frame / timestamp**, not only
for detected events.

**What "Indian context" means here** is the operating conditions, not politics:
- Very dense crowds: rallies, religious processions, melas, festival immersions, stations, markets.
- Mixed scenes: two-wheelers, autos, handcarts and pedestrians in the same space.
- Footage quality: low-res or compressed CCTV, night and sodium lighting, rain and dust, mobile and
  body-worn cameras, drones.
- Local objects: lathis and sticks, iron rods, swords, burning tyres, makeshift barricades.

**Out of scope, by design** (section 8 explains why):
- Identifying people, including face recognition.
- Classifying people or crowds by religion, caste, party, flags, attire or slogans.
- Treating a peaceful gathering or protest as an incident.

## 2. Incident taxonomy (v2)

Each row is detected from cheap per-frame signals (Tier 1–2, §3) and confirmed or summarised by the VLM (Tier 3).

| Category | What triggers it | Main signal |
|---|---|---|
| `violent_clash` | Group fighting, assault, people beaten on the ground | Clip violence score + pose + close, erratic tracks |
| `projectile_throwing` | Stone pelting, bottles or objects thrown | Overhead-arm throwing pose + small fast objects in optical flow (stones are too small to detect reliably) |
| `arson_fire` | Fire on vehicles, shops or tyres | Fire/smoke detector (D-Fire classes), persistent over N frames |
| `smoke_gas` | Dense smoke, including tear-gas clouds | Smoke detector + area growth. Cannot reliably tell tear gas from other smoke; reported as smoke. |
| `weapon_visible` | Lathi/stick/rod, knife/sword, firearm | Fine-tuned detector. Note: police carry lathis too, so this flags *visible weapons*, not offenders. |
| `vandalism` | Damaging vehicles or property, breaking barricades | Clip classifier (UCF-Crime "Vandalism") + VLM |
| `crowd_crush_risk` | Density above a safe limit, compressive or turbulent flow | Density map (people/m² with camera calibration) + flow coherence. **Highest life-safety value.** |
| `panic_dispersal` | Sudden scattering or running | Mean speed jump + divergent flow (already planned as `dispersal`) |
| `cordon_breach` | Crossing a police line or barricade into a restricted zone | Line/zone crossing (existing tailgating geometry) |
| `person_down` | Injured or collapsed person | Existing rule (pose) |
| `police_use_of_force` | Lathi charge, baton strikes | Logged for **accountability**, with the same evidence standard as other events |

The `peaceful_gathering` / `crowd_present` *state* is recorded on the timeline as context only. It never
creates an incident and never alerts on its own.

## 3. Per-frame / per-timestamp inference

Today only triggers produce records. The new design writes a row for **every sampled frame**, so any moment can be queried.

```
every sampled frame (4–8 fps)          Tier 1  detector + tracker + density + optical flow + pose (on demand)
sliding 2 s window, every 0.5 s        Tier 2  video clip classifier → violence / vandalism / normal scores
events, and any timestamp on demand    Tier 3  VLM caption + category check (Gemma 4 / Qwen-VL)
```

New table (one row per sampled frame; downsampled to 1/s after N days):

| Column | Meaning |
|---|---|
| `camera_id, ts` | Key |
| `person_count`, `density_ppm2`, `density_level` | Detection count when sparse; density-map estimate when dense (>~80 people) |
| `flow_speed`, `flow_coherence`, `flow_divergence` | Optical-flow statistics (surge, panic, turbulence) |
| `violence_score`, `vandalism_score` | From Tier 2, assigned to every frame in the window |
| `fire_score`, `smoke_score` | Tier 1 detector max confidence |
| `weapons` | JSON list of `{class, conf, box}` |
| `anomaly_score` | Combined score that drives the timeline colour |
| `keyframe_ref` | Nearest stored thumbnail |

**UI:** a timeline under each video with score lanes (crowd, violence, fire/smoke, weapons). Clicking any
point shows that frame's inference card. "Describe this moment" runs the VLM on that timestamp.

**Chat:** "What was happening on camera 3 at 14:32?" reads the frame table (plus a VLM caption if one is
cached), alongside the existing event queries.

**Cost on the target GPU:** about 15–25 ms per frame for Tier 1 + Tier 2 together with nano/small models.
On this CPU laptop, expect about 2–4 fps per camera with OpenVINO INT8. The VLM stays event-driven.

## 4. Datasets

Checked to exist on 2026-10-15. **Licences vary and many are research-only.** Confirm each one before use.

| Use | Dataset | Notes |
|---|---|---|
| Violence clip classifier | **RWF-2000** | Real CCTV fights, 2,000 clips. The core training set. |
| Violence / riot | **XD-Violence** | 4,754 videos, 217 h, weak labels. Classes include **Riot**, Fighting, Explosion, Shooting, Abuse, Car accident. Movies + YouTube. |
| Anomaly (vandalism, arson, assault…) | **UCF-Crime** | 13 anomaly classes, long untrimmed CCTV |
| Weaponised violence on CCTV | **SCVD** (Smart-City CCTV Violence Detection, MBZUAI) | CCTV only, with weapon scenarios. Get it via the paper's links; not on Kaggle. |
| Small vs large violence vs peaceful gatherings | **MSV-PG** (Frontiers in CS, 2024) | Directly matches the "protest is not an incident" requirement. Check availability. |
| Fire / smoke detector | **D-Fire** | 21,527 images, YOLO format |
| Knives | **OD-WeaponDetection (knife)** | 2,078 images, CC BY-SA 4.0 |
| Knife / pistol | Kaggle "Knife vs Pistol" and Roboflow sets | No good public lathi/stick/rod set exists, so build a custom one (§5) |
| Crowd density | **JHU-CROWD++** | Includes Indian scenes among its sources; weather variety |
| Indian street domain | **IDD** (India Driving Dataset, IIIT-H) | 10k images of Indian roads. For person/vehicle domain adaptation; licence to confirm with IIIT-H. |
| Indian dense urban video | **DenseWorld 1.0** (arXiv 2608.01049) | ~1,000 h across 22 Indian cities. Public release unconfirmed. |
| Mass-gathering crowds | Kumbh Mela 2016 data (IISc) | Research data. Access only by request to the IISc lab. |
| Indian campus anomalies | **IITB-Corridor** | IIT Bombay, staged anomalies. Exact class list to confirm from the paper. |

**Gap:** there is no licensed Indian dataset of riots, stone pelting or lathi charges. Options, best first:
1. Footage from the partner police unit under a data-sharing MoU (proper legal basis; real conditions).
2. Staged recordings with consenting volunteers (college ground, mock clashes, foam "stones", props).
3. Synthetic data: game-engine crowds and fire for density and fire classes.

Avoid bulk-scraping YouTube or news riot footage. It raises copyright and privacy problems and has
mislabelled provenance (fact-checkers regularly find old or foreign clips relabelled).

## 5. Training on Kaggle

Kaggle notebooks give free GPUs (T4 ×2 or P100, with a weekly quota of about 30 h; check current limits).
Upload datasets as **private** Kaggle datasets and respect each licence.

| Notebook | Model | Data | Output |
|---|---|---|---|
| `det-public-order` | YOLO11s fine-tune; classes: person, stick/rod, knife/sword, firearm, fire, smoke | D-Fire + OD-Weapon + Knife-vs-Pistol + **custom lathi/rod set** (CVAT/Label Studio, ~1.5k boxes) + IDD persons | `.pt` → ONNX / OpenVINO INT8 |
| `clip-violence` | Light video classifier (e.g. X3D-S or MoViNet-A0 class), 16 frames × 2 s | RWF-2000 + XD-Violence (Fighting, Riot) + SCVD; hard negatives: sports, festivals, dancing, processions | ONNX |
| `crowd-density` | Density/point model (CSRNet/P2PNet-style) | JHU-CROWD++, fine-tuned on Indian frames | ONNX |
| `eval-india` | — | **Held-out Indian test split only**, never trained on | Report |

Hard negatives matter in India: garba, dhol processions, cricket celebrations, Holi and immersion crowds
look "violent" to generic models. They must be in the negative training set and the test set.

## 6. Accuracy

"Analyse everything accurately" is not achievable, and the report should not claim it. What we commit to:
- Precision and recall per category, on an **Indian held-out set** reported separately from the public benchmarks.
- **False alarms per camera-hour** on long normal footage (rallies, festivals, markets). This number
  decides whether operators keep the system switched on.
- Calibrated confidence: Tier 2 scores calibrated on validation data, plus agreement with the VLM (as now).
- Robustness slices: day vs night, rain, low resolution, drone vs fixed, crowd size.
- A human review queue. Nothing is "confirmed" for operational use without an operator.

## 7. Phased plan

| Phase | Deliverable |
|---|---|
| P1 | Frame-inference table + timeline UI, using current models (count, flow, motion, fire via D-Fire YOLO) |
| P2 | Kaggle: detector fine-tune (weapons, fire, smoke) + violence clip classifier → plug into Tier 1/2 |
| P3 | Crowd density + crush-risk and panic-dispersal rules with camera calibration |
| P4 | Indian test set, evaluation report, false-alarm tuning |
| P5 | Review queue, audit log, retention controls (§8) |

## 8. Guardrails (part of the design, not optional)

Public-order surveillance in India touches Article 19(1)(b) (peaceful assembly), the *Puttaswamy*
privacy judgment (legality, necessity, proportionality), and the DPDP Act 2023. A system that is
incident-centred and auditable is easier to defend, and more useful.

- **No identity:** no face recognition, re-identification across cameras, or person search. Track IDs
  are per-camera and short-lived.
- **No group profiling:** no classes or prompts for religion, caste, party, flags, slogans, attire or
  symbols. The VLM prompt forbids describing them, and summaries are filtered for them.
- **Peaceful gathering ≠ incident:** crowd presence is context; alerts need a behavioural incident.
- **Symmetric logging:** police use of force is recorded with the same evidence standard as other incidents.
- **Human in the loop** for every confirmation; full **audit log** of who searched what; **retention
  limits**, with automatic deletion of non-incident footage after N days.

## 9. Decisions needed

1. Deployment target: fixed CCTV, body-worn, drones, or all three? This changes the models (motion, viewpoint).
2. Is a partner police unit providing footage under an MoU? This is the main source of Indian training and test data.
3. Target GPU for deployment (sizes the models).
4. Which phase first. Recommendation: **P1 + P2**, since they deliver per-frame inference and the biggest accuracy gain.
