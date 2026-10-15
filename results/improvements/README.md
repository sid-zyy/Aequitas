# Accuracy and speed improvements, measured (2026-10-16, RTX 4050 laptop, Ollama gemma4:e2b)

Ideas from the research check (docs/RESEARCH_CHECK.md), each measured before it was kept.

## 1. Window level: 72 saved windows, same frames for every variant (`scripts/bench_variants.py`)

17 fight (5 faculty clips + punching), 12 arrest, 10 fire, 33 normal street windows. Gemma only, no violence
clip model. The fire windows were also looked at when the fire label fix was made, so fire is slightly favourable.

| Variant | Fight right | Arrest right | Fire right | Incident windows flagged | False alarms (normal) | Time / window |
|---|---|---|---|---|---|---|
| Current | 3/17 | 7/12 | 6/10 | 22/39 | 0/33 | 1.22 s |
| Grouped check questions (ASK-Hint style) | 0/17 | 7/12 | 5/10 | 16/39 | 0/33 | 1.57 s |
| **Previous-window context** | 3/17 | **8/12** | **8/10** | **25/39** | 0/33 | 1.25 s |
| Both | 0/17 | 8/12 | 6/10 | 17/39 | 0/33 | 1.35 s |

- Grouped questions: **worse and slower** on the small model (fewer incidents flagged, +0.35 s). Kept in the code,
  off (`vlm.guided_checks: false`). It may suit a larger model.
- Previous-window context: small gain, no cost. **On** (`analysis.label_context: true`).

## 2. Coarse stage: SigLIP scene gate (Flashback / VadCLIP idea), same 72 windows

`google/siglip-base-patch16-224` scores each frame against normal and incident descriptions: **17 ms per window**.

| Threshold | Normal windows skipped | Incident windows skipped |
|---|---|---|
| 0.05 | 21/33 | 0/39 |
| **0.30 (used)** | **28/33** | **0/39** |
| 0.50 | 31/33 | 0/39 |

Every incident window scored 0.90 or more. The threshold was picked on these windows (a demo setting, not a
measured accuracy). Safety rules: never the first window of a camera, never when the violence clip model scores
0.35 or more, and Gemma still looks at least every 30 s.

## 3. End to end: 21 demo clips through the real pipeline (`config/eval_demo*.yaml`)

10 incident clips, 11 normal or optional. "Confirmed" = two windows agree; "first flag" = shown before
confirmation.

| Run | Confirmed caught | Confirmed false alarms | First flag caught | Gemma windows | Gemma time in total |
|---|---|---|---|---|---|
| A: current (run 1) | 7/10 | 0 | 10/10 | 90 | 113.9 s |
| A: current (run 2, nothing changed) | 5/10 | 0 | 10/10 | | |
| C: + context | 7/10 | 1 (fire scene called arrest) | 10/10 | 90 | 118.8 s |
| D: + context + scene gate (run 1) | 5/10 | 1 | 10/10 | **77** | **81.7 s (-28 %)** |
| D: (run 2) | 5/10 | 1 | 10/10 | | |

**Run-to-run noise is about ±2 confirmed incidents**: run A gave 7/10 and then 5/10 with no change, because
windows run in parallel on the GPU and the small model's borderline answers flip. So the end-to-end set is too
small to show the context gain; the window-level test above is the better measure for that.

The scene gate skipped only normal street windows (13: Phoenix Mall 5, KSR station 2, 80 ft road 2, Forum 2,
100 ft road 1, Brigade Road 1) and no incident window, for **28 % less Gemma time**. **On** in config/settings.yaml.

## What limits accuracy now

Every incident clip is flagged at least once (10/10), but short clips (2-4 windows) often miss *confirmation*
because one window disagrees. A "2 of the last 3 windows" confirmation rule is the next thing to test.
