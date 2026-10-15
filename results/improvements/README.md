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

## 4. Confirmation rule: replay of the saved windows of all six runs (no GPU, no run-to-run noise)

Every incident clip was flagged at least once (10/10), but short clips missed *confirmation*. The per-window labels
saved by the six runs above were replayed through each rule; the current rule reproduces the reported numbers exactly.

| Rule | A | A rep | C ctx | B cascade | D gate | D rep | New false alarms |
|---|---|---|---|---|---|---|---|
| Current: 2 windows in a row, same label | 7/10 | 5/10 | 7/10 | 2/10 | 5/10 | 5/10 | |
| 2 of any 3 windows, same label | 7/10 | 5/10 | 7/10 | 2/10 | 5/10 | 5/10 | 0 |
| 2 in a row, any incident label | 8/10 | 6/10 | 8/10 | 3/10 | 6/10 | 6/10 | 1-3 (wrong labels) |
| **Clip model votes "fighting" at >= 0.9** | **8/10** | **7/10** | **8/10** | **5/10** | **7/10** | **7/10** | **0** |

Why "2 of 3" changes nothing: the faculty fight clips are only 8 s (two windows). The violence clip model scores
both windows 0.86-1.0, but Gemma calls one of them person_down or suspicious_activity, and the fusion rule keeps
Gemma's more specific label, so "fighting" got one vote. With `analysis.clip_votes: true`, a window at the clip
model's stand-alone threshold (0.9, set earlier on the street clips) also votes "fighting". No normal street clip
has any window at 0.9. **On.** The rule was chosen after looking at these clips, so this is a demo result, not a
measured accuracy. Still missed: arrest-1 and arrest-2 (one arrest window each), fac-v001 when its second window
scores 0.86; the fire-rescue "arrest" false alarm comes from Gemma, not from this rule.

## 5. The current setup end to end, twice (context + scene gate + clip votes, `config/eval_demo_final*.yaml`)

| Run | Confirmed caught | Confirmed false alarms | First flag caught | Gemma windows | Gemma time | Wall time, 21 clips |
|---|---|---|---|---|---|---|
| E (run 1) | 7/10 | 1 (fire-rescue: arrest) | 10/10 | 77 | 81.2 s | 1 min 48 s |
| E (run 2) | 7/10 | 1 (fire-rescue: arrest) | 10/10 | 77 | 76.6 s | 1 min 49 s |

Both runs match the replay's prediction for this setup. Still missed: fac-v001 (second window scores 0.86),
arrest-1, arrest-2.

## What limits accuracy now

Arrests: Gemma labels only one window of arrest-1 and arrest-2 as arrest. The fire-rescue clip gets a confirmed
"arrest" when previous-window context is on.
