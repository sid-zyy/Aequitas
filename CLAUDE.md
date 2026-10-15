# Minerva: instructions for Claude

## ⚠️ HACKATHON FACULTY REQUIREMENTS: TOP PRIORITY, ALWAYS

The hackathon faculty set these. They outrank every other plan, idea or product goal in this repo. Before
choosing what to work on, check that it serves them. Never drop or quietly work around one; if a
requirement seems impossible, say so to the owner.

1. **Three tasks, as separate stages:**
   - **(1) Summarization:** the VLM summarises the video.
   - **(2) Classification of the summarized info:** classify the *summary text*.
   - **(3) Chatbot** over the results.
2. **Serve models with a vLLM server.** Ollama is only a dev fallback.
3. **Faster inference with speculative decoding and multi-token prediction (MTP).** Gemma 4 has MTP
   drafters that vLLM supports. Measure the speed-up; don't just turn it on.
4. **Study and implement 4-bit and 8-bit quantization**, with a written comparison of latency and accuracy
   per scheme.
5. **React frontend, FastAPI backend. No Jupyter notebooks** in the deliverable.
6. **The faculty will provide a dataset.** It defines the real evaluation. Build adapters for it rather than
   hard-coding.
7. **How the faculty judge it:**
   - the **approach** (explain the design choices)
   - how well we **demonstrate it end to end**
   - **LATENCY, meaning total execution time (weighted most heavily).** Every stage must be timed, and the
     total time must be visible in the demo.

Status of each item and the build order: HANDOVER.md §6 (Phases 1–5 and 8) and docs/HACKATHON_PLAN.md.

---

Start every new session by reading **HANDOVER.md**. It holds the project state, the decisions, and the
phased plan. Its §6 checklist says what to do next.

## Rules from the project owner
- **Never add Claude as a contributor.** No `Co-Authored-By` trailers and no "Generated with" lines in
  commits or PRs. Commits use the owner's git identity only.
- Commit locally when a piece of work is done and tests pass. **Ask before every `git push`.**
- Before downloading anything large (model weights over ~1 GB, datasets), say what, from where, and how
  big, and get a yes.
- Report results honestly: failing tests, failed experiments, and numbers that got worse all get reported.
  Never present thresholds tuned on a test set as accuracy.

## Guardrails (product design, not optional)
No face recognition or identification of people. No classification of people or crowds by religion, caste,
party, flags, slogans or attire. A peaceful gathering is never an incident. A human confirms incidents.
Full text: docs/POLICE_EDITION.md §8.

## Technical facts that have caught us before
- **Ollama caches identical prompts and images.** Benchmarks must send a different image on every call, or
  latency looks ~10× better than reality. `minerva spike` already randomises per run.
- **supervision's ByteTrack** starts tracks only above `track_activation_threshold + 0.1`. Minerva subtracts
  0.1 so the configured value is the real threshold (`minerva/vision/detector.py`).
- **Gemma 4 E2B on its own** labelled almost everything "person_down" on CAVIAR (docs/EVAL_CAVIAR.md). Never
  trust a VLM label without a second signal: a rule trigger, a classifier, or agreement checks.
- vLLM's structured-output fields are `extra_body={"structured_outputs": {"choice": [...]}}`. The old
  `guided_*` fields were removed in v0.12.

## Commands
```bash
.venv/Scripts/python -m pytest                     # all tests (Windows path; use .venv/bin on Linux/WSL)
.venv/Scripts/python -m minerva web                # web app at http://127.0.0.1:8000
.venv/Scripts/python -m minerva spike --models gemma4:e2b
.venv/Scripts/python -m minerva --settings config/eval_caviar_v2.yaml --cameras config/caviar_cameras.yaml run
.venv/Scripts/python -m minerva --settings config/eval_caviar_v2.yaml eval-report --expected config/caviar_expected.yaml
```

## Code conventions
Match the surrounding code: small modules, pydantic config in `minerva/config.py`, defaults in code and
overrides in `config/*.yaml`, a unit test for every rule or behaviour change (`tests/`, synthetic data, no
model downloads in tests).
