# Judges' Q&A: what they might ask, and what we answer

Short answers first, the reason after. Numbers come from `results/` and `docs/`; where a number has a catch, the
catch is written next to it, so nobody is surprised on stage. Hardware for every laptop number: RTX 4050, 6 GB.

---

## 1. Stack and architecture

**What do we use FastAPI for?**
It is the whole backend: about 50 endpoints that the React app talks to.
- **Every screen's data:** live camera state, incidents, citizen reports, bodycam recordings, uploads.
- **The chatbot:** `/api/ask/stream` streams the answer over server-sent events, so the incident list appears
  first and the summary types itself out.
- **Video:** serves the camera clips and the uploaded videos, and streams the webcam as MJPEG.
- **Sign-in and roles:** police endpoints need a session; citizen endpoints are public and rate-limited; the audit
  log is admin-only.
- **Background work:** an upload queue runs each video through the pipeline while the API stays responsive.
- **Evidence and health:** the evidence-package ZIP, the audit-chain check, and the system health page.
- **The frontend itself:** it serves the built React app, so the whole system is one process on one port.

*Why FastAPI:* the brief asked for React + FastAPI. It is Python, so it runs in the same process as the models
(no second service to call). Its request checking uses pydantic, the same library as our config, so bad input
is rejected before it reaches a model. It streams responses (chat, MJPEG) natively, and documents itself at
`/api/docs`. It binds to 127.0.0.1 only: nothing is exposed to the network.

**Why React?** Brief requirement. In practice: the Live page polls every few seconds and updates many tiles at
once; the chat streams tokens; the citizen portal switches language without a reload. React handles that state
cleanly.

**Why no notebooks?** Brief requirement. The `.ipynb` files in `kaggle/` are generated from `.py` scripts and
used only to run training and benchmarks on Kaggle GPUs; they are not part of the product.

**Draw the pipeline in one sentence.**
Video is cut into 4-second windows (4 frames as a 2×2 grid) → cheap checks skip static or calm windows → Gemma
**summarises** the grid → Gemma **classifies the summary text** → a motion model votes → two agreeing windows
confirm an incident → it is stored with its key frame and file hash → an officer confirms it, and the **chatbot**
answers questions over the stored incidents.

**Why are the three stages separate?** The brief asks for it, and it makes each stage measurable and swappable.
The summary is written without seeing the label list, so it isn't steered towards a label; the label is then
chosen from the text alone. Each stage has its own timing.

**Why Ollama and not vLLM?** vLLM couldn't fit Gemma 4 E2B in 6 GB. Ollama runs it at 4-bit with the MTP
drafter. The vLLM path is still in the code (`vlm.backend: openai`) for a bigger GPU.

---

## 2. Model choice

**Why Gemma 4 E2B?** It makes the fastest whole system on our laptop and raises the fewest false alarms.
- Fastest end to end: 0.91 s per window vs 1.01 s for Qwen3-VL 2B (21 demo clips: 1 min 48 s vs 2 min 24 s).
- Fewest false alarms on normal clips among models that fit: 10 % (Qwen 25 %).
- It ships an MTP drafter for speculative decoding; Qwen has none.
- One model does every language job: Hindi/Kannada translation kept every fact in 6/6 reports (Qwen 2/6).

**But Qwen finds more incidents on its own (70 % vs 25 %). Why not Qwen?**
On its own, yes. In the app, Gemma never decides alone: the violence model votes with it. With that second
signal the gap shrinks to 1–2 demo incidents, and Qwen is 33 % slower end to end. We prioritise speed along with
accuracy, so we chose Gemma plus a second signal.

**Why not a bigger model, like Gemma 12B (70 %)?** It doesn't fit in 6 GB and is about 3× slower (5.1 s per
clip on a T4). We distilled 12B's answers into E2B instead (below).

**Why "25 %" on one run and "15 %" on another?** Two different random draws of 60 clips (46 shared), and only 40
incident clips per draw: one clip moves recall by 2.5 points. A 180-clip re-run is prepared on Kaggle
(`kaggle/model_compare.ipynb`, `SET = "e2b"`).

---

## 3. Accuracy

**How accurate is it on the faculty test clips?**
All 6 flagged as incidents; the right label confirmed on 4 (precision and recall 0.80). Misses: v003 is 3 s long,
one window, labelled "person down" and never confirmed (needs two agreeing windows); w001 shows a weapon and we
called it a chase (no trained weapon detector yet).
*Catch, if asked:* the violence model's 0.9 cut-off was chosen with these clips in view. Every clip scored 0.96
or more, so the result doesn't hinge on the exact cut-off. Our honest held-out number is the next one.

**What's the held-out number?** The violence model caught 77 % of fights on 300 unseen real CCTV clips, with
14 % false alarms (neither threshold tuned on that data).

**Gemma alone misses the fights. Isn't stage 2 then pointless?**
Gemma describes what it can see; on low-detail CCTV it sees people standing and sitting (0/5 fights). That is why
the label is never trusted alone: a motion model (r3d-18, trained on ~8,000 clips) watches movement and votes.
Stage 2 still labels everything Gemma does see (arrests, fire, accidents, person down), and scored 0.98 on 57
labelled summaries (a hand-written regression set, not a measured field accuracy).

**What did fine-tuning do?** E2B taught by 12B's answers: incidents found 20 % → 60 % on held-out SCVD clips.
*Catch:* same dataset as training (different videos), and the fine-tuned model isn't served in the app yet.

**Why RF-DETR for detection?** CCTV people are tiny. RF-DETR Small finds 48 % of people under 30 px vs 3 % for
YOLO11n, at 26 ms a frame, Apache-2.0 licence. On CAVIAR incident clips, alarms went from 0/6 to 3/6.

**How do you stop false alarms?** Three layers: the motion model must agree (or score ≥ 0.9 on its own), two
windows in a row must agree to confirm, and an officer confirms every incident. A peaceful gathering is written
into the "none" definition, so it is never an incident.

---

## 4. Latency (weighted most heavily)

**How fast is it end to end?** The six test clips (39 s of footage) in 25 s warm, 40 s on a cold start (model
loading). Summarise 0.58 s, classify 0.32 s, chat answer 0.45 s (p50).

**What made it faster?** Each measured on its own:
- Scene gate (SigLIP, 44 ms): calm scenes skip Gemma → 28 % less Gemma time on the demo clips.
- Skip static windows, cap summary length, and keep several windows in flight: 33 s → 5 s on an 80 s test video
  (that ladder was measured with Qwen on vLLM on a Kaggle T4).
- Speculative decoding (MTP): chat 22 % faster, a window 9.5 % faster.
- 4-bit quantization: fits the GPU with room for the gate.
- An officer's question jumps ahead of live analysis, and questions the keyword parser fully reads skip a model
  call: under live load, a question went from 4–5 s to near-instant, a streamed answer from 7–8 s to 2.9 s.

**Why does speculative decoding help chat more than video?** A small drafter guesses the next 3 tokens and Gemma
checks them in one pass. It speeds up *writing*. A window writes only ~36 tokens and spends most of its time
reading the image, so it gains 9.5 %; chat writes more and gains 22 %. Labels didn't change (0/33 false alarms at
every setting). Draft length 3 was best; 6 was slower.

**Benchmark trap we avoided:** Ollama caches identical prompts and images, which makes repeat calls look ~10×
faster. Every benchmark sends a different image or question each time.

---

## 5. Quantization

**What did you compare?** Gemma E2B at 16-bit, 8-bit, 4-bit (Q4_K_M) and 4-bit QAT on the same 72 windows.
- Detection didn't drop: 16-, 8- and 4-bit all flagged 20/39 incident windows with 0/33 false alarms (QAT 18/39).
- 16-bit is 2.2× slower than 8-bit and needs ~5.5 GB: no room left for the scene gate on 6 GB.
- 4-bit is 8 % faster than 8-bit on its own (QAT vs Q8, same GPU) and named the most fights and fires correctly.
- On the laptop, Gemma 4-bit with MTP: 1.16 s per window, vs Qwen 4-bit 1.60 s on the same windows.
*Catch:* the Gemma precision comparison ran on a Kaggle T4; only the Q4_K_M run had the MTP drafter on, so its
lead over 8-bit is partly speculative decoding.

---

## 6. Chatbot

**How do you stop the chatbot making things up?** The model never writes SQL and never lists incidents. It only
fills a small search form (incident type, place, time). Code runs the query; the incident list comes straight
from the database. The optional summary is thrown away if it cites any incident ID that wasn't retrieved.

**How well does it understand questions?** 48/48 test questions turned into the right search (28 used while
building it, 20 held out). A keyword parser is the floor; the model fills in what keywords can't read
("since 2 pm").

**Does it know where places are?** Yes. Every camera knows its locality: "fights in Indiranagar" means all its
cameras, misspellings like "Kormangala" work, and "Where is Double Road Restaurant?" answers "in Indiranagar".

**What won't it answer?** Who someone is, or anything about religion, caste or party. "Who is the attacker?"
gets "can't be determined from the analysis". A peaceful protest returns no incidents.

---

## 7. Citizen portal

**Why a citizen side?** Most incidents are filmed by bystanders, not police cameras. A student can report from a
phone in a minute, in English, Hindi or Kannada, without an account, and track it with a report ID.

**Why is the help assistant rule-based, not an LLM?** So it can never invent a helpline number or say "help is on
the way". It shows only official numbers (112, 100, 101, 108, 1091), marks urgent reports, and can take the whole
report inside the chat: attach a video, answer a few fixed questions, give consent, submit.

**How do you handle fake or AI-generated videos?** Every upload is fingerprinted (SHA-256) and re-uploads are
linked. An AI-generated check flags "AI-like frames" (34 of 50 AI clips caught, 1 of 80 real clips wrongly
flagged). It is set to prefer missing a fake over discrediting a real citizen, and it's evidence for an officer,
never a verdict.

---

## 8. Evidence, accountability, operations

**Can this footage be used as evidence?** Each incident exports as a self-checking ZIP: the original video byte
for byte, the key frame, what the AI said, what the officer decided and when, the models used, and SHA-256
hashes anyone can verify. It includes a *draft* certificate under Section 63 of the Bharatiya Sakshya
Adhiniyam, 2023 for the responsible officer to complete. (Draft, not legal advice.)

**How do you know nobody tampered with the record?** Every action (sign-in, view, decision, export) goes in an
audit log where each entry is sealed to the one before it with an HMAC key stored outside the database. Editing
or deleting an entry breaks the chain and the Audit log page shows where. *Honest limit:* someone with both the
key and the database could rewrite it, which is why each evidence export also records the latest chain hash.

**How would an operator know it's working?** The System page checks the model server, GPU memory and
temperature, how far live analysis is behind, failed answers, the upload queue, the audit chain and free disk,
every 5 seconds.

---

## 9. Ethics, privacy, guardrails

**Isn't this mass surveillance?** It describes actions, never identities. No face recognition, no tracking people
across cameras by appearance, no sorting people by religion, caste, party, flags, slogans or clothes. A peaceful
protest is a right and never an incident. A human confirms every incident and every decision is logged.

**Where does the data go?** Nowhere. Everything runs on one laptop, offline: no cloud API, nothing to leak, no
per-call bill. Non-local model hosts are refused unless explicitly allowed.

**What about bias?** The labels describe actions only, and training data with identity words was dropped. Small
models are biased (Gemma alone called 11/11 CAVIAR clips "person down"), which is exactly why no label is trusted
without a second signal and a human.

---

## 10. "How is this production ready?"

**The 20-second answer:** "It's built the way a station would need it: it runs offline on one affordable
laptop, every decision is made by a person and logged so it can't be quietly changed, every incident exports as
checkable evidence, it keeps working when a part fails, and an operator can see its health at a glance. It's
ready for a pilot in a real station; that pilot is the next step."

**If they push, the evidence:**
- **Runs where police can run it:** one 6 GB laptop, offline, no cloud bill, footage never leaves the station.
  Measured end to end: six test clips in 25 s; every stage timed and shown on screen.
- **People stay in charge:** the AI only flags; an officer confirms or dismisses, and the AI's call and the
  officer's call are stored separately.
- **Accountable:** a tamper-evident audit log of every sign-in, view, decision and export; one-click evidence
  packages with SHA-256 hashes and a draft certificate.
- **Secure by default:** salted scrypt passwords, sessions in HttpOnly SameSite cookies with only a hash stored,
  lockout after 5 failed sign-ins, admin and officer roles, bound to localhost; citizen uploads are rate-limited,
  size-capped and type-checked.
- **Keeps working when something fails:**
  - If the model server is down, the chat falls back to its keyword parser (48/48 on our test questions).
  - Queues have backpressure, so a slow model never piles up unbounded work.
  - Jobs cut off by a restart are marked failed; the AI-generated checks resume on their own.
  - An officer's question always goes ahead of background analysis.
  - Old databases upgrade themselves on start.
- **Operable:** a System health page checks the model server, GPU, live lag, failed answers, queue, audit chain
  and disk every 5 seconds. Cameras, categories, thresholds and the model are set in config files, not code.
- **Tested:** 258 automated tests (synthetic data, no model downloads), plus repeatable evaluation scripts for
  every number on the slides.

**Say plainly what it isn't yet (it builds trust):** it hasn't run in a real station or on real station
cameras (the Bengaluru feeds are stock footage); our test sets are small; one laptop covers one locality (about
5 cameras) at a time; HTTPS and single sign-on aren't set up. A station pilot with real cameras is what turns
"production-minded" into "in production".

---

## 11. Hard questions: answer honestly

**Your test sets are tiny.** Yes: 6 faculty clips, 21 demo clips, 60–180 benchmark clips. One clip moves the
faculty score by 17 points, so we read these as ranges, and we report held-out numbers next to tuned ones.

**Was anything tuned on the test clips?** The violence model's 0.9 cut-off and the scene gate threshold were set
with the test and demo clips in view. The held-out figure for the violence model is 77 % recall at 14 % false
alarms on 300 unseen clips.

**Is the footage real?** The six faculty clips are real CCTV. The Bengaluru camera feeds are free stock footage
standing in for cameras (`data/footage/SOURCES.md`); camera names are real landmarks so the demo reads naturally.

**What would you do with more time?** Serve the fine-tuned E2B and test it on the faculty clips; train the
weapon/fire detector; transcribe bodycam audio; warm the model at start-up to remove the cold-start delay; re-run
the model comparison on 180 clips.

**How does it scale beyond one laptop?** One laptop analyses one locality at a time (5 cameras). The same code
runs on a bigger GPU with vLLM batching (`vlm.backend: openai`, more `concurrency`), or one box per station.
