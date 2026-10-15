# Research check (2026-10-16): are we missing anything obvious?

A quick check of papers, GitHub and Kaggle against what Aequitas does. Links at the end.

## 1. Summarize → classify the text (our core design)

**LAVAD** (CVPR 2024, training-free video anomaly detection) uses the same idea: a captioning VLM describes
each frame window, then an LLM scores the text. It was evaluated on UCF-Crime and XD-Violence (real
surveillance). So our two-stage design is an established approach, which is useful for the "approach" part
of the judging. Two of LAVAD's extra steps map onto things we have or could add:

| LAVAD step | Aequitas |
|---|---|
| Score smoothing over neighbouring windows | `confirm_votes: 2`: an event is confirmed only when 2 windows agree |
| The LLM sees several captions in time, not one | **Not yet.** Idea: give the label step the previous window's summary, so "officers approach" → "officers hold his arms" reads as an arrest |

Newer work (Holmes-VAD, VERA, Probe-VAD) fine-tunes the VLM or learns guiding questions; those need
training data we do not have. Our hand-written label rules are the cheap version of VERA's guiding questions.

## 2. Fight detection (violence clip model)

Our r3d_18: 95 % accuracy on 1,569 held-out validation videos, 5/5 faculty clips. Published RWF-2000 results
range 87 % (the RWF-2000 baseline) to about 95 % (2025 work), so we are in the normal range; the numbers
are not directly comparable (different splits and datasets). A skeleton-based model (90 % on RWF-2000 with
60k parameters) is a possible lighter alternative, but not needed.

## 3. AI-generated video check: a real problem found

Our frame detector (Organika/sdxl-detector, built for still images) scored **5 of 22 real clips 100 %
"AI"** (accident, fire, punching, flyover, an arrest). Published model cards report 9-27 % false positives
for image detectors on real photos, worse on unfamiliar camera pipelines, which matches what we saw. No
video detector with released weights clearly survives compression yet (DeCoF and a Qwen2.5-ViT based
detector are the most promising research code).

**Done:** the frame detector alone can now only say "uncertain, needs a human look"; only provenance evidence
(C2PA content credentials, or an AI tool named in the file metadata) can say "likely AI". This matches the
field's advice: provenance first, detectors as a hint.

**Possible next step:** test a newer detector (e.g. a SigLIP2 + DINOv2 ensemble reporting about 9 % false
positives on casual real photos) on our 22 real clips before trusting it. Needs a model download.

## 4. Things we already match

- Fire / smoke / weapon detection with YOLO trained on D-Fire and weapon sets: standard (hazard notebook).
- Human confirmation of every incident, no face recognition: matches responsible-AI guidance for police use.

## Links

- LAVAD: https://github.com/lucazanella/lavad (paper: https://openaccess.thecvf.com/content/CVPR2024/papers/Zanella_Harnessing_Large_Language_Models_for_Training-free_Video_Anomaly_Detection_CVPR_2024_paper.pdf)
- Holmes-VAD: https://www.alphaxiv.org/abs/2406.12235v2 · survey: https://github.com/Darcyddx/VAD-LLM · Probe-VAD: https://arxiv.org/pdf/2609.17211
- RWF-2000: https://alphaxiv.org/abs/1911.05913 · Vi-SAFE: https://arxiv.org/pdf/2509.13210 · skeletons: https://github.com/atmguille/Violence-Detection-With-Human-Skeletons · short-clip method: https://arxiv.org/abs/2511.10866v1
- AI video detection: DeCoF https://github.com/LongMa-2025/DeCoF · native-scale detector https://arxiv.org/html/2604.04634 · compression study https://arxiv.org/pdf/2403.16638 · benchmark https://arxiv.org/html/2601.11035v1
- Image detector cards: https://huggingface.co/Bombek1/ai-image-detector-siglip-dinov2 · https://huggingface.co/husseinelsaadi/aidetect-vit-b16

## 5. Deep dive: coarse-to-fine ("first a big category, then the detail")

The idea is backed by recent papers, in three forms:

| Form | Paper | What it does | Result reported | Fit for us |
|---|---|---|---|---|
| Group-wise guiding questions | **ASK-Hint** (WACV 2026) | Groups classes (violence, property crime, public safety); each group has a few action questions the VLM answers ("is anyone being struck?") | UCF-Crime AUC 89.8 %, training-free; fine-grained prompts beat abstract ones by up to 30 % AUC | **High.** Same VLM call, no extra latency; our summary prompt already lists cues, this structures them by group |
| Cheap gate, then the big model | **Flashback** (2025), VadCLIP (AAAI 2024) | Embed the window with CLIP-style model, compare with text of normal / anomalous scenes; no LLM call per window | Flashback: UCF-Crime AUC 87.3 %, 0.7 s per segment, consumer GPU | **High for latency.** Skip the VLM on windows that are clearly normal; the VLM only explains candidates |
| Binary first, then class | "Learning to Tell Apart" (2025), VadCLIP | One branch: anomaly yes/no; second branch: which class | weakly supervised (needs training) | Medium: our violence / hazard models already act as the binary branch |

**View type (close-up / bodycam vs fixed CCTV):** no paper offers a ready router, but the cue is well known: a
fixed camera has a still background, a body-worn or phone camera moves with its wearer (egocentric vs
exocentric surveys). We already switch prompts per camera (`view: close`); this could be detected
automatically from background motion, with no model call.

## 6. Evaluation everyone compares against

**UCF-Crime** (1,900 real CCTV videos, 13 classes incl. Arrest, Arson, Assault, Fighting, Road Accidents,
Shooting, Vandalism) is on Kaggle and its classes map almost one-to-one onto ours. A Kaggle run on its test
split would give one number comparable with LAVAD (80.3), VERA (86.6), Flashback (87.3), Holmes-VAU (89.0) and
ASK-Hint (89.8) AUC. Different metric from our accuracy, so it needs a per-window anomaly score.

## 7. Measured on our laptop: the E4B cascade does not fit 6 GB

Run B (E2B + E4B re-checking flagged windows) crashed Ollama with a CUDA error while it swapped the two
models ("llama-server process no longer running ... CUDA error"), and the run stalled after 10 of 21 clips.
On a 6 GB GPU the cascade needs one model unloaded before the other loads, which costs seconds per switch.
Keep E2B + the violence clip model on the laptop; the cascade is for a bigger GPU.

## More links

- ASK-Hint: https://arxiv.org/html/2510.02155v1 (WACV 2026: https://openaccess.thecvf.com/content/WACV2026/html/Zou_Unlocking_Vision-Language_Models_for_Video_Anomaly_Detection_via_Fine-Grained_Prompting_WACV_2026_paper.html)
- Flashback: https://arxiv.org/pdf/2505.15205 · TRACES: https://arxiv.org/pdf/2511.00580
- VadCLIP: https://arxiv.org/abs/2308.11681 · VERA: https://arxiv.org/pdf/2412.01095 · Holmes-VAU: https://arxiv.org/pdf/2412.06171
- VADTree: https://arxiv.org/pdf/2510.22693 · Learning to Tell Apart: https://arxiv.org/pdf/2511.10334
- Egocentric vs exocentric survey: https://huggingface.co/papers/2506.06253
- UCF-Crime on Kaggle: https://www.kaggle.com/datasets/webadvisor/real-time-anomaly-detection-in-cctv-surveillance
