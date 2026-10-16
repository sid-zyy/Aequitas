# Fine-tuned Gemma 4 E2B vs base (Kaggle T4, Ollama, our two-stage pipeline)

Test: the 60 held-out SCVD clips of model_compare (20 per class), none used in training; faculty clips excluded. Student trained on 290 teacher summaries ({"arrest": "41/160", "fighting": "96/311", "fire": "6/160", "none": "109/159", "weapon": "38/165"}), 2 epochs, QLoRA r=16.
Note: the SCVD training clips come from the same dataset as the test clips (different videos), so this measures in-domain gain; the 21 Bengaluru demo clips are the out-of-domain check.

| model                                                 |   incident_recall |   incident_precision |   accuracy |   macro_f1 | valid   |   time_per_clip_p50_s |
|:------------------------------------------------------|------------------:|---------------------:|-----------:|-----------:|:--------|----------------------:|
| gemma4:e2b (base)                                     |               0.2 |               0.6154 |     0.25   |     0.1493 | 60/60   |                 1.877 |
| fine-tuned E2B (transformers; stage 2 base)           |               0.6 |               0.8571 |     0.45   |     0.4033 | 60/60   |                 7.668 |
| gemma4:12b-it-qat (teacher, from model_compare run 1) |               0.7 |               0.8235 |     0.3333 |     0.3172 | 60/60   |                 5.11  |

## Second test set: 30 held-out clips from the other datasets (riot, arrest, fire, peaceful crowds...)

Share of clips given an accepted label (for none: left as none, i.e. no false alarm).

| model | arrest (n=10) | fighting (n=10) | fire (n=10) |
|---|---|---|---|
| gemma4:e2b (base) | 0/10 | 0/10 | 1/10 |
| fine-tuned E2B (transformers; stage 2 base) | 5/10 | 5/10 | 0/10 |

## Same 60 SCVD clips with the close-range (bodycam / webcam) instructions

50 % of the training summaries used these instructions.

| model | incident recall | incident precision | accuracy | time per clip p50 |
|---|---|---|---|---|
| gemma4:e2b (base) | 0.2 | 0.6667 | 0.2667 | 1.938 s |
| fine-tuned E2B (transformers; stage 2 base) | 0.625 | 0.8929 | 0.45 | 8.257 s |

Ollama import failed: the fine-tuned row ran in transformers, so its time is not comparable.

Training data checks: {"label disagrees": 516, "check found an invented detail": 145, "teacher failed": 2, "guardrail words": 2} dropped. Spot check: review.html.