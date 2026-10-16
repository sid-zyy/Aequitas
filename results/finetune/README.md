# Fine-tune run 2: Gemma 4 E2B taught by 12B (Kaggle T4, 2026-10-16)

Notebook: `kaggle/finetune_gemma.py`. Raw output from Kaggle: `FINETUNE.md`, `finetune.csv`, `finetune.png`,
`reports/`, `teacher.jsonl`, `train_log.json`. The LoRA adapter (`lora/`, 172 MB), `ollama.log` and
`review.html` (frame grids) are kept locally and gitignored.

## Setup
- Teacher: gemma4:12b-it-qat summarised 955 training grids. **290 were kept** (arrest 41, fighting 96, fire 6,
  none 109, weapon 38); 516 were dropped because 12B's label disagreed with the dataset label, 145 because the
  self-check found an invented detail.
- Student: E2B, QLoRA r=16, 2 epochs, 74 steps, 16.5 min. Loss 3.96 → 0.08.
- Stage 2 (classifying the summary text) stays base gemma4:e2b. Only stage 1 changed.

## Results: 60 held-out SCVD clips (20 fighting, 20 weapon, 20 none)

| | incident recall | incident precision | accuracy | macro F1 | p50 / clip |
|---|---|---|---|---|---|
| base gemma4:e2b (Ollama) | 0.20 | 0.62 | 0.25 | 0.15 | 1.9 s |
| **fine-tuned E2B** (transformers) | **0.60** | **0.86** | **0.45** | **0.40** | 7.7 s ⚠ |
| gemma4:12b teacher (model_compare) | 0.70 | 0.82 | 0.33 | 0.32 | 5.1 s |

With the close-range (bodycam / webcam) instructions: base 0.20 / 0.67 / 0.27, fine-tuned 0.63 / 0.89 / 0.45.

Second set, 30 clips from other datasets: arrest 0/10 → 5/10, fighting 0/10 → 5/10, **fire 1/10 → 0/10**.

## Read this before quoting the numbers
- **The gain is in-domain.** Training and test clips come from the same SCVD dataset (different videos). The
  out-of-domain check, the 21 Bengaluru demo clips, has **not** been run yet.
- **The time is not comparable.** The Ollama import failed (`--quantize q4_K_M` is no longer accepted by
  Ollama for safetensors imports: only int4/int8/nvfp4/mxfp4/mxfp8), so the fine-tuned model ran in
  transformers with unquantized LoRA. 7.7 s vs 1.9 s says nothing about the served model. The notebook now
  falls back to `--quantize int4`.
- **Weapon: 0/20 labelled weapon.** 9 of 20 weapon clips were flagged as some incident (fighting 5, arrest 3,
  harassment 1), 11 stayed none.
- **Wrong "arrest" labels.** 6 of 60 SCVD predictions were arrest (3 on peaceful clips, 3 on weapon clips);
  SCVD has no arrests. Peaceful clips flagged as an incident: 4/20 (base 5/20).
- **Fire got worse** (6 training examples only).
- The teacher row is copied from model_compare run 1, not measured in this run. The student's accuracy is
  higher than the teacher's there, but its incident recall is still lower (0.60 vs 0.70).

## Next
1. Serve it: vLLM `--enable-lora` with `lora/` on WSL, or re-run the notebook for an int4 Ollama model; then
   time it properly next to base E2B.
2. Run the Bengaluru demo set (`config/eval_demo_final.yaml`) base vs fine-tuned: the out-of-domain number.
3. More fire and weapon teacher examples before another run.
