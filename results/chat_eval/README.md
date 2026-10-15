# Chatbot and stage-2 checks (2026-10-16, RTX 4050 laptop, Ollama gemma4:e2b)

Why: the fine-tuned E2B (kaggle/finetune_gemma.ipynb) is trained on summaries only, but the same model also labels
summaries (stage 2) and parses chat questions. These are the "before" numbers it must not fall below.

## Chat: question -> search filter (`aequitas eval-chat`)

Scored per field: categories, cameras, time range (+-1 min), intent (list/count), include_unconfirmed.
`config/chat_eval.yaml` (28 questions) was used to find and fix problems; `config/chat_eval_holdout.yaml`
(20 questions, different phrasings) was written before the fixes and not used to choose them.

| Set | Parser | All fields right, before fix | After fix |
|---|---|---|---|
| dev (28) | keywords only | 27/28 | 28/28 |
| dev (28) | gemma4:e2b + keyword floor | 19/28 | 28/28 |
| held-out (20) | keywords only | 20/20 | 20/20 |
| held-out (20) | gemma4:e2b + keyword floor | 15/20 | 20/20 |

What Gemma got wrong before the fix: yes/no questions ("Any fights...?") read as counts (7 of 48), and invented or
changed time ranges ("last 30 minutes" -> an hour, "today" where no time was asked: 7 of 48). Fix
(`aequitas/chat/parser.py` `reconcile`): intent comes from the explicit words (how many / count / number of), and
the keyword parser's time range wins when it found one; Gemma's time is kept only when the question has a time
the keywords cannot read (clock times, weekdays, "since"). Also: "thefts" (plural) now matches theft.

Honest caveat: both sets were written by the developer who wrote the keyword parser, so they favour phrasings it
handles. They show that the LLM no longer makes answers worse; they do not show how often real officers' questions
are understood. The peaceful-protest and religious-procession questions map to no incident category, as intended.
Gemma adds ~0.7 s per question.

## Stage 2: label the summary text (`aequitas eval-classify`, config/classify_eval.yaml, 57 summaries)

| Model | Accuracy | Macro-F1 | Incident precision / recall | Time p50 |
|---|---|---|---|---|
| gemma4:e2b (base) | 0.983 | 0.991 | 0.978 / 1.0 | 1.45 s |

Hand-written summaries, partly used during prompt work: a regression check, not a measured accuracy.

## When the fine-tuned model is back

```bash
.venv/Scripts/python -m aequitas eval-chat --models gemma4:e2b aequitas-e2b
.venv/Scripts/python -m aequitas eval-chat --questions config/chat_eval_holdout.yaml --models gemma4:e2b aequitas-e2b
.venv/Scripts/python -m aequitas eval-classify --model aequitas-e2b --label finetuned
```
