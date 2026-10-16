"""Speculative decoding (MTP) in Ollama, without vLLM: gemma4:e2b ships Gemma 4's MTP drafter (a 98 MB 'draft' layer),
and `draft_num_predict` sets how many tokens it proposes per step (0 = off; the model's default is 3).

    .venv/Scripts/python scripts/bench_spec.py --drafts 0 1 2 3 4 6

Workloads, all through the app's own client and prompts (aequitas.understand / aequitas.chat):
  stage 1  summary of each saved 2x2 window grid (data/bench_windows, 72 windows, image in, JSON out)
  stage 2  label of that summary (text in, one-word JSON out)
  chat     grounded answer over 10 stored summaries (text in, 1-3 sentences out, streamed as in the app)
Every setting runs the same inputs. Passes go 0..6 then 6..0 so GPU warm-up and heat hit every setting equally.
Per setting: median seconds per call, decode tokens/s (Ollama's eval_count / eval_duration) and whether the outputs
match the no-draft run (temperature 0, so speculation should not change them; GPU arithmetic can still flip a token).
Close the web app's Live page first: anything else using the GPU or Ollama skews the times. Write-up: docs/SPECULATIVE.md.
"""

import argparse
import json
import random
import statistics as st
import time
from pathlib import Path

from aequitas.chat.answer import _stream_prompt
from aequitas.config import Settings
from aequitas.understand.vlm_client import OllamaClient

QUESTIONS = ["What happened near the metro?", "Were there any fights today?", "Summarise the incidents.",
             "Was anyone arrested?", "Any fire or smoke?", "What did the crowds do?", "Which camera was busiest?",
             "Anything that needs an officer?", "Describe the last event.", "Were there any vehicles involved?",
             "What happened at night?", "Give me a short briefing."]


REPEAT_PENALTY: float | None = None  # --repeat-penalty


def make_client(model: str, draft: int) -> OllamaClient:
    s = Settings()
    s.vlm.repeat_penalty = REPEAT_PENALTY
    s.vlm.model = model
    s.vlm.backend = "ollama"
    s.vlm.draft_num_predict = draft
    return OllamaClient(s.vlm), s


def tok_s(stats: dict) -> float | None:
    return stats["eval_count"] / stats["eval_duration"] * 1e9 if stats.get("eval_duration") else None


def run_vision(draft: int, index: list[dict], root: Path, model: str) -> list[dict]:
    c, s = make_client(model, draft)
    rows = []
    for it in index:
        grid = (root / it["file"]).read_bytes()
        summ, ms1 = c.summarize(grid, highlighted=False)
        st1 = dict(c.last_stats)
        label, _, ms2 = c.classify_text(summ.summary, summ.visible_evidence, s.categories) if summ else (None, None, 0)
        st2 = dict(c.last_stats)
        rows.append({"draft": draft, "file": it["file"], "summary": summ.summary if summ else None, "label": label,
                     "s1_s": ms1 / 1000, "s1_tok": st1.get("eval_count"), "s1_tps": tok_s(st1),
                     "s2_s": ms2 / 1000, "s2_tok": st2.get("eval_count"), "s2_tps": tok_s(st2)})
    return rows


def chat_prompts(summaries: list[str]) -> list[str]:
    rnd = random.Random(7)
    out = []
    for i, q in enumerate(QUESTIONS):
        picks = rnd.sample(summaries, min(10, len(summaries)))
        rows = [{"event_id": 100 + 10 * i + j, "start_ts": 1_760_600_000_000 + j * 60_000, "camera_id": f"cam{j}",
                 "category": "none", "status": "new", "summary": p} for j, p in enumerate(picks)]
        out.append(_stream_prompt(q, rows, None))
    return out


def run_chat(draft: int, prompts: list[str], model: str) -> list[dict]:
    c, _ = make_client(model, draft)
    rows = []
    for i, p in enumerate(prompts):
        t0 = time.perf_counter()
        text = "".join(c.stream_text([{"role": "user", "content": p}], max_tokens=256))
        rows.append({"draft": draft, "q": i, "text": text, "s": time.perf_counter() - t0,
                     "tok": c.last_stats.get("eval_count"), "tps": tok_s(c.last_stats)})
    return rows


def med(xs) -> float:
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else float("nan")


def table(vis: list[dict], chat: list[dict], drafts: list[int]) -> str:
    base_v = {r["file"]: r for r in vis if r["draft"] == drafts[0]}
    base_c = {r["q"]: r for r in chat if r["draft"] == drafts[0]}
    out = ["| draft tokens | stage 1 s | stage 1 tok/s | stage 2 s | stage 2 tok/s | chat s | chat tok/s | "
           "window total s | same summary | same label | same chat answer |", "|" + "---|" * 11]
    for d in drafts:
        V = [r for r in vis if r["draft"] == d]
        C = [r for r in chat if r["draft"] == d]
        same_s = sum(r["summary"] == base_v[r["file"]]["summary"] for r in V)
        same_l = sum(r["label"] == base_v[r["file"]]["label"] for r in V)
        same_c = sum(r["text"] == base_c[r["q"]]["text"] for r in C)
        out.append(f"| {d}{' (off)' if d == 0 else ''} | {med(r['s1_s'] for r in V):.3f} | {med(r['s1_tps'] for r in V):.0f} | "
                   f"{med(r['s2_s'] for r in V):.3f} | {med(r['s2_tps'] for r in V):.0f} | "
                   f"{med(r['s'] for r in C):.2f} | {med(r['tps'] for r in C):.0f} | "
                   f"{med(r['s1_s'] + r['s2_s'] for r in V):.3f} | {same_s}/{len(V)} | {same_l}/{len(V)} | "
                   f"{same_c}/{len(C)} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="data/bench_windows")
    ap.add_argument("--drafts", nargs="+", type=int, default=[0, 1, 2, 3, 4, 6])
    ap.add_argument("--model", default="gemma4:e2b")
    ap.add_argument("--repeat-penalty", type=float, default=None, help="Ollama repeat_penalty (default: the model's)")
    ap.add_argument("--rounds", type=int, default=2, help="passes over all settings (alternating order)")
    ap.add_argument("--limit", type=int, default=0, help="first N windows only (a quick check)")
    ap.add_argument("--out", default="data/reports/bench_spec.jsonl")
    args = ap.parse_args()
    global REPEAT_PENALTY
    REPEAT_PENALTY = args.repeat_penalty
    root = Path(args.windows)
    index = json.loads((root / "index.json").read_text())
    index = index[: args.limit] if args.limit else index
    make_client(args.model, 0)[0].summarize((root / index[0]["file"]).read_bytes(), highlighted=False)  # load model
    vis, chat, prompts = [], [], None
    for rnd in range(args.rounds):
        order = args.drafts if rnd % 2 == 0 else args.drafts[::-1]
        for d in order:
            t0 = time.perf_counter()
            v = run_vision(d, index, root, args.model)
            vis += [{**r, "round": rnd} for r in v]
            if prompts is None:
                prompts = chat_prompts([r["summary"] for r in v if r["summary"]])
            chat += [{**r, "round": rnd} for r in run_chat(d, prompts, args.model)]
            print(f"round {rnd} draft {d}: {time.perf_counter() - t0:.0f} s", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(json.dumps(r) for r in vis + chat), encoding="utf-8")
    # outputs are compared within round 0 (the same-input reference); times use every round
    print(table(vis, chat, args.drafts))
    r0v = [r for r in vis if r["round"] == 0]
    r0c = [r for r in chat if r["round"] == 0]
    print("\nround 0 only (output match vs draft 0 in the same round):")
    print(table(r0v, r0c, args.drafts))


if __name__ == "__main__":
    main()
