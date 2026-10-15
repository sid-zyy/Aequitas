"""Before/after comparison of understanding variants on a FIXED set of saved windows.

    .venv/Scripts/python scripts/bench_variants.py --windows data/bench_windows --variants base guided context

The window set (2x2 grids exactly as the dashboard sends them, plus a gold 'kind' per window: fight, arrest,
fire, normal) is made once, so every variant sees the same frames. Prints per-variant:
  right      window label in the kind's accepted set (fight: fighting/weapon/harassment/riot)
  flagged    incident windows labelled anything but none (what reaches an officer as a candidate)
  false      normal windows labelled anything but none
  time       median summary + label seconds per window
Kinds are VLM-only here: the violence clip model is not used, so this measures the language stages alone.
"""

import argparse
import json
import statistics as st
import time
from pathlib import Path

from aequitas.config import Settings
from aequitas.understand.stages import understand
from aequitas.understand.vlm_client import make_client

ACCEPT = {"fight": {"fighting", "weapon", "harassment", "riot"}, "arrest": {"arrest"}, "fire": {"fire"},
          "normal": {"none"}}
VARIANTS = {"base": dict(guided=False, context=False), "guided": dict(guided=True, context=False),
            "context": dict(guided=False, context=True), "guided+context": dict(guided=True, context=True)}


def run(variant: str, index: list[dict], root: Path, model: str) -> list[dict]:
    v = VARIANTS[variant]
    s = Settings()
    s.vlm.model = model
    s.vlm.guided_checks = v["guided"]
    client = make_client(s.vlm)
    rows, prev = [], {}
    for it in index:
        grid = (root / it["file"]).read_bytes()
        previous = prev.get(it["clip"]) if v["context"] else None
        t0 = time.perf_counter()
        u = understand(client, client, grid, highlighted=False, categories=s.categories, previous=previous)
        rows.append({**it, "variant": variant, "label": u.answer.category if u.answer else None,
                     "s": round(time.perf_counter() - t0, 3), "summary": u.summary,
                     "evidence": u.answer.visible_evidence if u.answer else None})
        prev[it["clip"]] = u.summary
    return rows


def table(rows: list[dict]) -> str:
    out = ["| variant | right fight | right arrest | right fire | flagged incidents | false alarms (normal) | "
           "time / window (median) |", "|---|---|---|---|---|---|---|"]
    for variant in dict.fromkeys(r["variant"] for r in rows):
        R = [r for r in rows if r["variant"] == variant]
        cell = []
        for kind in ("fight", "arrest", "fire"):
            k = [r for r in R if r["kind"] == kind]
            cell.append(f"{sum(r['label'] in ACCEPT[kind] for r in k)}/{len(k)}")
        inc = [r for r in R if r["kind"] != "normal"]
        nor = [r for r in R if r["kind"] == "normal"]
        flagged = sum(r["label"] not in (None, "none") for r in inc)
        false = sum(r["label"] not in (None, "none") for r in nor)
        out.append(f"| {variant} | {' | '.join(cell)} | {flagged}/{len(inc)} | {false}/{len(nor)} | "
                   f"{st.median(r['s'] for r in R):.2f} s |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="data/bench_windows")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--model", default="gemma4:e2b")
    ap.add_argument("--out", default="data/reports/bench_variants.jsonl")
    args = ap.parse_args()
    root = Path(args.windows)
    index = json.loads((root / "index.json").read_text())
    rows = []
    for variant in args.variants:
        rows += run(variant, index, root, args.model)
        print(f"{variant}: done", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    print(table(rows))


if __name__ == "__main__":
    main()
