"""`aequitas eval-chat`: how well the chatbot turns questions into search filters.

Each item in config/chat_eval.yaml has a question and the filter it should give. The real parser runs (the LLM
fills the form, the keyword parse is its floor, code resolves cameras and times), and every field is scored:
categories, cameras, time range (+-1 minute), intent and include_unconfirmed. The keyword parser alone is the
baseline. Used to check that a fine-tuned model did not get worse at the chatbot's job."""

from __future__ import annotations

import json
import re
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from aequitas.chat.parser import QuestionParser
from aequitas.eval.report import percentile

FIELDS = ("categories", "cameras", "time", "intent", "unconfirmed")


def load(path: str | Path) -> tuple[list[dict], list[dict], datetime]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    now = datetime.fromisoformat(data["now"]).astimezone()
    cams = [{"camera_id": c["camera_id"], "name": c["name"], "aliases": c.get("aliases", [])} for c in data["cameras"]]
    return data["items"], cams, now


def _t(s: str | None, tz) -> datetime | None:
    return datetime.fromisoformat(s).replace(tzinfo=tz) if s else None


def _close(a: datetime | None, b: datetime | None) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) <= timedelta(minutes=1)


def score(item: dict, f, now: datetime) -> dict[str, bool]:
    """Which fields of the parsed QueryFilter `f` match the expected ones."""
    tz = now.tzinfo
    return {
        "categories": set(f.categories) == set(item.get("categories", [])),
        "cameras": set(f.camera_ids) == set(item.get("cameras", [])),
        "time": _close(f.start, _t(item.get("start"), tz)) and _close(f.end, _t(item.get("end"), tz)),
        "intent": f.intent == item.get("intent", "list"),
        "unconfirmed": (len(f.statuses) > 1) == bool(item.get("include_unconfirmed", False)),
    }


def run(parser: QuestionParser, items: list[dict], cams: list[dict], now: datetime) -> dict:
    rows, ms = [], []
    for it in items:
        t0 = time.perf_counter()
        res = parser.parse(it["q"], cams, now=now)
        ms.append((time.perf_counter() - t0) * 1000)
        s = score(it, res.filter, now)
        rows.append({"q": it["q"], "method": res.method, **s, "all": all(s.values()),
                     "got": {"categories": res.filter.categories, "cameras": res.filter.camera_ids,
                             "start": res.filter.start.isoformat() if res.filter.start else None,
                             "end": res.filter.end.isoformat() if res.filter.end else None,
                             "intent": res.filter.intent}})
    n = len(rows)
    return {"n": n, "all_correct": sum(r["all"] for r in rows),
            "per_field": {k: sum(r[k] for r in rows) for k in FIELDS},
            "llm_answered": sum(r["method"] == "llm" for r in rows),
            "latency_ms": {"p50": round(percentile(ms, .5), 1), "mean": round(statistics.mean(ms), 1)},
            "rows": rows}


def report_table(results: dict[str, dict]) -> str:
    lines = ["| parser | all fields right | " + " | ".join(FIELDS) + " | LLM answered | time p50 |",
             "|---|---|" + "---|" * len(FIELDS) + "---|---|"]
    for name, r in results.items():
        cells = " | ".join(f"{r['per_field'][k]}/{r['n']}" for k in FIELDS)
        lines.append(f"| {name} | {r['all_correct']}/{r['n']} | {cells} | {r['llm_answered']}/{r['n']} | "
                     f"{r['latency_ms']['p50'] / 1000:.2f} s |")
    return "\n".join(lines)


def save(results: dict[str, dict], label: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"chat-{re.sub(r'[^A-Za-z0-9_.-]+', '_', label) + '-' if label else ''}{datetime.now():%Y%m%d-%H%M%S}"
    path = out_dir / f"{stem}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path
