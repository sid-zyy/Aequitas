"""`aequitas eval-classify`: stage-2 accuracy on labelled summaries (accuracy, macro-F1, confusion
matrix, latency). Used to compare quantization schemes on the same labels; the faculty dataset's
summaries/labels plug in through the same YAML format:

    categories: {name: definition, ...}   # optional; default = settings.categories
    items:
      - {summary: "...", label: fire}

Small hand-written sets are a development check only. Do not report them as accuracy on real footage."""

from __future__ import annotations

import json
import re
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import yaml

from aequitas.eval.report import percentile
from aequitas.taxonomy import NONE


def load_items(path: str | Path) -> tuple[list[dict], dict | None]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    items = [{"summary": str(i["summary"]), "label": str(i["label"])} for i in data.get("items", [])]
    if not items:
        raise ValueError(f"{path}: no items")
    return items, data.get("categories")


def metrics(gold: list[str], pred: list[str | None]) -> dict:
    labels = sorted(set(gold) | {p for p in pred if p})
    confusion = {g: {} for g in labels}
    for g, p in zip(gold, pred):
        key = p or "<invalid>"
        confusion[g][key] = confusion[g].get(key, 0) + 1
    f1s = {}
    for c in sorted(set(gold)):  # macro-F1 over the classes that occur in the gold labels
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s[c] = round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0
    n = len(gold)
    # incident vs none: what matters operationally is whether an incident is flagged at all
    inc_tp = sum(g != NONE and p not in (None, NONE) for g, p in zip(gold, pred))
    inc_fp = sum(g == NONE and p not in (None, NONE) for g, p in zip(gold, pred))
    inc_fn = sum(g != NONE and p in (None, NONE) for g, p in zip(gold, pred))
    return {
        "n": n,
        "accuracy": round(sum(g == p for g, p in zip(gold, pred)) / n, 4) if n else 0.0,
        "macro_f1": round(statistics.mean(f1s.values()), 4) if f1s else 0.0,
        "valid": sum(p is not None for p in pred),
        "per_class_f1": f1s,
        "incident_precision": round(inc_tp / (inc_tp + inc_fp), 4) if inc_tp + inc_fp else None,
        "incident_recall": round(inc_tp / (inc_tp + inc_fn), 4) if inc_tp + inc_fn else None,
        "confusion": confusion,
    }


def run_eval(client, items: list[dict], categories: dict | None, concurrency: int = 4, label: str = "",
             out_dir: Path = Path("data/reports")) -> dict:
    def one(item):
        return client.classify_text(item["summary"], "", categories)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        results = list(pool.map(one, items))
    gold = [i["label"] for i in items]
    pred = [r[0] for r in results]
    ms = [r[2] for r in results]
    confs = [r[1] for r in results if r[1] is not None]
    report = {
        "label": label, "model": client.model, "host": client.cfg.host, "run": datetime.now().isoformat(timespec="seconds"),
        **metrics(gold, pred),
        "latency_ms": {"p50": round(percentile(ms, .5), 1), "p95": round(percentile(ms, .95), 1),
                       "mean": round(statistics.mean(ms), 1)},
        "mean_confidence": round(statistics.mean(confs), 4) if confs else None,
        "errors": [{"summary": i["summary"], "gold": g, "pred": p}
                   for i, g, p in zip(items, gold, pred) if g != p],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"classify-{re.sub(r'[^A-Za-z0-9_.-]+', '_', label) + '-' if label else ''}{datetime.now():%Y%m%d-%H%M%S}"
    path = out_dir / f"{stem}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["_path"] = str(path)
    return report


def summary_line(r: dict) -> str:
    return (f"{r['label'] or r['model']}: accuracy {r['accuracy']:.3f}, macro-F1 {r['macro_f1']:.3f}, "
            f"valid {r['valid']}/{r['n']}, incident P/R {r['incident_precision']}/{r['incident_recall']}, "
            f"latency p50 {r['latency_ms']['p50']:.0f} ms. Report: {r['_path']}")
