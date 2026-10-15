"""Faculty dataset adapter and `aequitas eval-dataset`.

The format is not known yet, so the loader accepts the common shapes:
- a folder per label:           root/<label>/<video>
- a manifest (CSV, JSON, JSONL): one row per video with a path column and, optionally, a label and a
  reference summary. Column names are matched loosely (video/path/file/filename, label/category/class,
  summary/caption/description).

Labels are mapped onto our categories with a small YAML map ({dataset label: our category}); without a
map, the dataset's own labels are used as the category list (definitions = the label name, which can
be edited in settings). Then every video goes through the real stages (summarize -> classify) and the
report gives accuracy, macro-F1, summary similarity to the references, and time per video and in total."""

from __future__ import annotations

import csv
import json
import re
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from aequitas.config import Zones
from aequitas.eval.classify_eval import metrics
from aequitas.eval.report import percentile
from aequitas.ingest.ring_buffer import BufferedFrame
from aequitas.taxonomy import NONE
from aequitas.understand.keyframes import encode_jpeg, make_grid
from aequitas.understand.stages import understand

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}
PATH_COLS = ("video", "path", "file", "filename", "video_path", "clip", "video_id")
LABEL_COLS = ("label", "category", "class", "class_name", "labels", "event", "action")
SUMMARY_COLS = ("summary", "caption", "description", "reference", "text", "annotation")


@dataclass
class Item:
    path: Path
    label: str | None = None
    summary: str | None = None


def _pick(row: dict, names: tuple[str, ...]) -> str | None:
    low = {k.strip().lower(): v for k, v in row.items() if isinstance(k, str)}
    for n in names:
        if low.get(n) not in (None, ""):
            return str(low[n]).strip()
    return None


def _norm(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")


def load_dataset(root: str | Path, manifest: str | Path | None = None) -> list[Item]:
    root = Path(root)
    if manifest is None:  # a manifest next to the videos is used automatically
        found = [p for p in root.glob("*") if p.suffix.lower() in (".csv", ".json", ".jsonl")]
        manifest = found[0] if len(found) == 1 else None
    if manifest is not None:
        manifest = Path(manifest)
        text = manifest.read_text(encoding="utf-8-sig")
        if manifest.suffix.lower() == ".csv":
            rows = list(csv.DictReader(text.splitlines()))
        elif manifest.suffix.lower() == ".jsonl":
            rows = [json.loads(x) for x in text.splitlines() if x.strip()]
        else:
            data = json.loads(text)
            rows = data if isinstance(data, list) else next(v for v in data.values() if isinstance(v, list))
        items = []
        for r in rows:
            p = _pick(r, PATH_COLS)
            if not p:
                continue
            path = Path(p) if Path(p).is_absolute() else root / p
            if not path.suffix:  # IDs without an extension
                path = next((path.with_suffix(e) for e in VIDEO_EXTS if path.with_suffix(e).exists()), path)
            lab = _pick(r, LABEL_COLS)
            items.append(Item(path, lab, _pick(r, SUMMARY_COLS)))
        return items
    return [Item(p, p.parent.name if p.parent != root else None)
            for p in sorted(root.rglob("*")) if p.suffix.lower() in VIDEO_EXTS]


def map_labels(items: list[Item], label_map: dict[str, str] | None) -> tuple[list[Item], dict[str, str] | None]:
    """Applies {dataset label: our category}. Without a map, returns the dataset's labels as a category
    list (normalised names; 'normal', 'none', 'nonviolence'... become 'none')."""
    none_like = {"none", "normal", "no_event", "nonviolence", "non_violence", "negative", "background", "neutral"}
    if label_map:
        lm = {_norm(k): v for k, v in label_map.items()}
        for it in items:
            if it.label is not None:
                it.label = lm.get(_norm(it.label), NONE)
        return items, None
    cats = {}
    for it in items:
        if it.label is None:
            continue
        n = _norm(it.label)
        it.label = NONE if n in none_like else n
        if it.label != NONE:
            cats[it.label] = it.label.replace("_", " ")
    return items, cats or None


def clip_grid(path: Path, tile: int = 448, n: int = 4) -> bytes | None:
    """n frames spread over the whole clip, as one 2x2 key-frame grid (the stage-1 input)."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((k + 0.5) * total / n) if total else 0)
        ok, f = cap.read()
        if ok:
            frames.append(BufferedFrame(int(1000 * k * max(total, 1) / n / fps), f, 1.0, {}))
    cap.release()
    return encode_jpeg(make_grid(frames, None, Zones(), tile)) if frames else None


def rouge_l(a: str, b: str) -> float:
    x, y = a.lower().split(), b.lower().split()
    if not x or not y:
        return 0.0
    prev = [0] * (len(y) + 1)
    for xi in x:
        cur = [0]
        for j, yj in enumerate(y):
            cur.append(prev[j] + 1 if xi == yj else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    p, r = lcs / len(y), lcs / len(x)
    return round(2 * p * r / (p + r), 4) if p + r else 0.0


def run_dataset_eval(vlm, classifier, items: list[Item], categories: dict[str, str], tile: int = 448,
                     label: str = "", out_dir: Path = Path("data/reports"), on_progress=None) -> dict:
    """on_progress(done, total, row) is called after every video (progress logs on long runs)."""
    t_all = time.perf_counter()
    rows = []
    for it in items:
        if on_progress is not None and rows:
            on_progress(len(rows), len(items), rows[-1])
        t0 = time.perf_counter()
        grid = clip_grid(it.path, tile)
        if grid is None:
            rows.append({"video": str(it.path), "gold": it.label, "error": "unreadable video"})
            continue
        u = understand(vlm, classifier, grid, highlighted=False, categories=categories)
        rows.append({
            "video": str(it.path), "gold": it.label, "pred": u.answer.category if u.answer else None,
            "summary": u.summary, "reference": it.summary,
            "rougeL": rouge_l(u.summary, it.summary) if u.summary and it.summary else None,
            "summarize_s": round((u.summarize_ms or 0) / 1000, 3), "classify_s": round((u.classify_ms or 0) / 1000, 3),
            "video_total_s": round(time.perf_counter() - t0, 3),
        })
    if on_progress is not None and rows:
        on_progress(len(rows), len(items), rows[-1])
    total_s = time.perf_counter() - t_all
    scored = [r for r in rows if r.get("gold") is not None and "error" not in r]
    times = [r["video_total_s"] for r in rows if "video_total_s" in r]
    sims = [r["rougeL"] for r in rows if r.get("rougeL") is not None]
    report = {
        "label": label, "run": datetime.now().isoformat(timespec="seconds"), "n_videos": len(items),
        "total_execution_s": round(total_s, 2),
        "per_video_s": {"p50": round(percentile(times, .5), 3), "p95": round(percentile(times, .95), 3),
                        "mean": round(statistics.mean(times), 3)} if times else None,
        "classification": metrics([r["gold"] for r in scored], [r["pred"] for r in scored]) if scored else None,
        "summary_rougeL_mean": round(statistics.mean(sims), 4) if sims else None,
        "categories": categories, "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"dataset-{re.sub(r'[^A-Za-z0-9_.-]+', '_', label) + '-' if label else ''}{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["_path"] = str(path)
    return report


def summary_line(r: dict) -> str:
    c = r.get("classification") or {}
    parts = [f"{r['n_videos']} videos in {r['total_execution_s']} s total"]
    if r.get("per_video_s"):
        parts.append(f"per video p50 {r['per_video_s']['p50']} s")
    if c:
        parts.append(f"accuracy {c['accuracy']:.3f}, macro-F1 {c['macro_f1']:.3f}")
    if r.get("summary_rougeL_mean") is not None:
        parts.append(f"summary ROUGE-L {r['summary_rougeL_mean']:.3f}")
    return "; ".join(parts) + f". Report: {r['_path']}"


def load_label_map(path: str | Path | None) -> dict | None:
    if not path:
        return None
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or None
