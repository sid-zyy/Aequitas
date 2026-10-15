"""Clip-level evaluation: compares recorded events per camera (one camera per clip) with expected labels.

Scored twice: on everything the rules fired (trigger-only) and on events the VLM confirmed, which shows
what the VLM check adds."""

from __future__ import annotations

from pathlib import Path

import yaml

from aequitas.store.db import EventStore
from aequitas.store.query import QueryFilter
from aequitas.types import Status


def _label(ev: dict) -> str:
    return f"{ev['category']}/{ev['subtype']}" if ev.get("subtype") else ev["category"]


def _matches(label: str, wanted: str) -> bool:
    # "suspicious_activity" matches any subtype; "suspicious_activity/loitering" only that subtype.
    return label == wanted or label.split("/")[0] == wanted


def score(found: set[str], required: list[str], optional: list[str]) -> tuple[int, int, int, list[str], list[str]]:
    tp = sum(any(_matches(f, r) for f in found) for r in required)
    missed = [r for r in required if not any(_matches(f, r) for f in found)]
    false_alarms = sorted(f for f in found if not any(_matches(f, w) for w in required + optional))
    return tp, len(false_alarms), len(missed), missed, false_alarms


def clip_report(store: EventStore, expected_path: str | Path) -> str:
    expected = yaml.safe_load(Path(expected_path).read_text(encoding="utf-8")) or {}
    lines = []
    totals = {"trigger": [0, 0, 0], "confirmed": [0, 0, 0]}
    for clip, spec in expected.items():
        spec = spec or {}
        req, opt = spec.get("required", []), spec.get("optional", [])
        evs = store.query(QueryFilter(camera_ids=[clip], statuses=list(Status), limit=100, sort="oldest"))
        trig = {_label(e) for e in evs}
        conf = {_label(e) for e in evs if e["status"] == Status.CONFIRMED.value}
        lines.append(f"\n## {clip}  (expected: {', '.join(req) or 'nothing'}"
                     f"{'; optional: ' + ', '.join(opt) if opt else ''})")
        if not evs:
            lines.append("  no events")
        for e in evs:
            t = (e["start_ts"] - evs[0]["start_ts"]) / 1000
            lines.append(f"  #{e['event_id']} +{t:5.1f}s {_label(e):38} {e['status']:10} conf={e['confidence']:.2f} "
                         f"rule={e['trigger_score']:.2f} vlm={e['vlm_category']}  | {(e['summary'] or '')[:110]}")
        for name, found in (("trigger", trig), ("confirmed", conf)):
            tp, fp, fn, missed, fas = score(found, req, opt)
            for i, v in enumerate((tp, fp, fn)):
                totals[name][i] += v
            lines.append(f"  [{name:9}] hits {tp}/{len(req)}"
                         + (f"  missed: {', '.join(missed)}" if missed else "")
                         + (f"  false alarms: {', '.join(fas)}" if fas else ""))
    head = ["# Clip-level results", "", f"{'view':10} {'hits':>5} {'false alarms':>13} {'misses':>7} {'precision':>10} {'recall':>7}"]
    for name, (tp, fp, fn) in totals.items():
        p = tp / (tp + fp) if tp + fp else float("nan")
        r = tp / (tp + fn) if tp + fn else float("nan")
        head.append(f"{name:10} {tp:>5} {fp:>13} {fn:>7} {p:>10.2f} {r:>7.2f}")
    return "\n".join(head + lines)
