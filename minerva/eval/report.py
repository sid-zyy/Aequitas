"""Latency summary from the event_timings table (per-stage p50/p95)."""

from __future__ import annotations

import statistics

from minerva.store.db import EventStore

STAGES = ["detect_ms", "trigger_ms", "queue_ms", "vlm_ms", "db_ms", "proc_ms", "e2e_ms"]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def latency_table(store: EventStore) -> str:
    rows = store.timings()
    if not rows:
        return "No events recorded yet."
    lines = [f"{len(rows)} events", f"{'stage':<12}{'p50 ms':>10}{'p95 ms':>10}{'mean ms':>10}"]
    for s in STAGES:
        vals = [r[s] for r in rows if r[s] is not None]
        if vals:
            lines.append(f"{s:<12}{percentile(vals, .5):>10.0f}{percentile(vals, .95):>10.0f}{statistics.mean(vals):>10.0f}")
    return "\n".join(lines)
