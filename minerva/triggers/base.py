"""Shared machinery for trigger rules: scene snapshot, track history, geometry, and Episodes,
the latch that turns "condition holds for N seconds" into exactly one firing."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Hashable, Iterable

from minerva.types import Box, Candidate, Category, PoseResult, TrackObs

Point = tuple[float, float]


@dataclass
class Scene:
    camera_id: str
    ts_ms: int
    width: int
    height: int
    persons: list[TrackObs]
    objects: list[TrackObs] = field(default_factory=list)
    motion_ratio: Callable[[Box], float] = lambda box: 0.0
    pose: Callable[[TrackObs], PoseResult | None] = lambda track: None
    violence: float | None = None  # calibrated clip-model score for the last ~2 s, if a model is loaded
    violence_threshold: float = 0.5
    hazards: list = field(default_factory=list)  # minerva.vision.hazards.Hazard, if a hazard model is loaded


@dataclass
class RuleOutput:
    candidates: list[Candidate] = field(default_factory=list)
    ongoing: list[tuple] = field(default_factory=list)  # keys of already-fired episodes still in progress


# ---------- geometry ----------

def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def union_box(a: Box, b: Box) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def denorm(points: Iterable[Point], w: int, h: int) -> list[Point]:
    return [(x * w, y * h) for x, y in points]


def point_in_polygon(pt: Point, poly: list[Point]) -> bool:
    x, y = pt
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def line_side(p1: Point, p2: Point, pt: Point) -> int:
    cross = (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])
    return 0 if abs(cross) < 1e-9 else (1 if cross > 0 else -1)


def segment_param(p1: Point, p2: Point, pt: Point) -> float:
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    denom = dx * dx + dy * dy
    return ((pt[0] - p1[0]) * dx + (pt[1] - p1[1]) * dy) / denom if denom else 0.0


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


# ---------- track history ----------

class TrackHistory:
    """Recent observations per track, for rules that need motion over time."""

    def __init__(self, keep_ms: int = 60_000, forget_ms: int = 20_000):
        self.keep_ms = keep_ms
        self.forget_ms = forget_ms
        self._h: dict[int, deque[tuple[int, TrackObs]]] = {}

    def update(self, ts: int, tracks: Iterable[TrackObs]) -> None:
        for t in tracks:
            d = self._h.setdefault(t.track_id, deque())
            d.append((ts, t))
            while d and ts - d[0][0] > self.keep_ms:
                d.popleft()
        for tid in [tid for tid, d in self._h.items() if ts - d[-1][0] > self.forget_ms]:
            del self._h[tid]

    def window(self, tid: int, ts: int, span_ms: int) -> list[tuple[int, TrackObs]]:
        return [(t, o) for t, o in self._h.get(tid, ()) if ts - t <= span_ms]

    def erratic(self, tid: int, ts: int, span_ms: int, min_step_frac: float = 0.05) -> bool:
        """True if the track reversed direction at least once in the window while moving."""
        obs = self.window(tid, ts, span_ms)
        vel = []
        for (_, a), (_, b) in zip(obs, obs[1:]):
            vx, vy = b.center[0] - a.center[0], b.center[1] - a.center[1]
            if math.hypot(vx, vy) >= min_step_frac * b.h:
                vel.append((vx, vy))
            else:
                vel.append(None)
        for u, v in zip(vel, vel[1:]):
            if u is not None and v is not None and u[0] * v[0] + u[1] * v[1] < 0:
                return True
        return False


# ---------- episodes ----------

@dataclass
class _Episode:
    onset: int
    last_true: int
    fired: bool = False


class Episodes:
    """Turns a per-frame boolean condition into a single firing once it has held for `sustain_ms`.
    Brief dropouts shorter than `release_ms` do not reset the episode."""

    def __init__(self, sustain_ms: float, release_ms: float):
        self.sustain_ms = sustain_ms
        self.release_ms = release_ms
        self._eps: dict[Hashable, _Episode] = {}

    def active(self, key: Hashable) -> bool:
        return key in self._eps

    def onset(self, key: Hashable) -> int:
        return self._eps[key].onset

    def step(
        self, ts: int, holding: Iterable[Hashable], ready: Callable[[Hashable], bool] | None = None
    ) -> tuple[list[Hashable], list[Hashable]]:
        fired, ongoing = [], []
        holding = set(holding)
        for key in holding:
            ep = self._eps.get(key)
            if ep is None:
                ep = self._eps[key] = _Episode(onset=ts, last_true=ts)
            ep.last_true = ts
            if ep.fired:
                ongoing.append(key)
            elif ts - ep.onset >= self.sustain_ms and (ready is None or ready(key)):
                ep.fired = True
                fired.append(key)
        for key in [k for k, ep in self._eps.items() if k not in holding and ts - ep.last_true > self.release_ms]:
            del self._eps[key]
        return fired, ongoing


class Rule:
    category: Category

    def evaluate(self, scene: Scene, hist: TrackHistory) -> RuleOutput:  # pragma: no cover - interface
        raise NotImplementedError
