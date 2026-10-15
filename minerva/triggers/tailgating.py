"""Tailgating (video-only): two or more distinct people cross the entry line in the entering
direction within `max_gap_s` of each other. No badge data is used; see docs/DESIGN.md §2.3."""

from __future__ import annotations

from collections import deque

from minerva.config import EntryLine, TailgatingCfg
from minerva.triggers.base import Rule, RuleOutput, Scene, TrackHistory, denorm, line_side, segment_param
from minerva.types import Candidate, Category


class TailgatingRule(Rule):
    category = Category.TAILGATING

    def __init__(self, cfg: TailgatingCfg, line: EntryLine):
        self.cfg = cfg
        self.line = line
        self.enter_sign = 1 if line.entering == "positive" else -1
        self.gap_ms = int(cfg.max_gap_s * 1000)
        self.last_side: dict[int, int] = {}
        self.crossings: deque[tuple[int, int]] = deque()  # (ts, track_id)
        self.group: dict | None = None  # the currently open tailgating group

    def evaluate(self, scene: Scene, hist: TrackHistory) -> RuleOutput:
        ts = scene.ts_ms
        p1, p2 = denorm([self.line.p1, self.line.p2], scene.width, scene.height)
        out = RuleOutput()
        new_crossers = []
        for p in scene.persons:
            side = line_side(p1, p2, p.feet)
            prev = self.last_side.get(p.track_id)
            if side != 0:
                self.last_side[p.track_id] = side
            if prev is not None and side == self.enter_sign and prev == -self.enter_sign:
                if -0.05 <= segment_param(p1, p2, p.feet) <= 1.05:
                    new_crossers.append(p.track_id)

        while self.crossings and ts - self.crossings[0][0] > self.gap_ms:
            self.crossings.popleft()
        if self.group and ts - self.group["last"] > self.gap_ms:
            self.group = None

        for tid in new_crossers:
            others = [(t, i) for t, i in self.crossings if i != tid]
            self.crossings.append((ts, tid))
            if not others:
                continue
            if self.group is not None:
                self.group["tracks"].add(tid)
                self.group["last"] = ts
                out.ongoing.append(self.group["key"])
                continue
            nearest_t = max(t for t, _ in others)
            onset = min(t for t, _ in others)
            key = ("tailgating", onset)
            self.group = {"key": key, "tracks": {tid, *(i for _, i in others)}, "last": ts}
            score = 1 - (ts - nearest_t) / self.gap_ms if self.gap_ms else 1.0
            out.candidates.append(Candidate(
                scene.camera_id, self.category, None, onset, ts, round(max(0.05, score), 3),
                tuple(sorted(self.group["tracks"])), key=key,
                detail={"gap_ms": ts - nearest_t},
            ))

        live = {p.track_id for p in scene.persons}
        for tid in [t for t in self.last_side if t not in live]:
            del self.last_side[tid]
        return out
