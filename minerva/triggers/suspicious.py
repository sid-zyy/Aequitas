"""Suspicious activity, limited to two subtypes: loitering and abandoned objects."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from minerva.config import AbandonedCfg, LoiteringCfg
from minerva.triggers.base import (
    Episodes, Point, Rule, RuleOutput, Scene, TrackHistory, clamp01, denorm, dist, iou, point_in_polygon,
)
from minerva.types import Box, Candidate, Category


class LoiteringRule(Rule):
    category = Category.SUSPICIOUS_ACTIVITY

    def __init__(self, cfg: LoiteringCfg, zones: dict[str, list[Point]], release_ms: float):
        self.cfg = cfg
        self.zones = zones
        # Longer release: people briefly step out of a zone edge or get occluded.
        self.ep = Episodes(cfg.min_dwell_s * 1000, max(release_ms, 5000))
        self.evidence: dict[tuple, float] = {}

    def evaluate(self, scene: Scene, hist: TrackHistory) -> RuleOutput:
        ts, cfg = scene.ts_ms, self.cfg
        window = int(cfg.displacement_window_s * 1000)
        holding = set()
        for name, poly in self.zones.items():
            pix = denorm(poly, scene.width, scene.height)
            for p in scene.persons:
                if point_in_polygon(p.feet, pix):
                    holding.add(("loitering", name, p.track_id))

        def ready(key) -> bool:
            obs = hist.window(key[2], ts, window)
            if len(obs) < 2 or ts - obs[0][0] < 0.8 * window:
                return False
            last = obs[-1][1]
            disp = dist(obs[0][1].feet, last.feet)
            limit = cfg.max_displacement_heights * last.h
            if disp >= limit:
                return False
            self.evidence[key] = 0.5 + 0.5 * clamp01(1 - disp / limit)
            return True

        fired, ongoing = self.ep.step(ts, holding, ready)
        out = RuleOutput(ongoing=ongoing)
        for key in fired:
            out.candidates.append(Candidate(
                scene.camera_id, self.category, "loitering", self.ep.onset(key), ts,
                round(self.evidence.pop(key, 0.5), 3), (key[2],), zone=key[1], key=key,
                detail={"dwell_s": (ts - self.ep.onset(key)) / 1000},
            ))
        return out


@dataclass
class _ObjState:
    anchor: Box
    first_seen: int
    owner: int | None
    last_seen: int


class AbandonedObjectRule(Rule):
    category = Category.SUSPICIOUS_ACTIVITY
    WARMUP_MS = 10_000  # objects already present when the stream starts are treated as background

    def __init__(self, cfg: AbandonedCfg, release_ms: float):
        self.cfg = cfg
        # Bags flicker in and out of detection, so allow longer dropouts.
        self.ep = Episodes(cfg.unattended_s * 1000, max(release_ms, 5000))
        self.state: dict[int, _ObjState] = {}
        self.stream_start: int | None = None
        self.evidence: dict[tuple, float] = {}

    def evaluate(self, scene: Scene, hist: TrackHistory) -> RuleOutput:
        ts, cfg = scene.ts_ms, self.cfg
        if self.stream_start is None:
            self.stream_start = ts
        person_h = statistics.median(p.h for p in scene.persons) if scene.persons else None
        persons = {p.track_id: p for p in scene.persons}
        holding = set()
        for o in scene.objects:
            if o.label not in cfg.labels:
                continue
            radius = cfg.radius_person_heights * (person_h if person_h else 3 * o.h)

            def nearest_person() -> int | None:
                near = [(dist(p.center, o.center), tid) for tid, p in persons.items()]
                near = [n for n in near if n[0] <= radius]
                return min(near)[1] if near else None

            st = self.state.get(o.track_id)
            if st is None or iou(o.box, st.anchor) < cfg.stationary_iou:
                # New object, or it moved: whoever is nearest now is its owner.
                st = self.state[o.track_id] = _ObjState(o.box, ts, nearest_person(), ts)
            st.last_seen = ts

            if st.owner is None and st.first_seen - self.stream_start < self.WARMUP_MS:
                continue
            attended = any(dist(p.center, o.center) <= radius for p in persons.values())
            owner = persons.get(st.owner) if st.owner is not None else None
            owner_gone = owner is None or dist(owner.center, o.center) > radius
            if not attended and owner_gone:
                key = ("abandoned", o.track_id)
                holding.add(key)
                self.evidence[key] = o.conf

        fired, ongoing = self.ep.step(ts, holding)
        out = RuleOutput(ongoing=ongoing)
        for key in fired:
            st = self.state.get(key[1])
            track_ids = (key[1],) + ((st.owner,) if st and st.owner is not None else ())
            out.candidates.append(Candidate(
                scene.camera_id, self.category, "abandoned_object", self.ep.onset(key), ts,
                round(self.evidence.get(key, 0.5), 3), track_ids, key=key,
                detail={"owner_track": st.owner if st else None},
            ))
        for tid in [t for t, s in self.state.items() if ts - s.last_seen > 60_000]:
            del self.state[tid]
        for key in [k for k in self.evidence if not self.ep.active(k)]:
            del self.evidence[key]
        return out
