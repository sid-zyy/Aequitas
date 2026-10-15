"""Crowd incident, overcrowding subtype: too many people in the crowd zone for too long.
The dispersal subtype is a stretch goal and not implemented yet."""

from __future__ import annotations

from aequitas.config import CrowdCfg, CrowdZone
from aequitas.triggers.base import Episodes, Rule, RuleOutput, Scene, TrackHistory, clamp01, denorm, point_in_polygon
from aequitas.types import Candidate, Category


class OvercrowdingRule(Rule):
    category = Category.CROWD_INCIDENT

    def __init__(self, cfg: CrowdCfg, zone: CrowdZone, release_ms: float):
        self.cfg = cfg
        self.zone = zone
        self.ep = Episodes(cfg.sustain_s * 1000, release_ms)
        self.key = ("crowd", "overcrowding", zone.name)
        self.peak = 0
        self.tracks: set[int] = set()

    def evaluate(self, scene: Scene, hist: TrackHistory) -> RuleOutput:
        poly = denorm(self.zone.polygon, scene.width, scene.height)
        inside = [p.track_id for p in scene.persons if point_in_polygon(p.feet, poly)]
        crowded = len(inside) >= self.zone.max_people
        if crowded:
            self.peak = max(self.peak, len(inside))
            self.tracks.update(inside)
        fired, ongoing = self.ep.step(scene.ts_ms, [self.key] if crowded else [])
        out = RuleOutput(ongoing=ongoing)
        if fired:
            m = self.zone.max_people
            out.candidates.append(Candidate(
                scene.camera_id, self.category, "overcrowding", self.ep.onset(self.key), scene.ts_ms,
                round(clamp01((self.peak - m) / m + 0.5), 3), tuple(sorted(self.tracks)),
                zone=self.zone.name, key=self.key, detail={"peak_count": self.peak, "max_people": m},
            ))
        if not self.ep.active(self.key):
            self.peak, self.tracks = 0, set()
        return out
