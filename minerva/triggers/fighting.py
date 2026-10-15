"""Fighting: two or more people close together with high, erratic motion; or, when a violence clip
model is loaded, a sustained high clip score with people in view. The clip path needs no tracking,
which is what failed on low-res CCTV (CAVIAR)."""

from __future__ import annotations

from itertools import combinations

from minerva.config import FightingCfg
from minerva.triggers.base import Episodes, Rule, RuleOutput, Scene, TrackHistory, clamp01, dist, iou, union_box
from minerva.types import Candidate, Category


class FightingRule(Rule):
    category = Category.FIGHTING

    def __init__(self, cfg: FightingCfg, release_ms: float):
        self.cfg = cfg
        self.ep = Episodes(cfg.sustain_s * 1000, release_ms)
        self.evidence: dict[tuple, float] = {}

    def evaluate(self, scene: Scene, hist: TrackHistory) -> RuleOutput:
        ts, cfg = scene.ts_ms, self.cfg
        window = int(cfg.erratic_window_s * 1000)
        holding = set()
        for a, b in combinations(scene.persons, 2):
            mean_h = (a.h + b.h) / 2
            if dist(a.center, b.center) >= cfg.proximity_frac * mean_h and iou(a.box, b.box) <= 0:
                continue
            ratio = scene.motion_ratio(union_box(a.box, b.box))
            if ratio < cfg.motion_ratio:
                continue
            erratic = sum(hist.erratic(t.track_id, ts, window) for t in (a, b))
            if erratic < cfg.min_erratic_tracks:
                continue
            key = ("fighting", frozenset((a.track_id, b.track_id)))
            holding.add(key)
            self.evidence[key] = max(self.evidence.get(key, 0.0), clamp01(ratio / 6))

        clip_key = ("fighting", "clip")
        if (cfg.use_clip_model and scene.violence is not None and scene.violence >= scene.violence_threshold
                and len(scene.persons) >= cfg.min_persons_for_clip):
            holding.add(clip_key)
            self.evidence[clip_key] = max(self.evidence.get(clip_key, 0.0), scene.violence)
            self.clip_tracks = tuple(sorted(p.track_id for p in scene.persons))

        fired, ongoing = self.ep.step(ts, holding)
        out = RuleOutput(ongoing=ongoing)
        for key in fired:
            tracks = getattr(self, "clip_tracks", ()) if key == clip_key else tuple(sorted(key[1]))
            out.candidates.append(Candidate(
                scene.camera_id, self.category, None, self.ep.onset(key), ts,
                round(self.evidence.get(key, 0.5), 3), tracks, key=key,
                detail={"source": "clip_model" if key == clip_key else "tracks"},
            ))
        for key in [k for k in self.evidence if not self.ep.active(k)]:
            del self.evidence[key]
        return out
