"""Person down: a track goes from upright to horizontal and stays down, roughly in place."""

from __future__ import annotations

from minerva.config import PersonDownCfg
from minerva.triggers.base import Episodes, Rule, RuleOutput, Scene, TrackHistory, clamp01, dist
from minerva.types import Candidate, Category, TrackObs


class PersonDownRule(Rule):
    category = Category.PERSON_DOWN

    def __init__(self, cfg: PersonDownCfg, release_ms: float):
        self.cfg = cfg
        self.ep = Episodes(cfg.sustain_s * 1000, release_ms)
        self.last_upright: dict[int, int] = {}
        self.anchor: dict[int, tuple[float, float]] = {}
        self.evidence: dict[tuple, tuple[float, str]] = {}

    def _horizontal(self, scene: Scene, p: TrackObs) -> tuple[bool, float, str]:
        pose = scene.pose(p)
        if pose is not None and pose.conf >= self.cfg.min_pose_conf:
            down = pose.torso_angle_deg > self.cfg.torso_angle_deg
            return down, pose.conf * clamp01((pose.torso_angle_deg - 30) / 45), "pose"
        # Pose unavailable or unsure: fall back to box shape, with a weaker score.
        down = p.w / max(p.h, 1.0) >= self.cfg.lying_ratio
        return down, 0.5 * p.conf, "aspect"

    def evaluate(self, scene: Scene, hist: TrackHistory) -> RuleOutput:
        ts, cfg = scene.ts_ms, self.cfg
        holding = set()
        for p in scene.persons:
            tid = p.track_id
            key = ("person_down", tid)
            if p.h / max(p.w, 1.0) >= cfg.upright_ratio:
                self.last_upright[tid] = ts
                self.anchor.pop(tid, None)
                continue
            # Only a transition counts: the person must have been upright recently,
            # unless an episode for them is already running.
            recently_upright = ts - self.last_upright.get(tid, -10**15) <= cfg.upright_window_s * 1000
            if not (recently_upright or self.ep.active(key)):
                continue
            down, score, method = self._horizontal(scene, p)
            if not down:
                self.anchor.pop(tid, None)
                continue
            anchor = self.anchor.setdefault(tid, p.center)
            if dist(anchor, p.center) > cfg.max_drift_frac * p.w:
                self.anchor[tid] = p.center  # moving along the ground (crawling, or a bad track): restart
                continue
            holding.add(key)
            self.evidence[key] = (score, method)

        fired, ongoing = self.ep.step(ts, holding)
        out = RuleOutput(ongoing=ongoing)
        for key in fired:
            score, method = self.evidence.get(key, (0.5, "aspect"))
            out.candidates.append(Candidate(
                scene.camera_id, self.category, None, self.ep.onset(key), ts, round(score, 3),
                (key[1],), key=key, detail={"method": method},
            ))
        live = {p.track_id for p in scene.persons}
        for tid in [t for t in self.last_upright if t not in live and ts - self.last_upright[t] > 60_000]:
            del self.last_upright[tid]
        for key in [k for k in self.evidence if not self.ep.active(k)]:
            del self.evidence[key]
        return out
