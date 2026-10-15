"""Person down: someone goes from upright to down and stays down, roughly in place.

"Down" is any of: torso near horizontal (pose), box wider than tall, or box height collapsed to a
fraction of the person's recent upright height. The last one is what works on high, angled CCTV,
where a person lying along the viewing direction still looks vertical in the image.

"Was upright" is checked per track, and also by place: trackers often switch ID during a fall, so a
new track lying where someone stood a moment ago inherits that upright reference."""

from __future__ import annotations

from collections import deque

from aequitas.config import PersonDownCfg
from aequitas.triggers.base import Episodes, Rule, RuleOutput, Scene, TrackHistory, clamp01, dist
from aequitas.types import Candidate, Category, TrackObs

Upright = tuple[int, tuple[float, float], float]  # (ts, feet, height)


class PersonDownRule(Rule):
    category = Category.PERSON_DOWN

    def __init__(self, cfg: PersonDownCfg, release_ms: float):
        self.cfg = cfg
        self.ep = Episodes(cfg.sustain_s * 1000, release_ms)
        self.window_ms = int(cfg.upright_window_s * 1000)
        self.last_upright: dict[int, Upright] = {}
        self.recent_upright: deque[Upright] = deque()  # all tracks, for ID-switch recovery
        self.ref_h: dict[int, float] = {}  # upright height each candidate is compared against
        self.anchor: dict[int, tuple[float, float]] = {}
        self.evidence: dict[tuple, tuple[float, str]] = {}

    def _upright_ref(self, p: TrackObs, ts: int) -> Upright | None:
        own = self.last_upright.get(p.track_id)
        if own and ts - own[0] <= self.window_ms:
            return own
        near = [u for u in self.recent_upright
                if ts - u[0] <= self.window_ms and dist(u[1], p.feet) <= self.cfg.spatial_match_heights * u[2]]
        return max(near, key=lambda u: u[0]) if near else None

    def _down(self, scene: Scene, p: TrackObs, ref_h: float | None) -> tuple[bool, float, str]:
        pose = scene.pose(p)
        if pose is not None and pose.conf >= self.cfg.min_pose_conf and pose.torso_angle_deg > self.cfg.torso_angle_deg:
            return True, pose.conf * clamp01((pose.torso_angle_deg - 30) / 45), "pose"
        if p.w / max(p.h, 1.0) >= self.cfg.lying_ratio:
            return True, 0.5 * p.conf, "aspect"
        if ref_h and p.h <= self.cfg.height_drop_frac * ref_h:
            return True, 0.5 * p.conf * clamp01(1.5 - p.h / ref_h), "height_drop"
        return False, 0.0, ""

    def evaluate(self, scene: Scene, hist: TrackHistory) -> RuleOutput:
        ts, cfg = scene.ts_ms, self.cfg
        while self.recent_upright and ts - self.recent_upright[0][0] > self.window_ms:
            self.recent_upright.popleft()
        holding = set()
        for p in scene.persons:
            tid = p.track_id
            key = ("person_down", tid)
            if p.h / max(p.w, 1.0) >= cfg.upright_ratio and not (
                tid in self.ref_h and p.h <= cfg.height_drop_frac * self.ref_h[tid]
            ):
                entry = (ts, p.feet, p.h)
                self.last_upright[tid] = entry
                self.recent_upright.append(entry)
                self.anchor.pop(tid, None)
                self.ref_h.pop(tid, None)
                continue
            if not self.ep.active(key):
                ref = self._upright_ref(p, ts)
                if ref is None:
                    continue  # never seen upright here: not a transition
                self.ref_h[tid] = ref[2]
            down, score, method = self._down(scene, p, self.ref_h.get(tid))
            if not down:
                self.anchor.pop(tid, None)
                continue
            anchor = self.anchor.setdefault(tid, p.center)
            if dist(anchor, p.center) > cfg.max_drift_frac * max(p.w, p.h):
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
        for tid in [t for t, u in self.last_upright.items() if t not in live and ts - u[0] > 60_000]:
            del self.last_upright[tid]
        for key in [k for k in self.evidence if not self.ep.active(k)]:
            del self.evidence[key]
            if key[1] not in live:
                self.ref_h.pop(key[1], None)
        return out
