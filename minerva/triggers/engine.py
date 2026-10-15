"""Runs every enabled rule for one camera and applies the cross-rule cooldown."""

from __future__ import annotations

from minerva.config import CameraCfg, TriggersCfg
from minerva.triggers.base import Rule, RuleOutput, Scene, TrackHistory
from minerva.triggers.crowd import OvercrowdingRule
from minerva.triggers.fighting import FightingRule
from minerva.triggers.person_down import PersonDownRule
from minerva.triggers.suspicious import AbandonedObjectRule, LoiteringRule
from minerva.triggers.tailgating import TailgatingRule


def build_rules(cfg: TriggersCfg, cam: CameraCfg) -> list[Rule]:
    release = cfg.release_s * 1000
    z = cam.zones
    rules: list[Rule] = []
    if cfg.person_down.enabled:
        rules.append(PersonDownRule(cfg.person_down, release))
    if cfg.fighting.enabled:
        rules.append(FightingRule(cfg.fighting, release))
    if cfg.tailgating.enabled and z.entry_line is not None:
        rules.append(TailgatingRule(cfg.tailgating, z.entry_line))
    if cfg.crowd.enabled and z.crowd_zone is not None:
        rules.append(OvercrowdingRule(cfg.crowd, z.crowd_zone, release))
    if cfg.loitering.enabled and z.watch_zones:
        rules.append(LoiteringRule(cfg.loitering, z.watch_zones, release))
    if cfg.abandoned_object.enabled:
        rules.append(AbandonedObjectRule(cfg.abandoned_object, release))
    return rules


class TriggerEngine:
    def __init__(self, cfg: TriggersCfg, cam: CameraCfg, rules: list[Rule] | None = None):
        self.cooldown_ms = int(cfg.cooldown_s * 1000)
        self.rules = rules if rules is not None else build_rules(cfg, cam)
        self.hist = TrackHistory()
        self._last_fired: dict[tuple, int] = {}

    def step(self, scene: Scene) -> RuleOutput:
        self.hist.update(scene.ts_ms, scene.persons + scene.objects)
        result = RuleOutput()
        for rule in self.rules:
            out = rule.evaluate(scene, self.hist)
            result.ongoing.extend(out.ongoing)
            for c in out.candidates:
                last = self._last_fired.get(c.key)
                if last is not None and c.trigger_ms - last < self.cooldown_ms:
                    result.ongoing.append(c.key)  # same thing again too soon: extend, don't re-report
                    continue
                self._last_fired[c.key] = c.trigger_ms
                result.candidates.append(c)
        cutoff = scene.ts_ms - self.cooldown_ms
        for key in [k for k, t in self._last_fired.items() if t < cutoff]:
            del self._last_fired[key]
        return result
