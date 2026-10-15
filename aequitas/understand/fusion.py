"""Combines the trigger's evidence with the VLM's answer into a status and a confidence.
Confidence comes from agreement between the two signals, not from the VLM's self-report."""

from __future__ import annotations

from dataclasses import dataclass

from aequitas.config import FusionCfg
from aequitas.types import Candidate, Status
from aequitas.understand.prompts import VlmAnswer


@dataclass
class Fused:
    status: Status
    confidence: float
    vlm_category: str | None
    summary: str | None


def fuse(cand: Candidate, ans: VlmAnswer | None, cfg: FusionCfg) -> Fused:
    s = max(0.0, min(1.0, cand.score))
    if ans is None:  # VLM unavailable or gave invalid output twice
        return Fused(Status.DISPUTED, round(cfg.disputed_gain * s, 4), None, None)
    if ans.category == cand.category.value:
        conf = cfg.confirmed_base + cfg.confirmed_gain * s
        return Fused(Status.CONFIRMED, round(min(1.0, conf), 4), ans.category, ans.summary.strip())
    if ans.category == "none":
        return Fused(Status.REJECTED, round(cfg.rejected_gain * s, 4), ans.category, ans.summary.strip())
    return Fused(Status.DISPUTED, round(cfg.disputed_gain * s, 4), ans.category, ans.summary.strip())
