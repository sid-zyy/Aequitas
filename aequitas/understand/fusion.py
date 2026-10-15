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


VIOLENT = {"fighting", "riot", "weapon", "harassment"}


def fuse_window(ans: VlmAnswer | None, violence: float | None, threshold: float) -> tuple[VlmAnswer | None, str]:
    """Periodic mode: the violence clip model as the second signal (CLAUDE.md: never trust a VLM label
    alone). Returns (answer, source) with source = vlm | clip_model | both.

    - clip model confident (>= threshold) and the VLM did not see violence -> fighting, at the model's score
    - both see violence -> unchanged, marked 'both'
    - only the VLM sees violence -> unchanged: a missed fight is worse than a false alarm a human dismisses"""
    if violence is None:
        return ans, "vlm"
    if violence >= threshold:
        if ans is not None and ans.category in VIOLENT:
            return ans, "both"
        summary = ans.summary if ans is not None else f"Violent motion detected by the clip model ({violence:.2f})."
        return VlmAnswer(category="fighting", summary=summary,
                         visible_evidence=(ans.visible_evidence if ans else ""), confidence=round(violence, 4)), \
            "clip_model"
    return ans, "vlm"
