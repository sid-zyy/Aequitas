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


def fuse_window(ans: VlmAnswer | None, violence: float | None, threshold: float,
                alone: float | None = None) -> tuple[VlmAnswer | None, str]:
    """Periodic mode: the violence clip model as the second signal (design rule: never trust a VLM label
    alone). Returns (answer, source) with source = vlm | clip_model | both | vlm+clip.

    - both see violence (score >= threshold) -> unchanged, marked 'both'
    - the VLM named another incident (arrest, person_down...) -> kept: a struggle during an arrest scores
      high on the clip model, and the VLM's label is the more specific one
    - only the clip model sees violence -> fighting, but only at >= `alone` (stricter than the calibrated
      threshold: on busy street footage single windows reach 0.8-0.9, faculty fights reach 0.95-1.0)
    - only the VLM sees violence -> unchanged: a missed fight is worse than a false alarm a human dismisses"""
    alone = threshold if alone is None else max(alone, threshold)
    if violence is None:
        return ans, "vlm"
    if violence >= threshold and ans is not None and ans.category in VIOLENT:
        return ans, "both"
    if ans is not None and ans.category != "none":
        return ans, ("vlm+clip" if violence >= threshold else "vlm")
    if violence >= alone:
        summary = ans.summary if ans is not None else f"Violent motion detected by the clip model ({violence:.2f})."
        return VlmAnswer(category="fighting", summary=summary,
                         visible_evidence=(ans.visible_evidence if ans else ""), confidence=round(violence, 4)), \
            "clip_model"
    return ans, "vlm"
