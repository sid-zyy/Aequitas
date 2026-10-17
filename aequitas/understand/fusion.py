"""Combines the trigger's evidence with the VLM's answer into a status and a confidence.
Confidence comes from agreement between the two signals, not from the VLM's self-report."""

from __future__ import annotations

import re
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

# How many people Gemma's summary says are there: "Two men push...", "A group of people...", "Several people...".
_COUNT = re.compile(r"\b(two|three|four|five|six|seven|eight|nine|ten|\d+|several|many|a few|a group of|a crowd of)"
                    r"\s+(?:\w+\s+){0,2}?(people|persons|men|women|boys|girls|individuals|youths|students|"
                    r"officers|figures)\b", re.I)


def people_involved(summary: str | None) -> str | None:
    """'two men', 'a group of people', 'several people' from a summary, for the lead of a clip-model fight."""
    m = _COUNT.search(summary or "")
    if not m:
        return None
    return f"{m.group(1).lower()} {m.group(2).lower()}"


def fuse_window(ans: VlmAnswer | None, violence: float | None, threshold: float,
                alone: float | None = None) -> tuple[VlmAnswer | None, str]:
    """Periodic mode: the violence clip model as the second signal (CLAUDE.md: never trust a VLM label
    alone). Returns (answer, source) with source = vlm | clip_model | both | vlm+clip.

    - both see violence (score >= threshold) -> unchanged, marked 'both'
    - the VLM named another incident (arrest, person_down...) -> kept: a struggle during an arrest scores
      high on the clip model, and the VLM's label is the more specific one
    - only the clip model sees violence -> fighting, but only at >= `alone` (stricter than the calibrated
      threshold: on busy street footage single windows reach 0.8-0.9, faculty fights reach 0.95-1.0). The summary
      then starts "Possible fight among several people (violent motion 0.98)." so it explains the label, using the
      people count from Gemma's own summary
    - only the VLM sees violence -> unchanged: a missed fight is worse than a false alarm a human dismisses"""
    alone = threshold if alone is None else max(alone, threshold)
    if violence is None:
        return ans, "vlm"
    if violence >= threshold and ans is not None and ans.category in VIOLENT:
        return ans, "both"
    if ans is not None and ans.category != "none":
        return ans, ("vlm+clip" if violence >= threshold else "vlm")
    if violence >= alone:
        # The label comes from the clip model, so the summary says so first: the VLM's own description ("people
        # gathered") alone read as if it contradicted the label. The VLM's words are kept unchanged after it.
        people = people_involved(ans.summary) if ans is not None else None
        lead = (f"Possible fight among {people} (violent motion {violence:.2f})." if people
                else f"Possible fight (violent motion {violence:.2f}).")
        summary = f"{lead} {ans.summary.strip()}" if ans is not None and ans.summary.strip() else (
            f"Violent motion detected by the clip model ({violence:.2f}).")
        return VlmAnswer(category="fighting", summary=summary,
                         visible_evidence=(ans.visible_evidence if ans else ""), confidence=round(violence, 4)), \
            "clip_model"
    return ans, "vlm"
