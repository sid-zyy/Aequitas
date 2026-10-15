"""VLM prompt and its forced output schema. The prompt never says which rule fired, so the
VLM's answer is an independent check rather than an echo."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from minerva.types import CATEGORY_DEFINITIONS

VlmCategory = Literal[
    "person_down", "fighting", "tailgating", "crowd_incident", "suspicious_activity", "none"
]


class VlmAnswer(BaseModel):
    category: VlmCategory
    summary: str = Field(max_length=500)
    visible_evidence: str = Field(default="", max_length=300)


def classify_prompt() -> str:
    cats = "\n".join(f"- {c.value}: {d}" for c, d in CATEGORY_DEFINITIONS.items())
    return (
        "You are reviewing footage from a security camera.\n"
        "The image is a 2x2 grid of four frames from the same camera in time order: "
        "top-left, top-right, bottom-left, bottom-right. Each frame shows its time in seconds "
        "relative to the last frame. Red boxes (labelled P = person, B = bag) mark the people or "
        "objects to focus on. An amber line or outline marks an entrance line or zone.\n\n"
        "Choose the ONE category that best describes what the highlighted people or objects are doing:\n"
        f"{cats}\n"
        "- none: none of the above; ordinary activity\n\n"
        "Rules:\n"
        "- Use only what is visible. Do not guess identities, intentions or anything outside the frames.\n"
        "- If you are unsure, answer none.\n"
        "- summary: one or two plain sentences describing what happens across the four frames.\n"
        "- visible_evidence: a short phrase naming what you saw that supports the category.\n"
        "Respond with JSON only."
    )
