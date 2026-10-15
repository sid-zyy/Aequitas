"""Prompts and forced output schemas for the two understanding stages (docs/HACKATHON_PLAN.md):

1. summarize: the VLM describes what is visible. It is never shown the category list, so the summary
   is not steered towards a label.
2. classify: a text-only call labels the summary, constrained to the category list plus "none".

The prompts never say which rule fired, so the answer is an independent check rather than an echo."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, create_model

from aequitas.taxonomy import DEFAULT_CATEGORIES, choices, definition_list

CHOICES: list[str] = choices(DEFAULT_CATEGORIES)  # the default label set


class SummaryAnswer(BaseModel):
    """Stage 1 output."""

    summary: str = Field(max_length=500)
    visible_evidence: str = Field(default="", max_length=300)


@lru_cache(maxsize=16)
def label_model(labels: tuple[str, ...]) -> type[BaseModel]:
    """Stage 2 output schema for backends without choice-constrained decoding (Ollama): a JSON-schema
    enum over the configured labels."""
    return create_model("LabelAnswer", category=(Literal[labels], ...))


LabelAnswer = label_model(tuple(CHOICES))


class VlmAnswer(BaseModel):
    """Both stages combined. `confidence` is the classifier's label probability when the backend
    reports log-probabilities (vLLM), else None. The single-call legacy prompt also returns this."""

    category: str  # one of the configured labels, or "none"
    summary: str = Field(max_length=500)
    visible_evidence: str = Field(default="", max_length=300)
    confidence: float | None = None


_GRID = (
    "You are reviewing footage from a security camera.\n"
    "The image is a 2x2 grid of four frames from the same camera in time order: "
    "top-left, top-right, bottom-left, bottom-right. Each frame shows its time in seconds "
    "relative to the last frame. "
)
_HIGHLIGHT = (
    "Red boxes (labelled P = person, B = bag) mark the people or objects to focus on. "
    "An amber line or outline marks an entrance line or zone.\n\n"
)


def _category_list(categories: dict[str, str] | None = None) -> str:
    return definition_list(categories or DEFAULT_CATEGORIES)


def summarize_prompt(highlighted: bool = True) -> str:
    if highlighted:
        focus = _HIGHLIGHT + "Describe what the highlighted people or objects do.\n"
    else:
        focus = "\nDescribe what happens in the scene.\n"
    return (
        f"{_GRID}{focus}\n"
        "Rules:\n"
        "- Describe only what is visible. Do not guess identities, intentions or anything outside the frames.\n"
        "- Say how many people there are, their posture (standing, walking, running, sitting, lying on the "
        "ground) and any contact between them, and whether any bag or object is left alone.\n"
        "- summary: one or two short sentences (at most 35 words) describing what happens across the "
        "four frames.\n"
        "- visible_evidence: at most 8 words naming the most important thing you saw.\n"
        "Respond with JSON only."
    )


def classify_text_prompt(summary: str, evidence: str = "", categories: dict[str, str] | None = None) -> str:
    ev = f"\nKey evidence: {evidence.strip()}" if evidence.strip() else ""
    return (
        "A security camera clip was described as follows.\n"
        f"Description: {summary.strip()}{ev}\n\n"
        "Choose the ONE category that best matches the description:\n"
        f"{_category_list(categories)}\n"
        "Use only the description. If it does not clearly describe one of the categories, answer none.\n"
        "Answer with the category name only."
    )


def classify_prompt(highlighted: bool = True, categories: dict[str, str] | None = None) -> str:
    """Legacy single-call prompt (category + summary in one VLM call). Kept for `aequitas spike`
    comparisons; the pipeline uses summarize_prompt + classify_text_prompt."""
    if highlighted:
        focus = _HIGHLIGHT + "Choose the ONE category that best describes what the highlighted people or objects are doing:\n"
    else:
        focus = "\nChoose the ONE category that best describes what is happening in the scene:\n"
    return (
        f"{_GRID}{focus}"
        f"{_category_list(categories)}\n"
        "Rules:\n"
        "- Use only what is visible. Do not guess identities, intentions or anything outside the frames.\n"
        "- If you are unsure, answer none.\n"
        "- summary: one or two plain sentences describing what happens across the four frames.\n"
        "- visible_evidence: a short phrase naming what you saw that supports the category.\n"
        "Respond with JSON only."
    )
