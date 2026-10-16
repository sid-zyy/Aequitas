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
    checks: str = Field(default="", max_length=300)  # guided mode: the check questions answered yes


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
    "relative to the last frame. The same person usually appears in several frames: count people within one "
    "frame, not across frames. "
)
_HIGHLIGHT = (
    "Red boxes (labelled P = person, B = bag) mark the people or objects to focus on. "
    "An amber line or outline marks an entrance line or zone.\n\n"
)


def _category_list(categories: dict[str, str] | None = None) -> str:
    return definition_list(categories or DEFAULT_CATEGORIES)


# Guided checks (ASK-Hint, WACV 2026: fine-grained, grouped action questions instead of "describe the
# scene"). Grouped by family; worded without category names so the summary is not steered to a label.
GUIDED_CHECKS = (
    "Check questions, grouped (answer from the frames only):\n"
    "  Violence: Is anyone hitting, kicking, pushing, choking or attacking another person? "
    "Is anyone holding a knife, a gun or a raised stick?\n"
    "  Police: Are people in uniform holding someone's arms, handcuffing them or leading them away?\n"
    "  People in danger: Is anyone lying or falling on the ground? Is a group running, scattering or "
    "throwing things?\n"
    "  Hazards: Are there flames or thick smoke? Is a vehicle crashed, overturned or hitting a person?\n"
    "  Property: Is anyone snatching belongings or breaking things?\n"
)


def summarize_prompt(highlighted: bool = True, concise: bool = True, view: str = "cctv",
                     guided: bool = False) -> str:
    """concise=False is the original, longer wording, kept only to measure the time the short one saves.
    view="close": a camera close to the people (a laptop webcam in a live demo): describe gestures in detail."""
    if highlighted:
        focus = _HIGHLIGHT + "Describe what the highlighted people or objects do.\n"
    else:
        focus = "\nDescribe what happens in the scene.\n"
    if concise:
        length = ("- summary: one or two short sentences (at most 35 words) describing what the people do over "
                  "these seconds.\n"
                  "- visible_evidence: at most 8 words naming the most important thing you saw.\n")
    else:
        length = ("- summary: one or two plain sentences describing what the people do over these seconds.\n"
                  "- visible_evidence: a short phrase naming the most important thing you saw.\n")
    if guided:
        length += ("- checks: a few words for every check question answered yes (for example 'punching; "
                   "flames'); leave it empty if all are no.\n")
    close = ""
    if view == "close":
        close = ("- The camera is close to the people. For each person, describe what their hands and arms do "
                 "(raising, waving, pointing, a punching or slapping motion, pushing, grabbing), what they hold, "
                 "and posture changes (standing up, sitting down, falling).\n"
                 "- Never describe faces or try to say who anyone is.\n")
    return (
        f"{_GRID}{focus}\n"
        "Rules:\n"
        f"{close}"
        "- Describe only what is visible. Do not guess identities, intentions or anything outside the frames.\n"
        "- Say how many people there are, their posture (standing, walking, running, sitting, lying on the "
        "ground) and any contact between them, and whether any bag or object is left alone.\n"
        "- If anyone hits, kicks, pushes, grabs, chokes, restrains or attacks another person, say so plainly.\n"
        "- Name the main action with a specific verb (walking, running, punching, pushing, handcuffing, "
        "falling, crashing, burning). Never write vague phrases such as 'some action', 'interacting' or "
        "'engaged in an activity' or 'in various poses'.\n"
        "- Write as if describing a short video: never mention frames, the grid, tiles or time points.\n"
        "- When two people touch (grabbing, holding, grappling, wrestling, one on top of the other on the "
        "ground), say what one does to the other, for example 'one man grabs the other around the neck'.\n"
        "- Mention police or other uniforms, handcuffs, knives, guns or raised sticks, flames, smoke, and "
        "crashed or damaged vehicles whenever they are visible.\n"
        "- If anyone throws stones, bricks, bottles or other objects, or smashes property, say so plainly.\n"
        f"{GUIDED_CHECKS if guided else ''}"
        f"{length}"
        "Respond with JSON only."
    )


def _fire_rule(categories: dict[str, str] | None) -> str:
    """Fire beats the labels a burning scene gets confused with (seen on the fire-rescue clip: responders
    near flames were labelled arrest, fighting, accident). Names only categories that are configured."""
    names = list(categories or DEFAULT_CATEGORIES)
    if "fire" not in names:
        return ""
    rivals = [c for c in ("accident", "fighting", "riot", "vandalism", "arrest") if c in names]
    return ("If flames, a burning vehicle or thick smoke are described, choose fire"
            + (f" over {', '.join(rivals)}" if rivals else "") + ", unless someone is clearly attacked. ")


def classify_text_prompt(summary: str, evidence: str = "", categories: dict[str, str] | None = None,
                         previous: str | None = None) -> str:
    ev = f"\nKey evidence: {evidence.strip()}" if evidence.strip() else ""
    prev = ""
    if previous and previous.strip():
        prev = (f"\nThe 4 seconds before, on the same camera: {previous.strip()}\n"
                "Use that only to understand a sequence (for example someone being led away after being held); "
                "label what happens now.")
    return (
        "A security camera clip was described as follows.\n"
        f"Description: {summary.strip()}{ev}{prev}\n\n"
        "Choose the ONE category that best matches the description:\n"
        f"{_category_list(categories)}\n"
        "Use only the description. If it does not clearly describe one of the categories, answer none.\n"
        "If several fit, prefer the more serious one: a person attacked, held down or hurt by someone is "
        "fighting (or harassment), even if they fall; person_down is for someone who falls or lies down with "
        f"nobody attacking them. {_fire_rule(categories)}"
        "Sports, games, exercise and play are none.\n"
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
