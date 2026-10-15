"""The event categories stage 2 classifies summaries into. Configurable (`categories:` in settings), so
the faculty dataset's labels can replace them without code changes. The default is the police list.

Definitions are written for the guardrails (docs/POLICE_EDITION.md §8): they describe actions, never
who people are, and a peaceful gathering is never an incident."""

from __future__ import annotations

import re

NONE = "none"

# Order matters a little: it is the order shown to the model and in the UI.
DEFAULT_CATEGORIES: dict[str, str] = {
    "harassment": ("a person follows, blocks, touches, gropes, grabs or threatens another person who tries to "
                   "move away or resist (stalking, eve-teasing, molestation)"),
    "fighting": "two or more people hit, kick, push or grapple with each other (assault, brawl)",
    "riot": "a group acts violently: throwing stones or objects, damaging property, or clashing with police",
    "weapon": "a person holds or uses a weapon such as a knife, gun, sword, or a stick or rod raised to strike",
    "fire": ("uncontrolled fire, flames or thick smoke, such as a burning vehicle, shop or building, also while "
             "firefighters or police work at it (not a stove, lamp, candle or cooking flame)"),
    "arrest": ("police or officers (people in uniform) restrain, handcuff, hold the arms of, detain or escort a "
               "person who is being held. Officers who only stand, talk, work or direct traffic are not an arrest"),
    "chase": "one person or vehicle runs after or pursues another at speed",
    "person_down": "a person has fallen or is lying on the ground and stays down",
    "accident": "a road accident: vehicles collide, or a vehicle hits a person or overturns",
    "theft": "a person snatches, grabs or takes someone's belongings and moves away quickly",
    "vandalism": "a person deliberately breaks, damages or defaces property without a crowd clash",
    "crowd_incident": "an area is dangerously overcrowded, or a crowd suddenly runs or scatters (stampede)",
    "tailgating": "two or more people pass through an entrance one right behind the other",
    "suspicious_activity": "a person lingers in one place for a long time, or a bag is left unattended",
}

NONE_DEFINITION = ("none of the above; ordinary activity, including normal traffic (vehicles moving, waiting or "
                   "parked) and police or rescue workers doing their job with nobody held or hurt. A peaceful "
                   "gathering, protest, procession, queue, "
                   "festival or market crowd is none")

_NAME = re.compile(r"^[a-z][a-z0-9_]{1,39}$")


def validate(categories: dict[str, str]) -> dict[str, str]:
    if not categories:
        raise ValueError("at least one category is required")
    for name in categories:
        if not _NAME.match(name) or name == NONE:
            raise ValueError(f"category name {name!r}: use lowercase letters, digits and _, and not 'none'")
    return categories


def choices(categories: dict[str, str]) -> list[str]:
    """Labels the classifier may output: every category plus 'none'."""
    return [*categories, NONE]


def label(name: str) -> str:
    """Human-readable label: 'person_down' -> 'person down'."""
    return name.replace("_", " ")


def definition_list(categories: dict[str, str]) -> str:
    lines = [f"- {name}: {d}" for name, d in categories.items()]
    return "\n".join([*lines, f"- {NONE}: {NONE_DEFINITION}"]) + "\n"


# Keyword fallback for the chatbot when no LLM is available. Categories without an entry are matched
# by their own name.
KEYWORDS: dict[str, str] = {
    "person_down": r"\b(fall|fell|fallen|falling|person down|man down|lying|collaps\w*)\b",
    "fighting": r"\b(fight\w*|brawl\w*|assault\w*|punch\w*|beat\w*|violen\w*)\b",
    "tailgating": r"\b(tailgat\w*|piggyback\w*)\b",
    "crowd_incident": r"\b(crowd\w*|overcrowd\w*|stampede|dispers\w*)\b",
    "suspicious_activity": r"\b(suspicious|loiter\w*|abandon\w*|unattended|left behind|bags?|luggage)\b",
    "riot": r"\b(riot\w*|stone[- ]?pelt\w*|pelting|mob|unrest|clash\w*)\b",
    "weapon": r"\b(weapons?|knife|knives|guns?|pistol|firearm|sword|lathi|armed)\b",
    "fire": r"\b(fires?|flames?|smoke|burning|blaze|arson)\b",
    "arrest": r"\b(arrest\w*|detain\w*|handcuff\w*|custody|apprehend\w*)\b",
    "chase": r"\b(chas\w*|pursu\w*|running after|fleeing|fled)\b",
    "accident": r"\b(accidents?|crash\w*|collision|collided|hit[- ]and[- ]run)\b",
    "theft": r"\b(thefts?|thie\w*|snatch\w*|stole|stolen|steal\w*|robbery|rob\w*)\b",
    "harassment": r"\b(harass\w*|stalk\w*|eve[- ]?teas\w*|molest\w*|grop\w*|catcall\w*|follow(ed|ing) (her|him|a woman))\b",
    "vandalism": r"\b(vandal\w*|smash\w*|graffiti|damag\w*)\b",
}


def keyword_pattern(name: str) -> str:
    return KEYWORDS.get(name, rf"\b{re.escape(label(name))}\b")
