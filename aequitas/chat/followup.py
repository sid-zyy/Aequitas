"""Conversation memory for the analyst chat. The client sends back the context of the previous answer (its
filter and the events it listed), and this module reads a new question against it:

- references: "#12", "event 12", "the second one", "it" -> event IDs from the previous answer
- follow-ups: "what about Koramangala?", "and yesterday?", "only confirmed", "show them" -> the previous filter
  with only the parts the new question names replaced

All rule-based: no model call, so a follow-up costs no extra latency."""

from __future__ import annotations

import re

from aequitas.chat.parser import wants_unconfirmed
from aequitas.store.query import QueryFilter
from aequitas.taxonomy import keyword_pattern
from aequitas.types import Status

_FOLLOW = re.compile(r"^\s*(and|also|what about|how about|only|just|now|same|but|then|show them|list them|"
                     r"which ones?|those|them|ok(ay)?,? (and|what about)|include)\b", re.I)
_PRONOUN = re.compile(r"\b(them|those|these|they|it|that one|this one)\b", re.I)
_ALL_PLACES = re.compile(r"\b(everywhere|all (areas|cameras|localities|places)|any ?where)\b", re.I)
_ANY_TIME = re.compile(r"\b(all time|ever|any time|anytime)\b", re.I)
_ALL_KINDS = re.compile(r"\b(all (events|incidents|kinds|types)|any (event|incident)s?|everything)\b", re.I)
_ONLY_CONFIRMED = re.compile(r"\bonly confirmed\b|\bjust confirmed\b", re.I)

_SIMILAR = re.compile(r"\b(similar|like (this|that|it|#?\d+)|more like|resembl\w*|same kind)\b", re.I)
_ORDINALS = {"first": 0, "1st": 0, "top": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "4th": 3,
             "fifth": 4, "5th": 4, "last": -1, "latest": 0}


def references(question: str, context: dict | None) -> list[int]:
    """Event IDs the question points at, explicitly (#12) or through the previous answer ("the second one")."""
    q = question.lower()
    ids = [int(a or b) for a, b in re.findall(r"#\s*(\d+)|\bevent\s+(?:number\s+|no\.?\s*)?(\d+)\b", q)]
    if ids:
        return list(dict.fromkeys(ids))
    prev = (context or {}).get("event_ids") or []
    if not prev:
        return []
    m = re.search(r"\b(first|1st|top|second|2nd|third|3rd|fourth|4th|fifth|5th|last|latest)\s+one\b", q)
    if m:
        i = _ORDINALS[m[1]]
        return [prev[i]] if -len(prev) <= i < len(prev) else []
    if len(prev) == 1 and re.search(r"\b(it|that one|this one|that event|this event|that incident)\b", q):
        return [prev[0]]
    return []


def wants_similar(question: str) -> bool:
    return bool(_SIMILAR.search(question))


def is_followup(question: str, context: dict | None) -> bool:
    """A question that only makes sense with the previous one: "and yesterday?", "what about MG Road?"."""
    if not (context or {}).get("filter"):
        return False
    q = question.strip()
    words = re.findall(r"\w+", q)
    return bool(_FOLLOW.match(q)) or (len(words) <= 6 and bool(_PRONOUN.search(q)))


def merge(prev: dict, new: QueryFilter, question: str, categories: dict[str, str]) -> tuple[QueryFilter, list[str]]:
    """The previous filter with whatever the follow-up names put in its place. Returns the filter and the parts
    carried over (shown to the officer: "kept from before: fighting, Indiranagar")."""
    old = QueryFilter.model_validate(prev)
    q = question.lower()
    kept: list[str] = []
    upd: dict = {}
    names_kind = any(re.search(keyword_pattern(c), q) for c in categories) or new.subtypes
    if _ALL_KINDS.search(q):
        upd["categories"], upd["subtypes"] = [], []
    elif names_kind:
        upd["categories"], upd["subtypes"] = new.categories, new.subtypes
    elif old.categories or old.subtypes:
        upd["categories"], upd["subtypes"] = old.categories, old.subtypes
        kept.append("incident type")
    if _ALL_PLACES.search(q):
        upd["camera_ids"] = []
    elif new.camera_ids:
        upd["camera_ids"] = new.camera_ids
    elif old.camera_ids:
        upd["camera_ids"] = old.camera_ids
        kept.append("place")
    if _ANY_TIME.search(q):
        upd["start"] = upd["end"] = None
    elif new.start or new.end:
        upd["start"], upd["end"] = new.start, new.end
    elif old.start or old.end:
        upd["start"], upd["end"] = old.start, old.end
        kept.append("time range")
    if _ONLY_CONFIRMED.search(q):
        upd["statuses"] = [Status.CONFIRMED]
    elif wants_unconfirmed(q) or re.search(r"\bunconfirmed\b", q):
        upd["statuses"] = list(Status)
    else:
        upd["statuses"] = old.statuses
    upd["text"] = new.text or old.text  # the caller sets new.text to the question's descriptive words only
    upd["intent"] = new.intent
    upd["sort"] = new.sort if new.sort != "newest" else old.sort
    return new.model_copy(update=upd), kept
