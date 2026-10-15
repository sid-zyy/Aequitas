"""Turns a natural-language question into a QueryFilter.

The LLM only fills a small JSON form (categories, camera mentions, time range...). Code then
resolves camera names, parses and sanity-checks times, and builds the SQL. If the LLM is not
available, a keyword parser handles the common phrasings."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, tzinfo
from typing import Literal

from pydantic import BaseModel, Field

from minerva.store.query import QueryFilter
from minerva.taxonomy import DEFAULT_CATEGORIES, keyword_pattern
from minerva.types import Status

log = logging.getLogger(__name__)

SubtypeName = Literal["loitering", "abandoned_object", "overcrowding"]


class LlmFilter(BaseModel):
    """What the LLM fills in. Deliberately close to the user's words; code does the resolving."""

    categories: list[str] = Field(default_factory=list)  # configured names; unknown ones are dropped
    subtypes: list[SubtypeName] = Field(default_factory=list)
    cameras: list[str] = Field(default_factory=list)
    start: str | None = None  # local time, YYYY-MM-DDTHH:MM
    end: str | None = None
    include_unconfirmed: bool = False
    text: str | None = None
    sort: Literal["newest", "oldest", "confidence"] = "newest"
    limit: int = 20
    intent: Literal["list", "count"] = "list"


@dataclass
class ParseResult:
    filter: QueryFilter
    warnings: list[str] = field(default_factory=list)
    method: str = "llm"


def _prompt(question: str, now: datetime, cameras: list[dict], categories: dict[str, str] | None = None) -> str:
    cams = "\n".join(
        f"- {c['camera_id']}: {c['name']}" + (f" (also called: {', '.join(c['aliases'])})" if c["aliases"] else "")
        for c in cameras
    ) or "- (no cameras registered)"
    cats = "\n".join(f"- {name}: {d}" for name, d in (categories or DEFAULT_CATEGORIES).items())
    return (
        "Convert the user's question about stored security-camera events into a JSON search filter.\n\n"
        f"Current local time: {now.strftime('%A %Y-%m-%d %H:%M')}\n\n"
        f"Cameras:\n{cams}\n\n"
        f"Event categories:\n{cats}\n"
        "Subtypes: loitering and abandoned_object (both suspicious_activity), overcrowding (crowd_incident).\n\n"
        "Rules:\n"
        "- Only fill fields the question asks about; leave the rest empty or default.\n"
        "- cameras: copy the camera words the user used (e.g. 'camera 2', 'entrance').\n"
        "- start/end: local time as YYYY-MM-DDTHH:MM. 'today' = 00:00 today to now. "
        "'yesterday' = all of yesterday. 'this morning' = 06:00-12:00 today, 'this afternoon' = 12:00-18:00, "
        "'this evening'/'tonight' = 18:00-23:59. 'last N hours' = now minus N hours to now.\n"
        "- intent: 'count' for 'how many' questions, else 'list'.\n"
        "- include_unconfirmed: true only if the user asks for unconfirmed, disputed or rejected events.\n"
        "- text: a few keywords ONLY when the question describes something no category covers "
        "(e.g. 'red jacket', 'unusual').\n\n"
        f"Question: {question}"
    )


def _parse_local(s: str | None, tz: tzinfo) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.strip().replace(" ", "T"))
    except ValueError:
        return None
    return dt.replace(tzinfo=tz) if dt.tzinfo is None else dt


def resolve_cameras(mentions: list[str], cameras: list[dict]) -> tuple[list[str], list[str]]:
    """Map the user's camera words to camera IDs. Returns (ids, unknown mentions)."""
    ids, unknown = [], []
    for m in mentions:
        q = m.strip().lower()
        if not q:
            continue
        hit = None
        for c in cameras:
            names = [c["camera_id"].lower(), c["name"].lower(), *(a.lower() for a in c["aliases"])]
            if q in names:
                hit = c["camera_id"]
                break
        if hit is None:
            num = re.search(r"(?:cam(?:era)?\s*#?\s*)(\d+)", q)
            if num:
                hit = next((c["camera_id"] for c in cameras if c["camera_id"].lower() in (f"cam{num[1]}", num[1])), None)
        if hit is None:
            hit = next((c["camera_id"] for c in cameras
                        if any(q in n or n in q for n in [c["name"].lower(), *(a.lower() for a in c["aliases"])])),
                       None)
        if hit is None:
            unknown.append(m)
        elif hit not in ids:
            ids.append(hit)
    return ids, unknown


def finalize(lf: LlmFilter, now: datetime, cameras: list[dict],
             categories: dict[str, str] | None = None) -> ParseResult:
    tz = now.tzinfo
    warnings = []
    cam_ids, unknown = resolve_cameras(lf.cameras, cameras)
    if unknown:
        warnings.append(f"I don't know a camera called {', '.join(repr(u) for u in unknown)}.")
    start, end = _parse_local(lf.start, tz), _parse_local(lf.end, tz)
    if start is not None and end is not None and start > end:
        start, end = end, start
    if end is not None and end > now:
        end = now
    if start is not None and start > now:
        warnings.append("That time range is in the future, so it was ignored.")
        start, end = None, None
    statuses = list(Status) if lf.include_unconfirmed else [Status.CONFIRMED]
    f = QueryFilter(
        categories=[c for c in lf.categories if c in (categories or DEFAULT_CATEGORIES)],
        subtypes=list(lf.subtypes),
        camera_ids=cam_ids,
        start=start,
        end=end,
        statuses=statuses,
        text=lf.text or None,
        sort=lf.sort,
        limit=max(1, min(100, lf.limit)),
        intent=lf.intent,
    )
    return ParseResult(f, warnings)


# ---------- keyword fallback ----------

_SUBTYPE_WORDS = {
    "loitering": r"\bloiter\w*",
    "abandoned_object": r"\b(abandon\w*|unattended|left behind|bags?|luggage)\b",
    "overcrowding": r"\bovercrowd\w*",
}


def wants_unconfirmed(question: str) -> bool:
    return bool(re.search(r"\b(unconfirmed|unverified|disputed|rejected|all statuses)\b", question.lower()))


def keyword_filter(question: str, now: datetime, cameras: list[dict],
                   categories: dict[str, str] | None = None) -> LlmFilter:
    q = question.lower()
    cats = [c for c in (categories or DEFAULT_CATEGORIES) if re.search(keyword_pattern(c), q)]
    subs = [s for s, rx in _SUBTYPE_WORDS.items() if re.search(rx, q)]

    mentions = re.findall(r"\bcam(?:era)?\s*#?\s*\d+\b", q)
    for c in cameras:
        for n in [c["name"], *c["aliases"]]:
            if re.search(rf"\b{re.escape(n.lower())}\b", q):
                mentions.append(n)

    fmt = "%Y-%m-%dT%H:%M"
    day0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start = end = None
    if m := re.search(r"\b(?:last|past)\s+(\d+)\s*(hour|hr|minute|min)s?\b", q):
        delta = timedelta(hours=int(m[1])) if m[2].startswith("h") else timedelta(minutes=int(m[1]))
        start, end = now - delta, now
    elif re.search(r"\b(?:last|past) hour\b", q):
        start, end = now - timedelta(hours=1), now
    elif "yesterday" in q:
        start, end = day0 - timedelta(days=1), day0 - timedelta(minutes=1)
    elif "this morning" in q:
        start, end = day0.replace(hour=6), day0.replace(hour=12)
    elif "this afternoon" in q:
        start, end = day0.replace(hour=12), day0.replace(hour=18)
    elif re.search(r"\b(this evening|tonight)\b", q):
        start, end = day0.replace(hour=18), day0.replace(hour=23, minute=59)
    elif "this week" in q:
        start, end = day0 - timedelta(days=now.weekday()), now
    elif "today" in q:
        start, end = day0, now

    text = None
    if not cats and not mentions:
        words = [w for w in re.findall(r"[a-z]{4,}", q) if w not in {
            "show", "list", "find", "events", "event", "anything", "were", "there", "today", "what", "which",
            "happened", "that", "with", "from", "this", "morning", "yesterday", "camera", "cameras"}]
        text = " ".join(words) or None

    return LlmFilter(
        categories=cats, subtypes=subs, cameras=mentions,
        start=start.strftime(fmt) if start else None, end=end.strftime(fmt) if end else None,
        include_unconfirmed=wants_unconfirmed(q),
        text=text,
        intent="count" if re.search(r"\b(how many|count|number of)\b", q) else "list",
        sort="oldest" if re.search(r"\b(first|earliest|oldest)\b", q) else "newest",
    )


def reconcile(lf: LlmFilter, kw: LlmFilter, question: str) -> LlmFilter:
    """Check a small LLM's filter against what the question literally says. Small models invent
    cameras and drop categories; the keyword parse is a reliable floor for both."""
    q = question.lower()
    cams = [c for c in lf.cameras if c.strip() and c.lower() in q]  # only cameras the user actually named
    for c in kw.cameras:
        if c not in cams:
            cams.append(c)
    return lf.model_copy(update={
        "cameras": cams,
        "categories": lf.categories or kw.categories,
        "subtypes": lf.subtypes or kw.subtypes,
        "start": lf.start or kw.start,
        "end": lf.end or kw.end,
        "intent": "count" if kw.intent == "count" else lf.intent,
        "include_unconfirmed": lf.include_unconfirmed or kw.include_unconfirmed,
    })


class QuestionParser:
    def __init__(self, llm=None, tz: tzinfo | None = None, categories: dict[str, str] | None = None):
        self.llm = llm  # a VlmClient, or None for keyword-only
        self.tz = tz
        self.categories = categories or DEFAULT_CATEGORIES

    def now(self) -> datetime:
        return datetime.now(self.tz) if self.tz else datetime.now().astimezone()

    def parse(self, question: str, cameras: list[dict], now: datetime | None = None) -> ParseResult:
        now = now or self.now()
        if self.llm is not None:
            prompt = _prompt(question, now, cameras, self.categories)
            lf = self.llm.structured([{"role": "user", "content": prompt}], LlmFilter)
            if lf is not None:
                lf = reconcile(lf, keyword_filter(question, now, cameras, self.categories), question)
                res = finalize(lf, now, cameras, self.categories)
                res.method = "llm"
                return res
            log.warning("LLM parse failed, using keyword parser")
        res = finalize(keyword_filter(question, now, cameras, self.categories), now, cameras, self.categories)
        res.method = "keywords"
        return res
