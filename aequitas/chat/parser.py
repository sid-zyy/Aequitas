"""Turns a natural-language question into a QueryFilter.

The LLM only fills a small JSON form (categories, camera mentions, time range...). Code then
resolves camera names, parses and sanity-checks times, and builds the SQL. If the LLM is not
available, a keyword parser handles the common phrasings."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from datetime import datetime, timedelta, tzinfo
from typing import Literal

from pydantic import BaseModel, Field

from aequitas.store.query import QueryFilter
from aequitas.taxonomy import DEFAULT_CATEGORIES, keyword_pattern
from aequitas.types import Status

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


# ---------- places: cameras and the areas (localities) they are in ----------

FUZZY = 0.85  # how close a misspelt place name must be (difflib ratio, spaces and punctuation ignored)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _close(a: str, b: str) -> bool:
    """A near miss of a name: letters may differ a little, numbers never (80 Feet Road is not 100 Feet Road)."""
    na, nb = _norm(a), _norm(b)
    return (len(nb) >= 5 and re.findall(r"\d+", na) == re.findall(r"\d+", nb)
            and SequenceMatcher(None, na, nb).ratio() >= FUZZY)


def camera_names(c: dict) -> list[str]:
    """Every name a camera answers to; "Forum Mall, Hosur Road" also answers to "Forum Mall"."""
    names = [c["name"], *c["aliases"]]
    if "," in c["name"]:
        names.append(c["name"].split(",")[0].strip())
    return names


def areas(cameras: list[dict]) -> dict[str, dict]:
    """Area name -> {"ids": its camera IDs, "names": the name and its aliases}."""
    out: dict[str, dict] = {}
    for c in cameras:
        a = c.get("area")
        if not a:
            continue
        e = out.setdefault(a, {"ids": [], "names": [a]})
        e["ids"].append(c["camera_id"])
        for al in c.get("area_aliases") or []:
            if al not in e["names"]:
                e["names"].append(al)
    return out


def _mentioned(words: list[str], name: str) -> bool:
    """Is `name` in the question, allowing a small misspelling (Kormangala, Indra nagar, mgroad)?"""
    k = len(re.findall(r"[a-z0-9]+", name.lower())) or 1
    for size in {max(1, k - 1), k, k + 1}:
        for i in range(len(words) - size + 1):
            chunk = "".join(words[i:i + size])
            if chunk == _norm(name) or _close(chunk, name):
                return True
    return False


def _prompt(question: str, now: datetime, cameras: list[dict], categories: dict[str, str] | None = None) -> str:
    cams = "\n".join(
        f"- {c['camera_id']}: {c['name']}" + (f", in {c['area']}" if c.get("area") else "")
        + (f" (also called: {', '.join(c['aliases'])})" if c["aliases"] else "")
        for c in cameras
    ) or "- (no cameras registered)"
    places = areas(cameras)
    if places:
        cams += "\n\nAreas (a question about an area means all its cameras):\n" + "\n".join(
            f"- {a}" + (f" (also called: {', '.join(e['names'][1:])})" if len(e["names"]) > 1 else "")
            for a, e in places.items())
    cats = "\n".join(f"- {name}: {d}" for name, d in (categories or DEFAULT_CATEGORIES).items())
    return (
        "Convert the user's question about stored security-camera events into a JSON search filter.\n\n"
        f"Current local time: {now.strftime('%A %Y-%m-%d %H:%M')}\n\n"
        f"Cameras:\n{cams}\n\n"
        f"Event categories:\n{cats}\n"
        "Subtypes: loitering and abandoned_object (both suspicious_activity), overcrowding (crowd_incident).\n\n"
        "Rules:\n"
        "- Only fill fields the question asks about; leave the rest empty or default.\n"
        "- cameras: copy the camera or area words the user used (e.g. 'camera 2', 'entrance', 'Indiranagar').\n"
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
    """Map the user's place words to camera IDs. Returns (ids, unknown mentions).

    A camera's name wins; an area name (or one of its aliases) means every camera in that area; then camera
    numbers, partial names, and finally close misspellings. If the question names both an area and one of its
    cameras ("Forum Mall in Koramangala"), the camera is meant, so the area is dropped."""
    places = areas(cameras)
    cam_hits: list[str] = []
    area_hits: list[list[str]] = []
    unknown = []
    for m in mentions:
        q = m.strip().lower()
        if not q:
            continue
        hit = next((c["camera_id"] for c in cameras
                    if _norm(q) in {_norm(n) for n in [c["camera_id"], *camera_names(c)]}), None)
        area = None
        if hit is None:
            area = next((e["ids"] for e in places.values() if _norm(q) in {_norm(n) for n in e["names"]}), None)
        if hit is None and area is None:
            num = re.search(r"(?:cam(?:era)?\s*#?\s*)(\d+)", q)
            if num:
                hit = next((c["camera_id"] for c in cameras if c["camera_id"].lower() in (f"cam{num[1]}", num[1])), None)
        if hit is None and area is None:
            hit = next((c["camera_id"] for c in cameras
                        if any(q in n or n in q for n in [c["name"].lower(), *(a.lower() for a in c["aliases"])])),
                       None)
        if hit is None and area is None:
            area = next((e["ids"] for e in places.values() if any(_close(q, n) for n in e["names"])), None)
        if hit is None and area is None:
            hit = next((c["camera_id"] for c in cameras if _close(q, c["name"])), None)
        if hit is not None:
            cam_hits.append(hit)
        elif area is not None:
            area_hits.append(area)
        else:
            unknown.append(m)
    ids: list[str] = []
    for i in cam_hits:
        if i not in ids:
            ids.append(i)
    for group in area_hits:
        if any(i in group for i in cam_hits):  # a camera in this area was named: that camera is meant
            continue
        ids += [i for i in group if i not in ids]
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
    words = re.findall(r"[a-z0-9]+", q)
    for c in cameras:
        for n in camera_names(c):
            if re.search(rf"\b{re.escape(n.lower())}\b", q):
                mentions.append(n)
                break
    found = {_norm(m) for m in mentions}
    for c in cameras:  # misspelt camera names ("jyothi nivas college road")
        if not found & {_norm(n) for n in camera_names(c)} and len(_norm(c["name"])) >= 8 \
                and _mentioned(words, c["name"]):
            mentions.append(c["name"])
    for a, e in areas(cameras).items():  # area names, their aliases and close misspellings ("kormangala")
        if any(_mentioned(words, n) for n in e["names"]):
            mentions.append(a)

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


# A time the keyword parser does not read: only then may the LLM's start/end stand on their own.
_TIME_HINT = re.compile(r"\b(\d{1,2}\s*(:\d{2})?\s*(am|pm)|\d{1,2}:\d{2}|o'?clock|noon|midnight|since|before|after|"
                        r"until|till|between|ago|night|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
                        r"weekend|month|january|february|march|april|may|june|july|august|september|october|"
                        r"november|december|\d{4}-\d{2}-\d{2})\b")


def reconcile(lf: LlmFilter, kw: LlmFilter, question: str) -> LlmFilter:
    """Check a small LLM's filter against what the question literally says. Small models invent
    cameras and drop categories; the keyword parse is a reliable floor for both.

    Measured on config/chat_eval.yaml (gemma4:e2b): the LLM read yes/no questions ("Any fights...?") as counts
    and invented or changed time ranges ("last 30 minutes" -> an hour; "today" where no time was asked). So
    intent comes from the explicit words, and the keyword parser's time range wins whenever it found one."""
    q = question.lower()
    cams = [c for c in lf.cameras if c.strip() and c.lower() in q]  # only cameras the user actually named
    for c in kw.cameras:
        if c not in cams:
            cams.append(c)
    if kw.start or kw.end:
        start, end = kw.start, kw.end
    elif _TIME_HINT.search(q):
        start, end = lf.start, lf.end
    else:
        start = end = None
    return lf.model_copy(update={
        "cameras": cams,
        "categories": lf.categories or kw.categories,
        "subtypes": lf.subtypes or kw.subtypes,
        "start": start,
        "end": end,
        "intent": kw.intent,
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
