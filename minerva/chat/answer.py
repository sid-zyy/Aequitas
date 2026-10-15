"""Builds the chatbot's grounded reply: an optional short summary that may only cite retrieved
events, followed by the matching events, each with an [#id] reference. The event list always
comes from the database rows, never from the LLM, so it cannot invent events."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, tzinfo

from pydantic import BaseModel, Field

from minerva.chat.parser import ParseResult, QuestionParser
from minerva.store.db import EventStore
from minerva.store.query import QueryFilter
from minerva.types import Status

log = logging.getLogger(__name__)

_LABELS = {
    "person_down": "person down",
    "fighting": "fighting",
    "tailgating": "tailgating",
    "crowd_incident": "crowd incident",
    "suspicious_activity": "suspicious activity",
}
MAX_SUMMARY_ROWS = 15  # keep the prompt small on CPU-only machines


class GroundedSummary(BaseModel):
    answer: str = Field(max_length=800)
    cited_event_ids: list[int] = Field(default_factory=list)


@dataclass
class ChatAnswer:
    text: str
    rows: list[dict] = field(default_factory=list)
    parsed: ParseResult | None = None
    total: int = 0
    summary: str | None = None


def _fmt_time(ms: int, tz: tzinfo | None, with_date: bool = True) -> str:
    dt = datetime.fromtimestamp(ms / 1000, tz) if tz else datetime.fromtimestamp(ms / 1000).astimezone()
    return dt.strftime("%Y-%m-%d %H:%M:%S" if with_date else "%H:%M:%S")


def describe(f: QueryFilter, cameras: list[dict]) -> str:
    parts = []
    if f.subtypes:
        parts.append(" / ".join(s.replace("_", " ") for s in f.subtypes))
    elif f.categories:
        parts.append(" / ".join(_LABELS[c.value] for c in f.categories))
    statuses = {s.value for s in f.statuses}
    if statuses == {Status.CONFIRMED.value}:
        parts.insert(0, "confirmed")
    s = " ".join(parts + ["events"])
    if f.camera_ids:
        names = {c["camera_id"]: c["name"] for c in cameras}
        s += " on " + ", ".join(f"{names.get(i, i)} ({i})" for i in f.camera_ids)
    if f.start or f.end:
        fmt = "%Y-%m-%d %H:%M"
        s += f" between {f.start.strftime(fmt) if f.start else 'the beginning'} and {f.end.strftime(fmt) if f.end else 'now'}"
    if f.text:
        s += f" matching \"{f.text}\""
    return s


def format_row(r: dict, tz: tzinfo | None) -> str:
    label = (r.get("subtype") or r["category"]).replace("_", " ")
    status = "" if r["status"] == "confirmed" else f" [{r['status']}]"
    cam = r.get("camera_name") or r["camera_id"]
    summary = r.get("summary") or "(no summary)"
    return (f"- [#{r['event_id']}] {_fmt_time(r['start_ts'], tz)} · {cam} · {label}{status} · "
            f"conf {r['confidence']:.2f} — {summary}")


def _summary_prompt(question: str, rows: list[dict], tz: tzinfo | None) -> str:
    facts = "\n".join(
        f"[#{r['event_id']}] {_fmt_time(r['start_ts'], tz)}, camera {r.get('camera_name') or r['camera_id']}, "
        f"{(r.get('subtype') or r['category']).replace('_', ' ')}, status {r['status']}: {r.get('summary') or 'no description'}"
        for r in rows
    )
    return (
        "Answer the user's question using ONLY the events listed below. Do not add any event, time, "
        "camera or detail that is not listed.\n"
        "Write 1-3 sentences. Refer to events by their reference like [#12]. "
        "Put every event ID you mention in cited_event_ids.\n\n"
        f"Events:\n{facts}\n\nQuestion: {question}"
    )


def grounded_summary(llm, question: str, rows: list[dict], tz: tzinfo | None) -> str | None:
    """LLM summary of the retrieved rows. Discarded if it cites anything that was not retrieved."""
    rows = rows[:MAX_SUMMARY_ROWS]
    res = llm.structured([{"role": "user", "content": _summary_prompt(question, rows, tz)}], GroundedSummary)
    if res is None:
        return None
    allowed = {r["event_id"] for r in rows}
    mentioned = {int(m) for m in re.findall(r"\[#(\d+)\]", res.answer)}
    cited = set(res.cited_event_ids) | mentioned
    if not cited or not cited <= allowed:
        log.warning("discarding ungrounded summary (cited %s, allowed %s)", sorted(cited), sorted(allowed))
        return None
    return res.answer.strip()


class ChatService:
    def __init__(self, store: EventStore, parser: QuestionParser, summarize: bool = True):
        self.store = store
        self.parser = parser
        self.summarize = summarize

    def ask(self, question: str) -> ChatAnswer:
        cameras = self.store.cameras()
        parsed = self.parser.parse(question, cameras)
        f = parsed.filter
        total = self.store.count(f)
        desc = describe(f, cameras)
        lines = list(parsed.warnings)
        summary = None
        rows: list[dict] = []
        if f.intent == "count":
            lines.append(f"{total} {desc}.")
        elif not total:
            lines.append(f"No {desc}.")
        else:
            rows = self.store.query(f)
            if self.summarize and self.parser.llm is not None:
                summary = grounded_summary(self.parser.llm, question, rows, self.parser.tz)
            if summary:
                lines += [summary, ""]
            shown = f" (showing {len(rows)})" if total > len(rows) else ""
            lines.append(f"Found {total} {desc}{shown}:")
            lines.extend(format_row(r, self.parser.tz) for r in rows)
        return ChatAnswer("\n".join(lines), rows, parsed, total, summary)
