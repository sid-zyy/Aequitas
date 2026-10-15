"""Builds the chatbot's reply from database rows with a fixed template. The LLM never writes
the answer text, so it cannot invent events."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, tzinfo

from minerva.chat.parser import ParseResult, QuestionParser
from minerva.store.db import EventStore
from minerva.store.query import QueryFilter
from minerva.types import Status

_LABELS = {
    "person_down": "person down",
    "fighting": "fighting",
    "tailgating": "tailgating",
    "crowd_incident": "crowd incident",
    "suspicious_activity": "suspicious activity",
}


@dataclass
class ChatAnswer:
    text: str
    rows: list[dict] = field(default_factory=list)
    parsed: ParseResult | None = None
    total: int = 0


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
    noun = "events"
    s = " ".join(parts + [noun])
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
    return f"- {_fmt_time(r['start_ts'], tz)} · {cam} · {label}{status} · conf {r['confidence']:.2f} — {summary}"


class ChatService:
    def __init__(self, store: EventStore, parser: QuestionParser):
        self.store = store
        self.parser = parser

    def ask(self, question: str) -> ChatAnswer:
        cameras = self.store.cameras()
        parsed = self.parser.parse(question, cameras)
        f = parsed.filter
        total = self.store.count(f)
        desc = describe(f, cameras)
        lines = list(parsed.warnings)
        if f.intent == "count":
            lines.append(f"{total} {desc}.")
            rows = []
        else:
            rows = self.store.query(f) if total else []
            if not total:
                lines.append(f"No {desc}.")
            else:
                shown = f" (showing {len(rows)})" if total > len(rows) else ""
                lines.append(f"Found {total} {desc}{shown}:")
                lines.extend(format_row(r, self.parser.tz) for r in rows)
        return ChatAnswer("\n".join(lines), rows, parsed, total)
