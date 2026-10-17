"""Builds the chatbot's grounded reply: an optional short summary that may only cite retrieved
events, followed by the matching events, each with an [#id] reference. The event list always
comes from the database rows, never from the LLM, so it cannot invent events."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, tzinfo

from pydantic import BaseModel, Field

from aequitas.chat.parser import ParseResult, QuestionParser, areas
from aequitas.store.db import EventStore
from aequitas.store.query import QueryFilter
from aequitas.taxonomy import label
from aequitas.types import Status

log = logging.getLogger(__name__)

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
    timings: dict = field(default_factory=dict)  # ms: parse, query, summary, total


def _fmt_time(ms: int, tz: tzinfo | None, with_date: bool = True) -> str:
    dt = datetime.fromtimestamp(ms / 1000, tz) if tz else datetime.fromtimestamp(ms / 1000).astimezone()
    return dt.strftime("%Y-%m-%d %H:%M:%S" if with_date else "%H:%M:%S")


def place(c: dict) -> str:
    """A camera's name with the area it is in: "Double Road Restaurant, Indiranagar"."""
    area = c.get("area") or c.get("camera_area")
    name = c.get("name") or c.get("camera_name") or c.get("camera_id", "")
    return f"{name}, {area}" if area else name


def where_answer(f: QueryFilter, cameras: list[dict]) -> str:
    """For "Where is ...?": where the named place is, from the camera list (no LLM)."""
    whole = [(a, e) for a, e in areas(cameras).items() if set(e["ids"]) == set(f.camera_ids)]
    by_id = {c["camera_id"]: c for c in cameras}
    if whole:
        a, e = whole[0]
        return f"{a} is an area with {len(e['ids'])} camera points: " + ", ".join(
            by_id[i]["name"] for i in e["ids"] if i in by_id) + "."
    out = []
    for i in f.camera_ids:
        c = by_id.get(i)
        if c is None:
            continue
        out.append(f"{c['name']} is in {c['area']} (camera {i})." if c.get("area")
                   else f"{c['name']} is camera {i}; it isn't assigned to an area.")
    return " ".join(out)


def describe(f: QueryFilter, cameras: list[dict], count: int = 2) -> str:
    parts = []
    if f.subtypes:
        parts.append(" / ".join(s.replace("_", " ") for s in f.subtypes))
    elif f.categories:
        parts.append(" / ".join(label(c) for c in f.categories))
    statuses = {s.value for s in f.statuses}
    if statuses == {Status.CONFIRMED.value}:
        parts.insert(0, "confirmed")
    s = " ".join(parts + ["event" if count == 1 else "events"])
    if f.camera_ids:
        whole = [a for a, e in areas(cameras).items() if set(e["ids"]) == set(f.camera_ids)]
        if whole:
            s += f" in {whole[0]} (all {len(f.camera_ids)} cameras)"
        else:
            by_id = {c["camera_id"]: c for c in cameras}
            s += " on " + ", ".join(f"{place(by_id.get(i, {'name': i}))} ({i})" for i in f.camera_ids)
    if f.start or f.end:
        fmt = "%Y-%m-%d %H:%M"
        s += f" between {f.start.strftime(fmt) if f.start else 'the beginning'} and {f.end.strftime(fmt) if f.end else 'now'}"
    if f.text:
        s += f" matching \"{f.text}\""
    return s


def format_row(r: dict, tz: tzinfo | None) -> str:
    label = (r.get("subtype") or r["category"]).replace("_", " ")
    status = "" if r["status"] == "confirmed" else f" [{r['status']}]"
    cam = place(r)
    summary = r.get("summary") or "(no summary)"
    return (f"- [#{r['event_id']}] {_fmt_time(r['start_ts'], tz)} · {cam} · {label}{status} · "
            f"conf {r['confidence']:.2f} — {summary}")


def _summary_prompt(question: str, rows: list[dict], tz: tzinfo | None) -> str:
    facts = "\n".join(
        f"[#{r['event_id']}] {_fmt_time(r['start_ts'], tz)}, camera {place(r)}, "
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


def _stream_prompt(question: str, rows: list[dict], tz: tzinfo | None) -> str:
    """Plain-text variant for streaming: citations are read from the [#id] references in the text."""
    return _summary_prompt(question, rows, tz).replace(
        "Put every event ID you mention in cited_event_ids.", "Cite at least one event. Answer in plain text."
    )


def is_grounded(text: str, rows: list[dict], extra_cited: list[int] = ()) -> bool:
    """True if the text cites at least one event and only events that were retrieved."""
    allowed = {r["event_id"] for r in rows}
    cited = {int(m) for m in re.findall(r"\[#(\d+)\]", text)} | set(extra_cited)
    if not cited or not cited <= allowed:
        log.warning("discarding ungrounded summary (cited %s, allowed %s)", sorted(cited), sorted(allowed))
        return False
    return True


def grounded_summary(llm, question: str, rows: list[dict], tz: tzinfo | None) -> str | None:
    """LLM summary of the retrieved rows. Discarded if it cites anything that was not retrieved."""
    rows = rows[:MAX_SUMMARY_ROWS]
    res = llm.structured([{"role": "user", "content": _summary_prompt(question, rows, tz)}], GroundedSummary)
    if res is None or not is_grounded(res.answer, rows, res.cited_event_ids):
        return None
    return res.answer.strip()


class ChatService:
    def __init__(self, store: EventStore, parser: QuestionParser, summarize: bool = True):
        self.store = store
        self.parser = parser
        self.summarize = summarize

    def _retrieve(self, question: str) -> tuple[ParseResult, int, list[dict], list[str], list[str], dict]:
        """Parse and query. Returns (parsed, total, rows, lines before the summary, lines after it,
        timings in ms: parse = question -> filter (the LLM call), query = the database)."""
        t0 = time.perf_counter()
        cameras = self.store.cameras()
        parsed = self.parser.parse(question, cameras)
        timings = {"parse_ms": round((time.perf_counter() - t0) * 1000)}
        t1 = time.perf_counter()
        f = parsed.filter
        total = self.store.count(f)
        desc = describe(f, cameras, total)
        head, tail = list(parsed.warnings), []
        if f.camera_ids and not f.categories and re.match(r"\s*where\b", question, re.I):
            head.append(where_answer(f, cameras))
        rows: list[dict] = []
        if f.intent == "count":
            head.append(f"{total} {desc}.")
        elif not total:
            head.append(f"No {desc}.")
        else:
            rows = self.store.query(f)
            shown = f" (showing {len(rows)})" if total > len(rows) else ""
            tail.append(f"Found {total} {desc}{shown}:")
            tail.extend(format_row(r, self.parser.tz) for r in rows)
        timings["query_ms"] = round((time.perf_counter() - t1) * 1000)
        return parsed, total, rows, head, tail, timings

    def _wants_summary(self, rows: list[dict]) -> bool:
        return bool(rows) and self.summarize and self.parser.llm is not None

    def ask(self, question: str) -> ChatAnswer:
        t0 = time.perf_counter()
        parsed, total, rows, head, tail, timings = self._retrieve(question)
        summary = None
        if self._wants_summary(rows):
            t1 = time.perf_counter()
            summary = grounded_summary(self.parser.llm, question, rows, self.parser.tz)
            timings["summary_ms"] = round((time.perf_counter() - t1) * 1000)
        timings["total_ms"] = round((time.perf_counter() - t0) * 1000)
        mid = [summary, ""] if summary else []
        return ChatAnswer("\n".join(head + mid + tail), rows, parsed, total, summary, timings)

    def ask_stream(self, question: str) -> Iterator[dict]:
        """The same answer as ask(), as events for streaming:

        - retrieved: the event list, straight from the database (sent before any LLM text)
        - token: a piece of the summary as the LLM writes it
        - summary: the final summary, or null if it failed the grounding check (clients then drop
          the streamed text)
        - done: timings

        The summary is checked only once it is complete, so streamed text is provisional until then."""
        t0 = time.perf_counter()
        parsed, total, rows, head, tail, timings = self._retrieve(question)
        retrieve_ms = (time.perf_counter() - t0) * 1000
        yield {"type": "retrieved", "parsed": parsed, "total": total, "rows": rows, "head": head, "tail": tail,
               "retrieve_ms": round(retrieve_ms), **timings}
        summary_ms = ttft_ms = None
        if self._wants_summary(rows):
            t1 = time.perf_counter()
            llm, shown = self.parser.llm, rows[:MAX_SUMMARY_ROWS]
            text, summary = "", None
            if hasattr(llm, "stream_text"):
                try:
                    for delta in llm.stream_text([{"role": "user", "content": _stream_prompt(question, shown,
                                                                                            self.parser.tz)}]):
                        if ttft_ms is None:
                            ttft_ms = round((time.perf_counter() - t1) * 1000)
                        text += delta
                        yield {"type": "token", "text": delta}
                    summary = text.strip() if is_grounded(text, shown) else None
                except Exception as e:
                    log.warning("streamed summary failed: %s", e)
                    text = ""
            if not text:  # no streaming support, or it failed before any text: one structured call
                summary = grounded_summary(llm, question, rows, self.parser.tz)
            summary_ms = round((time.perf_counter() - t1) * 1000)
            yield {"type": "summary", "summary": summary}
        yield {"type": "done", "retrieve_ms": round(retrieve_ms), **timings, "summary_ttft_ms": ttft_ms,
               "summary_ms": summary_ms, "elapsed_ms": round((time.perf_counter() - t0) * 1000)}
