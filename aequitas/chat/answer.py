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

from aequitas.chat import followup
from aequitas.chat.parser import _SUBTYPE_WORDS, ParseResult, QuestionParser, areas, camera_names
from aequitas.config import VectorCfg
from aequitas.store.db import EventStore
from aequitas.store.query import QueryFilter, to_epoch_ms
from aequitas.taxonomy import keyword_pattern, label
from aequitas.types import Status

log = logging.getLogger(__name__)

MAX_SUMMARY_ROWS = 15  # keep the prompt small on CPU-only machines
MAX_MOMENTS = 5  # analysed windows (not incidents) shown next to the events
RRF_K = 60  # reciprocal rank fusion constant (Cormack et al. 2009)

# Words that say nothing about what was seen: question words, time and place words, chat verbs. What is left of a
# question after these, the category keywords and the place names is its description ("carrying a ladder").
_STOP = set("""
a an the and or but nor of in on at to for from by with without into onto near around about over under
what which when where who whom whose why how many much any anything anyone anybody someone somebody something
there here is are was were be been being do does did done have has had will would can could should may might must
show list find give tell me us all every each events event incident incidents happened happen happening occurred
occur camera cameras cctv footage video videos clip clips today yesterday tonight morning afternoon evening night
week month hour hours minute minutes last past this that these those them they it its their his her our my your
see seen saw look looking spot spotted notice noticed detect detected unconfirmed confirmed disputed rejected
only just also now then same please ago since before after between until till more most again other else area
areas place places location locations recent recently latest first second third one ones times time count number
total people person persons similar like details detail describe explain summary summarise summarize
monday tuesday wednesday thursday friday saturday sunday january february march april may june july august
september october november december includ include including everywhere anywhere kind kinds type types status
""".split())


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
    timings: dict = field(default_factory=dict)  # ms: parse, query, embed, vector, summary, total
    moments: list[dict] = field(default_factory=list)  # analysed windows that match (not incidents)
    mode: str = "filter"  # filter | hybrid | detail | similar
    suggestions: list[str] = field(default_factory=list)
    context: dict = field(default_factory=dict)  # sent back with the next question (follow-ups)
    weak: bool = False  # vector search found only distant summaries (shown as weak matches, no LLM summary)
    earlier: bool = False  # nothing in the asked time range; rows are the latest matches outside it


@dataclass
class Retrieval:
    parsed: ParseResult
    total: int
    rows: list[dict]
    head: list[str]
    tail: list[str]
    timings: dict
    moments: list[dict] = field(default_factory=list)
    mode: str = "filter"
    kept: list[str] = field(default_factory=list)  # parts of the previous filter a follow-up kept
    weak: bool = False
    earlier: bool = False
    asked_about: list[dict] = field(default_factory=list)  # similar mode: the event compared against (LLM may cite it)

    @property
    def llm_rows(self) -> list[dict]:
        return self.asked_about + self.rows


def content_terms(question: str, cameras: list[dict], categories: dict[str, str]) -> list[str]:
    """The descriptive words of a question: what remains after question, time and place words and the category
    keywords are taken out. "Anyone carrying a ladder near Forum Mall today?" -> ["carrying", "ladder"]."""
    q = question.lower()
    for c in categories:
        q = re.sub(keyword_pattern(c), " ", q)
    for rx in _SUBTYPE_WORDS.values():
        q = re.sub(rx, " ", q)
    names = set()
    for c in cameras:
        names.update(n.lower() for n in camera_names({"name": c["name"], "aliases": c.get("aliases") or []}))
        if c.get("area"):
            names.add(c["area"].lower())
        names.update(a.lower() for a in c.get("area_aliases") or [])
    for n in sorted(names, key=len, reverse=True):
        q = re.sub(rf"\b{re.escape(n)}\b", " ", q)
    return [w for w in re.findall(r"[a-z]{3,}", q) if w not in _STOP]


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
    match = f" · match {r['relevance']:.2f}" if r.get("relevance") is not None else ""
    return (f"- [#{r['event_id']}] {_fmt_time(r['start_ts'], tz)} · {cam} · {label}{status} · "
            f"conf {r['confidence']:.2f}{match} — {summary}")


def format_moment(w: dict, tz: tzinfo | None) -> str:
    lab = w.get("label")
    tag = f" · labelled {lab.replace('_', ' ')}" if lab and lab != "none" else ""
    return f"- [W{w['id']}] {_fmt_time(w['start_ts'], tz)} · {place(w)}{tag} — {w.get('summary')}"


def _summary_prompt(question: str, rows: list[dict], tz: tzinfo | None, moments: list[dict] = ()) -> str:
    facts = "\n".join(
        f"[#{r['event_id']}] {_fmt_time(r['start_ts'], tz)}, camera {place(r)}, "
        f"{(r.get('subtype') or r['category']).replace('_', ' ')}, status {r['status']}: {r.get('summary') or 'no description'}"
        for r in rows
    )
    seen = "\n".join(f"[W{w['id']}] {_fmt_time(w['start_ts'], tz)}, camera {place(w)}: {w.get('summary')}"
                     for w in moments)
    return (
        "Answer the user's question using ONLY the events listed below. Do not add any event, time, "
        "camera or detail that is not listed.\n"
        "Write 1-3 sentences. Refer to events by their reference like [#12]"
        + (" and to camera moments like [W7]" if moments else "") + ". "
        "Put every event ID you mention in cited_event_ids.\n\n"
        f"Events:\n{facts or '(none)'}\n\n"
        + (f"Camera moments (what a camera saw in a few seconds; not incidents):\n{seen}\n\n" if moments else "")
        + f"Question: {question}"
    )


def _stream_prompt(question: str, rows: list[dict], tz: tzinfo | None, moments: list[dict] = ()) -> str:
    """Plain-text variant for streaming: citations are read from the [#id] references in the text."""
    return _summary_prompt(question, rows, tz, moments).replace(
        "Put every event ID you mention in cited_event_ids.", "Cite at least one reference. Answer in plain text."
    )


def is_grounded(text: str, rows: list[dict], extra_cited: list[int] = (), moments: list[dict] = ()) -> bool:
    """True if the text cites at least one event or moment and only ones that were retrieved."""
    allowed = {r["event_id"] for r in rows}
    cited = {int(m) for m in re.findall(r"\[#(\d+)\]", text)} | set(extra_cited)
    allowed_w = {w["id"] for w in moments}
    cited_w = {int(m) for m in re.findall(r"\[W(\d+)\]", text)}
    if not (cited or cited_w) or not cited <= allowed or not cited_w <= allowed_w:
        log.warning("discarding ungrounded summary (cited %s %s, allowed %s %s)", sorted(cited), sorted(cited_w),
                    sorted(allowed), sorted(allowed_w))
        return False
    return True


def grounded_summary(llm, question: str, rows: list[dict], tz: tzinfo | None,
                     moments: list[dict] = ()) -> str | None:
    """LLM summary of the retrieved rows. Discarded if it cites anything that was not retrieved."""
    rows = rows[:MAX_SUMMARY_ROWS]
    res = llm.structured([{"role": "user", "content": _summary_prompt(question, rows, tz, moments)}],
                         GroundedSummary)
    if res is None or not is_grounded(res.answer, rows, res.cited_event_ids, moments):
        return None
    return res.answer.strip()


def _best_of(results: list[list]) -> list:
    """Merge searches with several query vectors: each summary keeps its best score, best first."""
    best: dict = {}
    for hits in results:
        for h in hits:
            if h.ref_id not in best or h.score > best[h.ref_id].score:
                best[h.ref_id] = h
    return sorted(best.values(), key=lambda h: -h.score)


def _passing(hits: list, cfg: VectorCfg, margin: bool = True) -> list:
    """Hits close enough to count as matches: above min_score and (optionally) within margin of the best."""
    if not hits:
        return []
    floor = max(cfg.min_score, hits[0].score - cfg.margin) if margin else cfg.min_score
    return [h for h in hits if h.score >= floor]


def suggestions(mode: str, f: QueryFilter, rows: list[dict], total: int, vectors: bool) -> list[str]:
    """Next questions, built from the answer (no model call). Each one is phrased so the follow-up rules or
    the parser read it correctly."""
    out = []
    if rows:
        top = rows[0]["event_id"]
        if mode != "detail":
            out.append(f"Tell me more about #{top}")
        if vectors and mode != "similar":
            out.append(f"Find events similar to #{top}")
    if mode in ("detail", "similar"):
        return out
    if f.intent == "count" and total:
        out.append("Show them")
    if f.statuses == [Status.CONFIRMED]:
        out.append("What about unconfirmed ones?")
    if f.camera_ids:
        out.append("What about everywhere?")
    if f.start is None and f.end is None:
        out.append("And in the last 24 hours?")
    elif not total:
        out.append("What about this week?")
    return out[:4]


class ChatService:
    """Stage 3. Every answer is built from database rows; the vector index (when on) only decides which rows
    and in what order. Modes:

    - filter: a structured question ("fights in Indiranagar today"): SQL alone, as before
    - hybrid: the question also describes something ("anyone carrying a ladder"): SQL picks the events the
      filter allows, then Qdrant (meaning) and SQLite FTS5 BM25 (exact words) rank them, fused by reciprocal rank;
      analysed windows that match are returned as moments, so it also finds things that were never incidents
    - detail: "tell me more about #12" / "the second one": that event and what the camera saw around it
    - similar: "events similar to #12": nearest events by the meaning of their summaries"""

    def __init__(self, store: EventStore, parser: QuestionParser, summarize: bool = True, index=None,
                 vector_cfg: VectorCfg | None = None):
        self.store = store
        self.parser = parser
        self.summarize = summarize
        self.index = index  # aequitas.chat.vectors.VectorIndex, or None for keyword search only
        self.vcfg = vector_cfg or VectorCfg()

    # ---------- retrieval ----------

    def _retrieve(self, question: str, context: dict | None = None) -> Retrieval:
        """Parse and query. Timings in ms: parse = question -> filter (the LLM call, if any), embed = the question's
        vector, vector = the Qdrant searches, query = everything in SQLite."""
        t0 = time.perf_counter()
        cameras = self.store.cameras()
        refs = followup.references(question, context)
        if refs:
            return (self._similar if followup.wants_similar(question) else self._detail)(question, refs, t0)
        parsed = self.parser.parse(question, cameras, semantic=self.index is not None)
        timings = {"parse_ms": round((time.perf_counter() - t0) * 1000)}
        f = parsed.filter
        terms = content_terms(question, cameras, self.parser.categories) if self.index is not None else []
        if self.index is not None:
            f = f.model_copy(update={"text": " ".join(terms) or None})
        kept: list[str] = []
        inherited = False  # the description came from the previous question, not this one
        if followup.is_followup(question, context):
            inherited = not terms
            f, kept = followup.merge(context["filter"], f, question, self.parser.categories)
            terms = (f.text or "").split() if self.index is not None else []
        parsed = ParseResult(f, parsed.warnings, parsed.method)
        if kept:
            parsed.warnings.append("Following up on your last question (kept: " + ", ".join(kept) + ").")
        if terms:
            return self._hybrid(question, parsed, terms, timings, kept, cameras, inherited)
        t1 = time.perf_counter()
        total = self.store.count(f)
        desc = describe(f, cameras, total)
        head, tail = list(parsed.warnings), []
        if f.camera_ids and not f.categories and re.match(r"\s*where\b", question, re.I):
            head.append(where_answer(f, cameras))
        rows: list[dict] = []
        earlier = False
        if f.intent == "count":
            head.append(f"{total} {desc}.")
        elif not total:
            head.append(f"No {desc}.")
            if f.start or f.end:  # "nothing in the last hour" alone hides the incident from 10:41
                rows = self.store.query(f.model_copy(update={"start": None, "end": None, "limit": 3,
                                                             "sort": "newest"}))
                if rows:
                    earlier = True
                    head.append("The latest outside that time:")
                    tail.extend(format_row(r, self.parser.tz) for r in rows)
        else:
            rows = self.store.query(f)
            shown = f" (showing {len(rows)})" if total > len(rows) else ""
            tail.append(f"Found {total} {desc}{shown}:")
            tail.extend(format_row(r, self.parser.tz) for r in rows)
        timings["query_ms"] = round((time.perf_counter() - t1) * 1000)
        return Retrieval(parsed, total, rows, head, tail, timings, kept=kept, earlier=earlier)

    def _hybrid(self, question: str, parsed: ParseResult, terms: list[str], timings: dict, kept: list[str],
                cameras: list[dict], inherited: bool = False) -> Retrieval:
        """The question and its descriptive words are both embedded and each summary keeps its better score:
        "Were there any trains or railway tracks?" scored a "near the train tracks" window below the threshold,
        "trains railway tracks" well above it. A follow-up that carried the topic over ("and unconfirmed
        ones?") is searched by the carried-over words only."""
        f = parsed.filter
        t1 = time.perf_counter()
        ids = self.store.event_ids(f)
        sql_ms = (time.perf_counter() - t1) * 1000
        t2 = time.perf_counter()
        texts = list(dict.fromkeys(([] if inherited else [question]) + [" ".join(terms)]))
        vecs = [self.index.embed_query(t) for t in texts]
        timings["embed_ms"] = round((time.perf_counter() - t2) * 1000)
        t3 = time.perf_counter()
        win_ids = self.store.window_ids(f.camera_ids, to_epoch_ms(f.start) if f.start else None,
                                        to_epoch_ms(f.end) if f.end else None)
        sql_ms += (time.perf_counter() - t3) * 1000
        t4 = time.perf_counter()
        hits = _best_of([self.index.search(v, "event", ids, limit=max(30, f.limit)) for v in vecs])
        w_hits = _best_of([self.index.search(v, "window", win_ids, limit=MAX_MOMENTS * 2) for v in vecs])
        timings["vector_ms"] = round((time.perf_counter() - t4) * 1000, 1)
        t5 = time.perf_counter()
        # One bar for events and moments alike, anchored on the best match anywhere: a window that matches at
        # 0.83 means events at 0.63 ("a group of people outdoors" for "people near train tracks") are noise.
        best = max([h.score for h in hits[:1] + w_hits[:1]], default=0.0)
        floor = max(self.vcfg.min_score, best - self.vcfg.margin)
        passing = [h for h in hits if h.score >= floor]
        w_hits = [h for h in w_hits if h.score >= floor]
        words = self.store.fts_ids(" ".join(terms), ids)
        fused: dict[int, float] = {}
        for rank, h in enumerate(passing):
            fused[h.ref_id] = fused.get(h.ref_id, 0) + 1 / (RRF_K + rank)
        for rank, i in enumerate(words):
            fused[i] = fused.get(i, 0) + 1 / (RRF_K + rank)
        order = sorted(fused, key=lambda i: -fused[i])
        sim = {h.ref_id: h.score for h in hits}
        by_meaning, by_words = {h.ref_id for h in passing}, set(words)
        rows = self.store.events_by_ids(order[:f.limit])
        for r in rows:
            r["relevance"] = round(sim[r["event_id"]], 3) if r["event_id"] in sim else None
            r["matched"] = "+".join(n for n, s in (("meaning", by_meaning), ("words", by_words)) if r["event_id"] in s)
        moments = self.store.windows_by_ids([h.ref_id for h in w_hits][:MAX_MOMENTS])
        wsim = {h.ref_id: h.score for h in w_hits}
        for w in moments:
            w["relevance"] = round(wsim[w["id"]], 3)
        timings["query_ms"] = round(sql_ms + (time.perf_counter() - t5) * 1000)
        weak = bool(fused or moments) and not words and best < self.vcfg.strong_score
        total = 0 if weak else len(order)
        desc = describe(f, cameras, total)
        head, tail = list(parsed.warnings), []
        if weak:
            head.append(f"Nothing matches \"{' '.join(terms)}\" closely (best similarity {best:.2f}). "
                        "The nearest summaries are below; they are probably unrelated.")
            tail.append("Nearest summaries (weak matches):")
            tail.extend(format_row(r, self.parser.tz) for r in rows)
            tail.extend(format_moment(w, self.parser.tz) for w in moments)
            return Retrieval(parsed, 0, rows, head, tail, timings, moments, "hybrid", kept, weak=True)
        if f.intent == "count":
            head.append(f"{total} {desc}.")
            rows = []
        elif not total:
            head.append(f"No {desc}." + (" These camera moments come closest:" if moments else ""))
        else:
            shown = f" (showing {len(rows)})" if total > len(rows) else ""
            tail.append(f"Found {total} {desc}{shown}, best match first:")
            tail.extend(format_row(r, self.parser.tz) for r in rows)
        if moments:
            tail.append("Camera moments that match (analysed windows, not incidents):")
            tail.extend(format_moment(w, self.parser.tz) for w in moments)
        return Retrieval(parsed, total, rows, head, tail, timings, moments, "hybrid", kept)

    def _detail(self, question: str, refs: list[int], t0: float) -> Retrieval:
        """One event in full, with what its camera saw just before and after."""
        tz = self.parser.tz
        rows = self.store.events_by_ids(refs[:5])
        head, tail = [], []
        moments: list[dict] = []
        for r in rows:
            review = f" Officer review: {r['review']} by {r.get('reviewed_by')}." if r.get("review") else ""
            head.append(
                f"#{r['event_id']} · {label(r.get('subtype') or r['category'])} · {r['status']} · confidence "
                f"{r['confidence']:.2f} · {_fmt_time(r['start_ts'], tz)} to {_fmt_time(r['end_ts'], tz, False)} · "
                f"{place(r)}.{review}")
            near = self.store.windows_near(r["camera_id"], r["start_ts"], r["end_ts"])
            moments += [w for w in near if w["id"] not in {m["id"] for m in moments}]
        missing = [i for i in refs[:5] if i not in {r["event_id"] for r in rows}]
        if missing:
            head.append("There is no event " + ", ".join(f"#{i}" for i in missing) + ".")
        tail.extend(format_row(r, tz) for r in rows)
        if moments:
            tail.append("What the camera saw around it:")
            tail.extend(format_moment(w, tz) for w in moments)
        f = QueryFilter(statuses=list(Status))
        timings = {"parse_ms": 0, "query_ms": round((time.perf_counter() - t0) * 1000)}
        return Retrieval(ParseResult(f, [], "reference"), len(rows), rows, head, tail, timings,
                         moments[:8], "detail")

    def _similar(self, question: str, refs: list[int], t0: float) -> Retrieval:
        """Events whose summaries mean much the same as #ref's (any status, any camera)."""
        tz = self.parser.tz
        f = QueryFilter(statuses=list(Status))
        timings: dict = {"parse_ms": 0}
        ref = refs[0]
        if self.index is None:
            return Retrieval(ParseResult(f, [], "reference"), 0, [], [
                "Finding similar events needs the vector index, which is switched off (chat.vector.enabled)."],
                [], timings, mode="similar")
        t1 = time.perf_counter()
        v = self.index.vector_of("event", ref)
        hits = _passing(self.index.search(v, "event", None, limit=10, exclude=ref), self.vcfg,
                        margin=False) if v else []
        timings["vector_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        rows = self.store.events_by_ids([h.ref_id for h in hits])
        sim = {h.ref_id: h.score for h in hits}
        for r in rows:
            r["relevance"], r["matched"] = round(sim[r["event_id"]], 3), "meaning"
        timings["query_ms"] = round((time.perf_counter() - t0) * 1000 - timings["vector_ms"])
        if v is None:
            head = [f"Event #{ref} has no summary in the vector index yet."]
        elif not rows:
            head = [f"No event is close in meaning to #{ref}."]
        else:
            head = [f"{len(rows)} event{'s' if len(rows) != 1 else ''} most like #{ref}, by the meaning of the "
                    "summaries (any status, any camera):"]
        return Retrieval(ParseResult(f, [], "reference"), len(rows), rows, head,
                         [format_row(r, tz) for r in rows], timings, mode="similar",
                         asked_about=self.store.events_by_ids([ref]))

    def _wants_summary(self, ret: Retrieval) -> bool:
        return bool(ret.rows or ret.moments) and not (ret.weak or ret.earlier) and self.summarize and self.parser.llm is not None

    def _extras(self, ret: Retrieval) -> dict:
        f = ret.parsed.filter
        shown = [] if ret.weak or ret.earlier else ret.rows  # don't offer "tell me more" about an unrelated nearest match
        return {"suggestions": suggestions(ret.mode, f, shown, ret.total, self.index is not None),
                "context": {"filter": f.model_dump(mode="json"), "event_ids": [r["event_id"] for r in ret.rows],
                            "mode": ret.mode}}

    def ask(self, question: str, context: dict | None = None) -> ChatAnswer:
        t0 = time.perf_counter()
        ret = self._retrieve(question, context)
        timings = ret.timings
        summary = None
        if self._wants_summary(ret):
            t1 = time.perf_counter()
            summary = grounded_summary(self.parser.llm, question, ret.llm_rows, self.parser.tz, ret.moments)
            timings["summary_ms"] = round((time.perf_counter() - t1) * 1000)
        timings["total_ms"] = round((time.perf_counter() - t0) * 1000)
        mid = [summary, ""] if summary else []
        ex = self._extras(ret)
        return ChatAnswer("\n".join(ret.head + mid + ret.tail), ret.rows, ret.parsed, ret.total, summary, timings,
                          ret.moments, ret.mode, ex["suggestions"], ex["context"], ret.weak, ret.earlier)

    def ask_stream(self, question: str, context: dict | None = None) -> Iterator[dict]:
        """The same answer as ask(), as events for streaming:

        - retrieved: the event list, straight from the database (sent before any LLM text)
        - token: a piece of the summary as the LLM writes it
        - summary: the final summary, or null if it failed the grounding check (clients then drop
          the streamed text)
        - done: timings

        The summary is checked only once it is complete, so streamed text is provisional until then."""
        t0 = time.perf_counter()
        ret = self._retrieve(question, context)
        rows, timings = ret.rows, ret.timings
        retrieve_ms = (time.perf_counter() - t0) * 1000
        yield {"type": "retrieved", "parsed": ret.parsed, "total": ret.total, "rows": rows, "head": ret.head,
               "tail": ret.tail, "moments": ret.moments, "mode": ret.mode, "kept": ret.kept, "weak": ret.weak, "earlier": ret.earlier,
               "retrieve_ms": round(retrieve_ms), **timings, **self._extras(ret)}
        summary_ms = ttft_ms = None
        if self._wants_summary(ret):
            t1 = time.perf_counter()
            llm, shown = self.parser.llm, ret.llm_rows[:MAX_SUMMARY_ROWS]
            text, summary = "", None
            if hasattr(llm, "stream_text"):
                try:
                    prompt = _stream_prompt(question, shown, self.parser.tz, ret.moments)
                    for delta in llm.stream_text([{"role": "user", "content": prompt}]):
                        if ttft_ms is None:
                            ttft_ms = round((time.perf_counter() - t1) * 1000)
                        text += delta
                        yield {"type": "token", "text": delta}
                    summary = text.strip() if is_grounded(text, shown, moments=ret.moments) else None
                except Exception as e:
                    log.warning("streamed summary failed: %s", e)
                    text = ""
            if not text:  # no streaming support, or it failed before any text: one structured call
                summary = grounded_summary(llm, question, ret.llm_rows, self.parser.tz, ret.moments)
            summary_ms = round((time.perf_counter() - t1) * 1000)
            yield {"type": "summary", "summary": summary}
        yield {"type": "done", "retrieve_ms": round(retrieve_ms), **timings, "summary_ttft_ms": ttft_ms,
               "summary_ms": summary_ms, "elapsed_ms": round((time.perf_counter() - t0) * 1000)}
