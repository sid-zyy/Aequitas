"""The chatbot's vector database (Qdrant) and the interactive chat: hybrid search, moments, follow-ups,
references, detail and similar-event answers. Real embedded Qdrant (in memory), word-hash embedder."""

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from helpers import WordEmbedder

from aequitas.chat import followup
from aequitas.chat.answer import ChatService, content_terms, is_grounded, suggestions
from aequitas.chat.parser import QuestionParser
from aequitas.chat.vectors import VectorIndex
from aequitas.config import CameraCfg, Settings, VectorCfg
from aequitas.store.db import EventRecord, EventStore
from aequitas.store.query import QueryFilter
from aequitas.taxonomy import DEFAULT_CATEGORIES
from aequitas.types import Status

CAMS = [
    CameraCfg(camera_id="cam1", name="Lobby entrance", aliases=["camera 1"], source="a.mp4", area="Indiranagar"),
    CameraCfg(camera_id="cam2", name="Corridor", aliases=["camera 2"], source="b.mp4", area="Koramangala"),
]
LOOSE = VectorCfg(min_score=0.1, margin=0.5, strong_score=0.1)  # the word-hash embedder scores lower than a real model


def ms(dt):
    return int(dt.timestamp() * 1000)


@pytest.fixture
def store(tmp_path):
    s = EventStore(tmp_path / "t.db")
    for c in CAMS:
        s.upsert_camera(c)
    now = datetime.now().astimezone()

    def add(cam, cat, ago_min, summary, status="confirmed"):
        t = now - timedelta(minutes=ago_min)
        return s.insert_event(EventRecord(
            event_uid=f"{cam}-{cat}-{ago_min}", camera_id=cam, start_ts=ms(t), end_ts=ms(t) + 4000, category=cat,
            subtype=None, status=status, confidence=0.8, trigger_score=0.5, vlm_category=cat, summary=summary,
            zone=None, track_ids=[], keyframe_path="x.jpg", model_versions={}))

    s.ids = {
        "ladder": add("cam2", "suspicious_activity", 30, "A man carries a tall ladder past the corridor doors."),
        "fight": add("cam2", "fighting", 60, "Two men punch and shove each other near the doors."),
        "fight1": add("cam1", "fighting", 90, "Two men punch each other at the lobby entrance."),
        "fire": add("cam1", "fire", 120, "Smoke and flames rise from a parked scooter."),
        "unverified": add("cam1", "fighting", 20, "Two men punch each other again.", status="unverified"),
    }
    t = now - timedelta(minutes=30, seconds=8)
    s.record_window("cam2", ms(t), ms(t) + 4000, "A worker holds a ladder against the wall.", "none", 0.9, 1, 1, 1)
    s.record_window("cam1", ms(t), ms(t) + 4000, "A dog walks along the footpath.", "none", 0.9, 1, 1, 1)
    return s


@pytest.fixture
def index(store):
    idx = VectorIndex(store, WordEmbedder())
    idx.sync()
    return idx


def chat(store, index=None):
    return ChatService(store, QuestionParser(None), summarize=False, index=index, vector_cfg=LOOSE)


def test_sync_indexes_events_and_windows_once_and_prunes_windows(store, index):
    assert index.stats()["events"] == 5 and index.stats()["windows"] == 2
    assert index.sync() == 0  # nothing new
    now = ms(datetime.now().astimezone())
    store.record_window("cam1", now, now + 4000, "A bus stops and people get on.", "none", 0.9, 1, 1, 1, keep=1)
    assert index.sync() == 1
    assert index.stats()["windows"] == 2  # cam1 kept only its newest window; the dog window's vector is gone
    assert all(h.ref_id != 2 for h in index.search(index.embed_query("dog footpath"), "window"))


def test_index_survives_a_restart_from_disk(store, tmp_path):
    idx = VectorIndex(store, WordEmbedder(), path=tmp_path / "vec")
    idx.sync()
    idx.close()
    again = VectorIndex(store, WordEmbedder(), path=tmp_path / "vec")
    assert again.stats()["events"] == 5 and again.sync() == 0


def test_content_terms_drop_places_times_and_category_words():
    cams = [c.model_dump() | {"aliases": c.aliases} for c in CAMS]
    assert content_terms("Anyone carrying a ladder near the Corridor in Koramangala today?", cams,
                         DEFAULT_CATEGORIES) == ["carrying", "ladder"]
    assert content_terms("Show all fights on camera 2 in the last hour", cams, DEFAULT_CATEGORIES) == []


def test_descriptive_question_is_answered_by_meaning_with_moments(store, index):
    ans = chat(store, index).ask("Was anyone carrying a ladder?")
    assert ans.mode == "hybrid"
    assert [r["event_id"] for r in ans.rows][0] == store.ids["ladder"]
    assert ans.rows[0]["relevance"] > 0 and "meaning" in ans.rows[0]["matched"]
    assert ans.moments and "ladder" in ans.moments[0]["summary"]
    assert "embed_ms" in ans.timings and "vector_ms" in ans.timings


def test_vector_search_stays_inside_the_structured_filter(store, index):
    ans = chat(store, index).ask("Anyone carrying a ladder on camera 1?")
    assert ans.mode == "hybrid" and store.ids["ladder"] not in [r["event_id"] for r in ans.rows]
    assert all(m["camera_id"] == "cam1" for m in ans.moments)


def test_structured_question_skips_vectors(store, index):
    ans = chat(store, index).ask("Show all fights on camera 2")
    assert ans.mode == "filter" and [r["event_id"] for r in ans.rows] == [store.ids["fight"]]


def test_without_the_index_the_chat_works_as_before(store):
    ans = chat(store, None).ask("Was anyone carrying a ladder?")
    assert ans.mode == "filter"


def test_follow_up_keeps_the_previous_filter(store, index):
    svc = chat(store, index)
    first = svc.ask("Show all fights on camera 2")
    nxt = svc.ask("What about camera 1?", first.context)
    assert [r["event_id"] for r in nxt.rows] == [store.ids["fight1"]]  # still fights, now on camera 1
    assert "kept: incident type" in nxt.text
    wider = svc.ask("and unconfirmed ones?", nxt.context)
    assert {r["event_id"] for r in wider.rows} == {store.ids["fight1"], store.ids["unverified"]}
    every = svc.ask("what about everywhere?", wider.context)
    assert len(every.rows) == 3


def test_references_resolve_against_the_previous_answer():
    ctx = {"filter": QueryFilter().model_dump(mode="json"), "event_ids": [7, 9, 4]}
    assert followup.references("tell me more about #12", ctx) == [12]
    assert followup.references("details of event 5", None) == [5]
    assert followup.references("what about the second one?", ctx) == [9]
    assert followup.references("and the last one", ctx) == [4]
    assert followup.references("tell me about it", ctx) == []  # three events: "it" is ambiguous
    assert followup.references("tell me about it", {**ctx, "event_ids": [7]}) == [7]
    assert followup.references("show the first event today", ctx) == []


def test_detail_answer_shows_the_event_and_what_the_camera_saw_around_it(store, index):
    ans = chat(store, index).ask(f"Tell me more about #{store.ids['ladder']}")
    assert ans.mode == "detail" and ans.rows[0]["event_id"] == store.ids["ladder"]
    assert ans.moments and ans.moments[0]["camera_id"] == "cam2"
    assert "There is no event #999" in chat(store, index).ask("what is #999?").text


def test_similar_events_by_meaning(store, index):
    ans = chat(store, index).ask(f"Find events similar to #{store.ids['fight']}")
    ids = [r["event_id"] for r in ans.rows]
    assert ans.mode == "similar" and store.ids["fight"] not in ids
    assert ids[0] in (store.ids["fight1"], store.ids["unverified"])
    assert store.ids["fire"] not in ids[:2]
    assert "vector index" in chat(store, None).ask("events similar to #1").text


def test_grounding_accepts_cited_moments_only_if_retrieved():
    rows, moments = [{"event_id": 3}], [{"id": 8}]
    assert is_grounded("A worker held a ladder [W8].", rows, moments=moments)
    assert not is_grounded("A worker held a ladder [W9].", rows, moments=moments)
    assert not is_grounded("Something happened.", rows, moments=moments)


def test_suggestions_are_follow_ups_the_chat_understands(store, index):
    svc = chat(store, index)
    ans = svc.ask("Show all fights on camera 2")
    assert f"Tell me more about #{store.ids['fight']}" in ans.suggestions
    for s in ans.suggestions:  # every suggestion is answerable with the returned context
        assert svc.ask(s, ans.context).text
    assert suggestions("filter", QueryFilter(statuses=list(Status)), [], 0, False) == ["And in the last 24 hours?"]


def test_api_returns_context_moments_and_index_stats(tmp_path):
    s = Settings(data_dir=tmp_path, auth={"required": False}, live={"localities_file": tmp_path / "none.yaml"})
    store = EventStore(s.db_path)
    for c in CAMS:
        store.upsert_camera(c)
    t = ms(datetime.now().astimezone()) - 60000
    store.record_window("cam2", t, t + 4000, "A worker holds a ladder against the wall.", "none", 0.9, 1, 1, 1)
    svc = ChatService(store, QuestionParser(None), summarize=False, vector_cfg=LOOSE)
    svc.index = VectorIndex(store, WordEmbedder())
    svc.index.sync()
    c = TestClient(__import__("aequitas.web.server", fromlist=["create_app"]).create_app(s, svc, []))
    r = c.post("/api/ask", json={"question": "anyone holding a ladder?"}).json()
    assert r["mode"] == "hybrid" and r["moments"][0]["summary"].startswith("A worker")
    assert r["context"]["filter"] and isinstance(r["suggestions"], list)
    r2 = c.post("/api/ask", json={"question": "and unconfirmed ones?", "context": r["context"]}).json()
    assert r2["filter"]["statuses"] != ["confirmed"]
    stats = c.get("/api/chat/index").json()
    assert stats["ready"] and stats["engine"] == "qdrant" and stats["windows"] == 1


def test_follow_up_searches_by_the_carried_over_topic(store, index):
    svc = chat(store, index)
    first = svc.ask("Was anyone carrying a ladder?")
    seen = []
    real = index.embed_query
    index.embed_query = lambda t: (seen.append(t), real(t))[1]
    nxt = svc.ask("and unconfirmed ones?", first.context)
    assert nxt.mode == "hybrid" and seen == ["carrying ladder"]  # not "and unconfirmed ones?"
    assert nxt.rows[0]["event_id"] == store.ids["ladder"]


def test_one_bar_for_events_and_moments_anchored_on_the_best_match(store, index):
    svc = ChatService(store, QuestionParser(None), summarize=False, index=index,
                      vector_cfg=VectorCfg(min_score=0.05, margin=0.05))
    ans = svc.ask("Was anyone carrying a ladder?")
    best = max([r["relevance"] or 0 for r in ans.rows] + [m["relevance"] for m in ans.moments])
    assert all(m["relevance"] >= best - 0.05 for m in ans.moments)
    assert all(r["relevance"] is None or r["relevance"] >= best - 0.05 or "words" in r["matched"] for r in ans.rows)


class CountingLlm:
    def __init__(self):
        self.calls = 0

    def structured(self, *a, **k):
        self.calls += 1
        return None


def test_with_the_index_descriptive_words_need_no_llm_parse(store, index):
    llm = CountingLlm()
    svc = ChatService(store, QuestionParser(llm), summarize=False, index=index, vector_cfg=LOOSE)
    svc.ask("Was anyone carrying a ladder?")
    assert llm.calls == 0
    svc.ask("Was anyone carrying a ladder since 2 pm?")  # a time the keyword parser can't read still goes to the LLM
    assert llm.calls == 1
    ChatService(store, QuestionParser(llm), summarize=False).ask("Was anyone carrying a ladder?")
    assert llm.calls == 2  # no index: the LLM reads the description as before


def test_distant_matches_are_shown_as_weak_and_get_no_summary(store, index):
    class Llm(CountingLlm):
        def stream_text(self, *a, **k):
            raise AssertionError("no summary for weak matches")

    svc = ChatService(store, QuestionParser(Llm()), index=index,
                      vector_cfg=VectorCfg(min_score=0.01, margin=1.0, strong_score=0.99))
    ans = svc.ask("Any helicopter landing on the roof?")
    assert ans.weak and ans.total == 0 and ans.summary is None
    assert "Nothing matches" in ans.text and "closely" in ans.text
    events = list(svc.ask_stream("Any helicopter landing on the roof?"))
    assert events[0]["weak"] and [e["type"] for e in events] == ["retrieved", "done"]


def test_similar_answer_may_cite_the_event_it_compares_against(store, index):
    ref = store.ids["fight"]

    class Llm(CountingLlm):
        def stream_text(self, messages, **k):
            assert f"[#{ref}]" in messages[0]["content"]  # the compared event is in the prompt
            yield f"Like [#{ref}], [#{store.ids['fight1']}] shows two men punching."

    svc = ChatService(store, QuestionParser(Llm()), index=index, vector_cfg=LOOSE)
    events = list(svc.ask_stream(f"Find events similar to #{ref}"))
    assert [e for e in events if e["type"] == "summary"][0]["summary"].startswith("Like")
    assert ref not in [r["event_id"] for r in events[0]["rows"]]


def test_empty_time_range_shows_the_latest_events_outside_it(store, index):
    class Llm(CountingLlm):
        def stream_text(self, *a, **k):
            raise AssertionError("no summary: these rows don't answer the question asked")

    svc = ChatService(store, QuestionParser(Llm()), index=index, vector_cfg=LOOSE)
    ans = svc.ask("Any fire in the last 10 minutes?")  # the fire was 2 hours ago
    assert ans.total == 0 and ans.earlier and ans.summary is None
    assert [r["event_id"] for r in ans.rows] == [store.ids["fire"]]
    assert "No confirmed fire event" in ans.text and "latest outside that time" in ans.text
    assert not any(s.startswith("Tell me more") for s in ans.suggestions)
    assert not svc.ask("Any arrests in the last 10 minutes?").earlier  # none at any time: nothing to show
