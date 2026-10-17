from datetime import datetime, timedelta

import numpy as np
import pytest

from aequitas.chat.answer import ChatService
from aequitas.chat.parser import LlmFilter, QuestionParser, finalize, keyword_filter, resolve_cameras
from aequitas.config import CameraCfg, FusionCfg, Zones
from aequitas.ingest.ring_buffer import RingBuffer
from aequitas.store.db import EventRecord, EventStore, Timings
from aequitas.store.query import QueryFilter
from aequitas.types import Candidate, Category, Status
from aequitas.understand.fusion import fuse
from aequitas.understand.keyframes import make_grid
from aequitas.understand.prompts import VlmAnswer

CAMS = [
    CameraCfg(camera_id="cam1", name="Lobby entrance", aliases=["camera 1", "entrance"], source="a.mp4"),
    CameraCfg(camera_id="cam2", name="Corridor", aliases=["camera 2", "hallway"], source="b.mp4"),
]
NOW = datetime(2026, 10, 15, 15, 0).astimezone()


def ms(dt):
    return int(dt.timestamp() * 1000)


@pytest.fixture
def store(tmp_path):
    s = EventStore(tmp_path / "t.db")
    for c in CAMS:
        s.upsert_camera(c)

    def add(cam, cat, when, status="confirmed", summary="", subtype=None):
        s.insert_event(EventRecord(
            event_uid=f"{cam}-{cat}-{ms(when)}-{status}", camera_id=cam, start_ts=ms(when),
            end_ts=ms(when) + 5000, category=cat, subtype=subtype, status=status, confidence=0.8,
            trigger_score=0.5, vlm_category=cat, summary=summary, zone=None, track_ids=[1],
            keyframe_path="x.jpg", model_versions={},
        ), Timings(detect_ms=10, vlm_ms=2000))

    add("cam2", "tailgating", NOW - timedelta(hours=2), summary="Two people enter through the door together.")
    add("cam2", "tailgating", NOW - timedelta(days=1))
    add("cam1", "tailgating", NOW - timedelta(hours=1))
    add("cam2", "fighting", NOW - timedelta(hours=3), summary="Two men shove each other near a red jacket rack.")
    add("cam2", "tailgating", NOW - timedelta(minutes=30), status="rejected")
    add("cam1", "suspicious_activity", NOW - timedelta(hours=4), subtype="abandoned_object",
        summary="A black backpack is left on the floor.")
    return s


def test_query_by_category_camera_and_time(store):
    f = QueryFilter(categories=[Category.TAILGATING], camera_ids=["cam2"],
                    start=NOW.replace(hour=0, minute=0), end=NOW)
    rows = store.query(f)
    assert len(rows) == 1 and rows[0]["camera_name"] == "Corridor"
    f.statuses = list(Status)
    assert store.count(f) == 2


def test_full_text_search(store):
    rows = store.query(QueryFilter(text="red jacket"))
    assert [r["category"] for r in rows] == ["fighting"]
    assert store.query(QueryFilter(text="'; DROP TABLE events; --")) == []


def test_resolve_cameras():
    cams = [{"camera_id": c.camera_id, "name": c.name, "aliases": c.aliases} for c in CAMS]
    assert resolve_cameras(["camera 2"], cams) == (["cam2"], [])
    assert resolve_cameras(["the entrance"], cams) == (["cam1"], [])
    assert resolve_cameras(["Cam 1", "hallway"], cams) == (["cam1", "cam2"], [])
    assert resolve_cameras(["camera 9"], cams) == ([], ["camera 9"])


def test_keyword_parser_example_question():
    cams = [{"camera_id": c.camera_id, "name": c.name, "aliases": c.aliases} for c in CAMS]
    lf = keyword_filter("Show all tailgating events from camera 2 today", NOW, cams)
    res = finalize(lf, NOW, cams)
    f = res.filter
    assert f.categories == [Category.TAILGATING]
    assert f.camera_ids == ["cam2"]
    assert f.start == NOW.replace(hour=0, minute=0, second=0, microsecond=0)
    assert f.intent == "list"


def test_keyword_parser_count_and_subtype():
    lf = keyword_filter("How many bags were left unattended this week?", NOW, [])
    assert lf.intent == "count"
    assert "suspicious_activity" in lf.categories and "abandoned_object" in lf.subtypes


def test_finalize_clips_future_and_swaps_reversed_times():
    lf = LlmFilter(start=(NOW + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M"),
                   end=(NOW - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M"))
    f = finalize(lf, NOW, []).filter
    assert f.start < f.end <= NOW


class FakeLlm:
    """Returns `result` for filter requests and `summary` for summary requests."""

    def __init__(self, result, summary=None):
        self.result = result
        self.summary = summary

    def structured(self, messages, model_cls, retries=1):
        return self.summary if model_cls.__name__ == "GroundedSummary" else self.result


def test_chat_service_end_to_end(store, monkeypatch):
    llm = FakeLlm(LlmFilter(categories=["tailgating"], cameras=["camera 2"],
                            start=NOW.strftime("%Y-%m-%dT00:00"), end=NOW.strftime("%Y-%m-%dT%H:%M")))
    parser = QuestionParser(llm)
    monkeypatch.setattr(parser, "now", lambda: NOW)
    ans = ChatService(store, parser).ask("Show all tailgating events from camera 2 today")
    assert ans.total == 1
    assert "Two people enter" in ans.text
    assert ans.parsed.method == "keywords"  # fully read by the keywords: no model call needed
    ans = ChatService(store, parser).ask("Was someone sneaking in behind another person on camera 2 today?")
    assert ans.parsed.method == "llm" and ans.total == 1  # words the keywords can't map go to the LLM


def test_chat_falls_back_to_keywords_when_llm_fails(store, monkeypatch):
    parser = QuestionParser(FakeLlm(None))
    monkeypatch.setattr(parser, "now", lambda: NOW)
    ans = ChatService(store, parser).ask("how many tailgating events on camera 2 today")
    assert ans.parsed.method == "keywords"
    assert ans.text.startswith("1 confirmed tailgating event on")


def test_llm_hallucinations_are_reconciled_with_the_question(store, monkeypatch):
    # Seen live: asked about falls, the model returned every camera and no category.
    llm = FakeLlm(LlmFilter(cameras=["Corridor", "Lobby entrance"], intent="list"))
    parser = QuestionParser(llm)
    monkeypatch.setattr(parser, "now", lambda: NOW)
    f = parser.parse("How many people fell this week?", [
        {"camera_id": c.camera_id, "name": c.name, "aliases": c.aliases} for c in CAMS]).filter
    assert f.camera_ids == []
    assert f.categories == [Category.PERSON_DOWN]
    assert f.intent == "count"
    assert f.start is not None


def test_llm_intent_and_invented_times_are_overruled(monkeypatch):
    # Seen with gemma4:e2b (config/chat_eval.yaml): yes/no questions read as counts, times invented or changed.
    cams = [{"camera_id": c.camera_id, "name": c.name, "aliases": c.aliases} for c in CAMS]
    llm = FakeLlm(LlmFilter(categories=["fighting"], intent="count", start="2026-10-15T14:00",
                            end="2026-10-15T15:00"))
    parser = QuestionParser(llm)
    f = parser.parse("Any fights in the last 30 minutes?", cams, now=NOW).filter
    assert f.intent == "list"  # no 'how many'
    assert f.end - f.start == timedelta(minutes=30)  # the keyword reading of the time wins
    f = parser.parse("Any fights on camera 2?", cams, now=NOW).filter
    assert f.start is None and f.end is None  # no time asked: the invented one is dropped
    llm.result = LlmFilter(categories=["fighting"], start="2026-10-15T14:00", end="2026-10-15T15:00")
    f = parser.parse("fights since 2 pm", cams, now=NOW).filter
    assert f.start is not None  # a time only the LLM can read is kept


def test_unknown_camera_is_reported(store, monkeypatch):
    parser = QuestionParser(FakeLlm(LlmFilter(cameras=["parking lot"])))
    monkeypatch.setattr(parser, "now", lambda: NOW)
    assert "parking lot" in ChatService(store, parser).ask("anything in the parking lot?").text


# ---------- fusion and keyframes ----------

CAND = Candidate("cam1", Category.FIGHTING, None, 0, 3000, 0.5, (1, 2))


@pytest.mark.parametrize("vlm_cat,status,conf", [
    ("fighting", Status.CONFIRMED, 0.8),
    ("none", Status.REJECTED, 0.1),
    ("person_down", Status.DISPUTED, 0.2),
])
def test_fusion(vlm_cat, status, conf):
    out = fuse(CAND, VlmAnswer(category=vlm_cat, summary="x"), FusionCfg())
    assert out.status == status
    assert out.confidence == pytest.approx(conf)


def test_fusion_without_vlm_answer():
    out = fuse(CAND, None, FusionCfg())
    assert out.status == Status.DISPUTED and out.summary is None


def test_ring_buffer_and_grid():
    rb = RingBuffer(seconds=5, max_width=320)
    img = np.zeros((480, 640, 3), np.uint8)
    for i in range(40):
        rb.push(i * 250, img, {1: (100, 100, 200, 300), 2: (300, 100, 400, 300)})
    assert rb._frames[0].ts_ms >= 39 * 250 - 5000
    picked = rb.pick(onset_ms=0, trigger_ms=39 * 250)  # onset older than the buffer: clamp
    assert len(picked) == 4 and picked[0].ts_ms == rb._frames[0].ts_ms and picked[-1].ts_ms == 39 * 250
    grid = make_grid(picked, CAND, Zones(), tile_px=224)
    assert grid.shape == (448, 448, 3)
    assert grid[..., 2].max() == 255  # red highlight boxes were drawn
