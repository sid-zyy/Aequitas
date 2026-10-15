"""Requirements added at the 2026-10-15 team meeting: optional YOLO (periodic mode), model-independent
local LLMs, and grounded answers with event references."""

import base64
from datetime import datetime, timedelta

import numpy as np
import pytest

from aequitas.chat.answer import ChatService, GroundedSummary, grounded_summary
from aequitas.chat.parser import LlmFilter, QuestionParser
from aequitas.config import ChatCfg, LlmEndpoint, VlmCfg, Zones
from aequitas.ingest.ring_buffer import BufferedFrame
from aequitas.store.db import EventRecord, EventStore
from aequitas.store.query import QueryFilter
from aequitas.types import Status
from aequitas.understand.keyframes import make_grid
from aequitas.understand.periodic import WindowAggregator
from aequitas.understand.prompts import VlmAnswer, classify_prompt
from aequitas.understand.vlm_client import is_local_host, make_client, openai_payload

from test_store_chat import FakeLlm, NOW, store  # noqa: F401  (fixture)


def ans(cat, summary="s"):
    return VlmAnswer(category=cat, summary=summary)


# ---------- periodic mode ----------

def test_windows_merge_and_confirm_on_second_vote():
    agg = WindowAggregator(confirm_votes=2, max_gap_ms=10_000)
    d1 = agg.add("cam1", 0, 4000, ans("fighting", "Two people grapple."))
    assert d1.action == "new" and d1.status == Status.UNVERIFIED and d1.confidence == 0.4
    d2 = agg.add("cam1", 4000, 8000, ans("fighting"))
    assert d2.action == "update" and d2.event.event_uid == d1.event.event_uid
    assert d2.status == Status.CONFIRMED and d2.confidence == pytest.approx(0.7)
    d3 = agg.add("cam1", 8000, 12000, ans("fighting"))
    assert d3.confidence == pytest.approx(0.75) and d3.event.end_ms == 12000


def test_none_or_other_category_closes_event_and_failures_are_ignored():
    agg = WindowAggregator()
    first = agg.add("cam1", 0, 4000, ans("person_down"))
    assert agg.add("cam1", 4000, 8000, None).action == "none"  # VLM error: event stays open
    assert agg.add("cam1", 8000, 12000, ans("person_down")).action == "update"
    assert agg.add("cam1", 12000, 16000, ans("none")).action == "none"
    again = agg.add("cam1", 16000, 20000, ans("person_down"))
    assert again.action == "new" and again.event.event_uid != first.event.event_uid
    other = agg.add("cam1", 20000, 24000, ans("fighting"))
    assert other.action == "new"


def test_cameras_are_independent_and_gaps_split_events():
    agg = WindowAggregator(max_gap_ms=5000)
    a = agg.add("cam1", 0, 4000, ans("tailgating"))
    assert agg.add("cam2", 0, 4000, ans("tailgating")).event.event_uid != a.event.event_uid
    assert agg.add("cam1", 20_000, 24_000, ans("tailgating")).action == "new"


def test_unverified_status_is_stored_and_hidden_by_default(tmp_path):
    from aequitas.config import CameraCfg

    s = EventStore(tmp_path / "p.db")
    s.upsert_camera(CameraCfg(camera_id="cam1", name="Lobby", source="a.mp4"))
    s.insert_event(EventRecord("u1", "cam1", 0, 4000, "fighting", None, "unverified", 0.4, 0.0,
                               "fighting", "x", None, [], "k.jpg", {}))
    assert s.count(QueryFilter()) == 0
    assert s.count(QueryFilter(statuses=list(Status))) == 1
    s.update_event("u1", 8000, "confirmed", 0.7)
    assert s.query(QueryFilter())[0]["end_ts"] == 8000


def test_grid_without_candidate_and_unhighlighted_prompt():
    img = np.full((240, 320, 3), 50, np.uint8)
    frames = [BufferedFrame(i * 1000, img, 1.0, {}) for i in range(4)]
    grid = make_grid(frames, None, Zones(), tile_px=128)
    assert grid.shape == (256, 256, 3)
    assert "Red boxes" not in classify_prompt(highlighted=False)
    assert "Red boxes" in classify_prompt(highlighted=True)


# ---------- model independence, local only ----------

@pytest.mark.parametrize("url,ok", [
    ("http://127.0.0.1:11434", True),
    ("http://localhost:8080/v1", True),
    ("http://192.168.1.20:8000", True),
    ("http://[::1]:11434", True),
    ("https://api.openai.com/v1", False),
    ("https://generativelanguage.googleapis.com", False),
])
def test_only_local_hosts_allowed(url, ok):
    assert is_local_host(url) is ok


def test_remote_host_refused_unless_allowed():
    with pytest.raises(ValueError):
        make_client(LlmEndpoint(backend="openai", host="https://api.openai.com/v1", model="x"))
    assert make_client(LlmEndpoint(backend="openai", host="https://my-box.example.com", model="x",
                                   allow_remote=True)).base == "https://my-box.example.com/v1"


def test_openai_payload_carries_images_and_schema():
    p = openai_payload("m", [{"role": "user", "content": "hi", "images": [b"\xff\xd8jpeg"]}],
                       {"type": "object"}, "VlmAnswer")
    parts = p["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "hi"}
    assert parts[1]["image_url"]["url"] == "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8jpeg").decode()
    assert p["response_format"]["json_schema"]["name"] == "VlmAnswer"
    plain = openai_payload("m", [{"role": "user", "content": "hi"}], {}, "X")
    assert plain["messages"][0]["content"] == "hi"


def test_chat_endpoint_falls_back_to_vlm_and_overrides_independently():
    vlm = VlmCfg(model="gemma4:e4b")
    assert ChatCfg().endpoint(vlm).model == "gemma4:e4b"
    ep = ChatCfg(backend="openai", host="http://127.0.0.1:8080", model="llama-3.2-3b").endpoint(vlm)
    assert (ep.backend, ep.host, ep.model) == ("openai", "http://127.0.0.1:8080", "llama-3.2-3b")


# ---------- grounded responses ----------

def _rows(store):
    return store.query(QueryFilter(limit=100))


def test_grounded_summary_accepts_only_retrieved_ids(store):  # noqa: F811
    rows = _rows(store)
    ids = [r["event_id"] for r in rows]
    good = FakeLlm(None, GroundedSummary(answer=f"One event [#{ids[0]}].", cited_event_ids=[ids[0]]))
    assert grounded_summary(good, "q", rows, None) == f"One event [#{ids[0]}]."
    bad = FakeLlm(None, GroundedSummary(answer="Event [#999] happened.", cited_event_ids=[999]))
    assert grounded_summary(bad, "q", rows, None) is None
    uncited = FakeLlm(None, GroundedSummary(answer="Something happened.", cited_event_ids=[]))
    assert grounded_summary(uncited, "q", rows, None) is None


def test_answer_has_summary_and_event_references(store, monkeypatch):  # noqa: F811
    lf = LlmFilter(categories=["tailgating"], cameras=["camera 2"],
                   start=NOW.strftime("%Y-%m-%dT00:00"), end=NOW.strftime("%Y-%m-%dT%H:%M"))
    row_id = store.query(QueryFilter(camera_ids=["cam2"], start=NOW - timedelta(hours=3), end=NOW,
                                     categories=["tailgating"]))[0]["event_id"]
    llm = FakeLlm(lf, GroundedSummary(answer=f"One tailgating event at the door [#{row_id}].",
                                      cited_event_ids=[row_id]))
    parser = QuestionParser(llm)
    monkeypatch.setattr(parser, "now", lambda: NOW)
    out = ChatService(store, parser).ask("Show all tailgating events from camera 2 today")
    assert out.summary and f"[#{row_id}]" in out.summary
    assert f"- [#{row_id}]" in out.text
    assert out.text.index(out.summary) < out.text.index("Found 1")
