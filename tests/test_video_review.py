"""Citizen-report videos reviewed like bodycam recordings, and the per-video chat (fake LLM, synthetic video)."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from aequitas.chat.answer import ChatService
from aequitas.chat.parser import QuestionParser
from aequitas.chat.video_qa import answer_video, cited_times, video_prompt
from aequitas.config import LlmEndpoint, Settings
from aequitas.store.db import EventRecord, EventStore
from aequitas.web import server as web
from aequitas.web.jobs import JobManager


class FakeLlm:
    def __init__(self, answer="The man grabs her arm at [00:04], then runs off at [09:59]."):
        self.cfg = LlmEndpoint()
        self.model = self.cfg.model
        self.answer, self.prompts = answer, []

    def stream_text(self, messages, max_tokens=256):
        self.prompts.append(messages[0]["content"])
        if self.answer is None:
            raise ConnectionError("model offline")
        yield from self.answer.split(" ")[:1]
        yield from (" " + w for w in self.answer.split(" ")[1:])


@pytest.fixture
def c(tmp_path, monkeypatch):
    monkeypatch.setattr(JobManager, "submit", lambda self, upload_id: None)
    s = Settings(data_dir=tmp_path / "data", auth={"required": False}, authenticity={"frame_model": None},
                 live={"localities_file": tmp_path / "none.yaml"})
    store = EventStore(s.db_path)
    client = TestClient(web.create_app(s, ChatService(store, QuestionParser(None)), []))
    client.store, client.tmp = store, tmp_path
    client.app.state.translator = FakeLlm()
    return client


def _report(c, description="A man followed a woman and grabbed her arm at the bus stop."):
    path = c.tmp / "r.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(30):
        vw.write(np.full((48, 64, 3), i * 7 % 255, np.uint8))
    vw.release()
    with open(path, "rb") as f:
        r = c.post("/api/citizen/reports", files={"file": ("r.mp4", f, "video/mp4")},
                   data={"description": description, "consent": "true"})
    assert r.status_code == 200, r.text
    rid = r.json()["report_id"]
    rep = c.get(f"/api/reports/{rid}").json()
    up = c.store.upload(rep["upload_id"])
    cam, t0 = up["camera_id"], up["start_ts"]
    c.store.record_window(cam, t0, t0 + 4000, "Two people stand at a bus stop.", "none", None, 900, 80, 1000)
    c.store.record_window(cam, t0 + 4000, t0 + 8000, "A man grabs a woman's arm.", "harassment", None, 900, 80, 1000)
    eid = c.store.insert_event(EventRecord("r1", cam, t0 + 4000, t0 + 8000, "harassment", None, "unverified", 0.5,
                                           0.0, "harassment", "A man grabs a woman's arm.", None, [], "k.jpg", {}))
    return rid, eid


def test_report_video_has_a_bodycam_style_timeline(c):
    rid, eid = _report(c)
    d = c.get(f"/api/reports/{rid}/analysis").json()
    assert [w["t"] for w in d["windows"]] == [0.0, 4.0] and d["windows"][1]["label"] == "harassment"
    assert d["events"][0]["id"] == eid and d["events"][0]["keyframe_url"] == f"/api/events/{eid}/keyframe"
    assert d["video_url"].startswith("/api/uploads/") and d["duration_s"] == pytest.approx(3.0, abs=0.2)
    assert c.get("/api/reports/nope/analysis").status_code == 404


def test_officer_confirms_a_moment_in_a_report(c):
    rid, eid = _report(c)
    ok = c.post(f"/api/reports/{rid}/events/{eid}/review", json={"decision": "confirmed"}).json()
    assert ok["review"] == "confirmed"
    assert c.post(f"/api/reports/{rid}/events/9999/review", json={"decision": "confirmed"}).status_code == 404


def test_report_chat_answers_from_this_video_only(c):
    rid, eid = _report(c)
    c.post(f"/api/reports/{rid}/events/{eid}/review", json={"decision": "dismissed"})
    r = c.post(f"/api/reports/{rid}/ask", json={"question": "What happened to the woman?",
                                                  "history": [{"q": "Who is there?", "a": "Two people [00:00]."}]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["cited_s"] == [4.0]  # [09:59] is past the end of the 3 s video, so it is not offered
    prompt = c.app.state.translator.prompts[0]
    assert "A man grabs a woman's arm." in prompt and "dismissed by an officer" in prompt
    assert "NOT evidence" in prompt and "grabbed her arm at the bus stop" in prompt
    assert "Earlier question: Who is there?" in prompt and "never name, identify" in prompt


def test_report_chat_reports_an_offline_model(c):
    rid, _ = _report(c)
    c.app.state.translator = FakeLlm(answer=None)
    assert c.post(f"/api/reports/{rid}/ask", json={"question": "What happened?"}).status_code == 503
    assert c.post(f"/api/reports/{rid}/ask", json={"question": ""}).status_code == 422


def test_cited_times_keep_order_and_drop_impossible_ones():
    assert cited_times("at [00:04] and [00:04], then [01:02] and [1:00:00]", 70) == [4.0, 62.0]


def test_prompt_without_analysis_says_so():
    p = video_prompt("Anything?", [], [], None)
    assert "(no window was analysed)" in p and "Question: Anything?" in p


def test_answer_video_times_the_call():
    res = answer_video(FakeLlm("Nothing happens [00:01]."), "q", [], [], None, 10.0)
    assert res["answer"] == "Nothing happens [00:01]." and res["cited_s"] == [1.0] and res["ms"] >= 0
