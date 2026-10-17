"""The officer's question goes ahead of live analysis, and camera tiles get a still so they are never black."""

import threading
import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

from aequitas.chat.answer import ChatService
from aequitas.chat.parser import QuestionParser
from aequitas.config import Settings
from aequitas.store.db import EventStore
from aequitas.understand.priority import GATE, PriorityGate
from aequitas.web import server as web


def test_background_work_waits_while_a_person_is_waiting():
    gate = PriorityGate()
    assert gate.wait_turn() < 0.05  # nobody waiting: no delay
    waited = []
    with gate.interactive():
        assert gate.busy
        t = threading.Thread(target=lambda: waited.append(gate.wait_turn()))
        t.start()
        time.sleep(0.3)
        assert t.is_alive()  # the worker is held back
    t.join(2)
    assert not gate.busy and 0.25 < waited[0] < 1.5


def test_the_hold_is_capped():
    gate = PriorityGate()
    with gate.interactive():
        assert 0.15 < gate.wait_turn(max_wait_s=0.2) < 0.6


def _app(tmp_path):
    clip = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 90))
    for i in range(20):
        vw.write(np.full((90, 160, 3), i * 10, np.uint8))
    vw.release()
    yml = tmp_path / "loc.yaml"
    yml.write_text(f"localities:\n  - id: ind\n    name: Indiranagar\n    cameras:\n"
                   f"      - {{id: ind-a, name: A, file: {clip.as_posix()}}}\n", encoding="utf-8")
    s = Settings(data_dir=tmp_path / "data", auth={"required": False}, live={"localities_file": yml})
    store = EventStore(s.db_path)
    service = ChatService(store, QuestionParser(None))
    return TestClient(web.create_app(s, service, [])), service


def test_tiles_get_a_poster_still(tmp_path):
    c, _ = _app(tmp_path)
    cam = c.get("/api/live/ind").json()["cameras"][0]
    assert cam["poster_url"].startswith("/api/live/cameras/ind-a/poster.jpg")
    r = c.get(cam["poster_url"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == 640  # small: it only has to fill a tile
    assert c.get(cam["poster_url"]).content == r.content  # cached on disk
    assert c.get("/api/live/cameras/nope/poster.jpg").status_code == 404


def test_chat_holds_live_analysis_while_it_answers(tmp_path, monkeypatch):
    c, service = _app(tmp_path)
    seen = []
    real = service.ask
    monkeypatch.setattr(service, "ask", lambda q, ctx=None: (seen.append(GATE.busy), real(q, ctx))[1])
    assert c.post("/api/ask", json={"question": "any fights today"}).status_code == 200
    assert seen == [True] and not GATE.busy
    streamed = []
    real_stream = service.ask_stream

    def stream(q, ctx=None):
        streamed.append(GATE.busy)
        yield from real_stream(q, ctx)

    monkeypatch.setattr(service, "ask_stream", stream)
    assert c.post("/api/ask/stream", json={"question": "any fights today"}).status_code == 200
    assert streamed == [True] and not GATE.busy
