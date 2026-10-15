"""Web API tests. Analysis jobs are stubbed so no models load."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from aequitas.chat.answer import ChatService
from aequitas.chat.parser import QuestionParser
from aequitas.config import CameraCfg, Settings
from aequitas.store.db import EventRecord, EventStore
from aequitas.web import server as web
from aequitas.web.jobs import JobManager


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(JobManager, "submit", lambda self, upload_id: None)
    settings = Settings(data_dir=tmp_path / "data", auth={"required": False})  # auth: tests/test_roles.py
    store = EventStore(settings.db_path)
    cams = [CameraCfg(camera_id="cam2", name="Corridor", aliases=["camera 2"], source="x.mp4")]
    app = web.create_app(settings, ChatService(store, QuestionParser(None)), cams)
    kf = tmp_path / "k.jpg"
    cv2.imwrite(str(kf), np.zeros((20, 20, 3), np.uint8))
    import time

    now = int(time.time() * 1000)
    store.insert_event(EventRecord("e1", "cam2", now - 60_000, now - 55_000, "tailgating", None, "confirmed",
                                   0.8, 0.6, "tailgating", "Two people enter together.", None, [1, 2],
                                   kf.as_posix(), {"vlm": "ollama:qwen2.5vl:3b"}))
    c = TestClient(app)
    c.store = store
    c.tmp = tmp_path
    return c


def _video(path, seconds=2, fps=10):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(seconds * fps):
        vw.write(np.full((48, 64, 3), i * 5 % 255, np.uint8))
    vw.release()
    return path


def test_index_and_static_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Aequitas" in r.text
    assert client.get("/app.js").status_code == 200


def test_events_list_detail_and_keyframe(client):
    r = client.get("/api/events").json()
    assert r["total"] == 1
    ev = r["events"][0]
    assert ev["camera_name"] == "Corridor" and ev["keyframe_url"] == f"/api/events/{ev['id']}/keyframe"
    assert client.get(ev["keyframe_url"]).headers["content-type"] == "image/jpeg"
    assert client.get("/api/events/999").status_code == 404
    assert client.get("/api/events?status=bogus").status_code == 422


def test_ask_returns_grounded_events(client):
    r = client.post("/api/ask", json={"question": "show tailgating on camera 2 today"}).json()
    assert r["total"] == 1 and r["events"][0]["summary"] == "Two people enter together."
    assert r["parsed_with"] == "keywords"
    assert client.post("/api/ask", json={"question": ""}).status_code == 422


def test_upload_validation(client):
    bad = client.post("/api/uploads", files={"file": ("notes.txt", b"hello")}, data={"camera_id": "cam2"})
    assert bad.status_code == 400
    fake = client.post("/api/uploads", files={"file": ("clip.mp4", b"not really a video")}, data={"camera_id": "cam2"})
    assert fake.status_code == 400 and "could not be read" in fake.json()["detail"]
    nocam = client.post("/api/uploads", files={"file": ("clip.mp4", b"x")}, data={"camera_id": "nope"})
    assert nocam.status_code == 400


def test_upload_creates_camera_job_and_links_video_to_events(client):
    path = _video(client.tmp / "clip.mp4")
    with path.open("rb") as f:
        r = client.post("/api/uploads", files={"file": ("Dock cam.mp4", f, "video/mp4")},
                        data={"new_camera_name": "Loading dock", "mode": "periodic"})
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["camera_id"] == "cam-loading-dock" and job["status"] == "queued" and job["mode"] == "periodic"
    assert job["frames_total"] == 8  # 2 s at 4 fps
    assert any(c["camera_id"] == "cam-loading-dock" for c in client.get("/api/cameras").json())

    # An event inside the uploaded clip's time span links back to the video at the right offset.
    store = client.store
    store.insert_event(EventRecord("e2", "cam-loading-dock", job["start_ts"] + 1500, job["start_ts"] + 1800,
                                   "person_down", None, "confirmed", 0.9, 0.5, "person_down", "x", None, [],
                                   "missing.jpg", {}))
    ev = next(e for e in client.get("/api/events?status=all").json()["events"] if e["camera_id"] == "cam-loading-dock")
    assert ev["video"]["url"] == f"/api/uploads/{job['id']}/video"
    assert ev["video"]["offset_s"] == pytest.approx(1.5)
    assert client.get(ev["video"]["url"]).status_code == 200
    assert client.get(ev["keyframe_url"]).status_code == 404  # missing file -> 404, not a server error

    assert client.post(f"/api/uploads/{job['id']}/cancel").json() == {"cancelled": True}
    assert client.get("/api/uploads").json()[0]["status"] == "cancelled"


def test_stats(client):
    s = client.get("/api/stats").json()
    assert s["by_category"] == {"tailgating": 1} and s["confirmed_today"] >= 0 and s["cameras"] == 1
