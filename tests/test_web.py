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
    settings = Settings(data_dir=tmp_path / "data", auth={"required": False},
                        live={"localities_file": tmp_path / "no-localities.yaml"})  # auth: tests/test_roles.py
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


def test_index_and_spa_routes_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Aequitas" in r.text
    if web.DIST.joinpath("index.html").is_file():  # the React app: client-side routes return the app
        assert client.get("/report").text == r.text and client.get("/reports").status_code == 200
        assert client.get("/api/nope").status_code == 404  # unknown API paths are not the app
        assert client.get("/../pyproject.toml").text == r.text  # no files outside the build
    else:
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


# ---------- bodycam review ----------

def test_bodycam_upload_timeline_review_and_report(client):
    vid = _video(client.tmp / "body.mp4", seconds=3)
    r = client.post("/api/bodycam", files={"file": ("shift-cam.mp4", vid.read_bytes())},
                    data={"officer": "PC 4412", "recorded_at": "2026-10-16T10:00:00"})
    assert r.status_code == 200, r.text
    rec = r.json()
    assert rec["officer"] == "PC 4412" and rec["view"] == "close" and rec["camera_name"] == "Bodycam · PC 4412"
    up = client.store.upload(rec["id"])
    assert up["kind"] == "bodycam" and up["mode"] == "periodic"
    assert [x["id"] for x in client.get("/api/bodycam").json()] == [rec["id"]]
    assert client.get("/api/bodycam/nope").status_code == 404

    # windows and an event as the pipeline would store them; times come back relative to the recording
    cam, t0 = up["camera_id"], up["start_ts"]
    client.store.record_window(cam, t0, t0 + 4000, "An officer talks to a man.", "none", None, 900, 80, 1000)
    client.store.record_window(cam, t0 + 4000, t0 + 8000, "The officer handcuffs the man.", "arrest", None,
                               900, 80, 1000, source="vlm")
    eid = client.store.insert_event(EventRecord("b1", cam, t0 + 4000, t0 + 8000, "arrest", None, "unverified", 0.4,
                                                0.0, "arrest", "The officer handcuffs the man.", None, [], "k.jpg", {}))
    d = client.get(f"/api/bodycam/{rec['id']}").json()
    assert [w["t"] for w in d["windows"]] == [0.0, 4.0] and d["windows"][1]["label"] == "arrest"
    assert d["events"][0]["t"] == 4.0 and d["events"][0]["review"] is None
    assert d["video_url"] == f"/api/uploads/{rec['id']}/video"

    # an officer confirms; the AI's own status is kept
    ok = client.post(f"/api/bodycam/{rec['id']}/events/{eid}/review", json={"decision": "confirmed"}).json()
    assert ok["review"] == "confirmed" and ok["status"] == "unverified"
    assert client.post(f"/api/bodycam/{rec['id']}/events/999/review", json={"decision": "confirmed"}).status_code == 404
    assert client.post(f"/api/bodycam/{rec['id']}/events/{eid}/review", json={"decision": "maybe"}).status_code == 422

    rep = client.get(f"/api/bodycam/{rec['id']}/report")
    assert rep.status_code == 200 and "attachment" in rep.headers["content-disposition"]
    assert "Confirmed by" in rep.text and "handcuffs" in rep.text and "AI-generated draft" in rep.text
    assert "0 of 1 are not reviewed" in rep.text
    cleared = client.post(f"/api/bodycam/{rec['id']}/events/{eid}/review", json={"decision": None}).json()
    assert cleared["review"] is None and cleared["reviewed_by"] is None


def test_bodycam_report_escapes_text():
    from aequitas.web.bodycam import report_html

    up = {"upload_id": "u1", "filename": "<script>x</script>.mp4", "officer": None, "start_ts": 1_792_000_000_000,
          "duration_ms": 8000, "vlm_model": "gemma4:e2b"}
    tl = {"windows": [{"t": 0, "label": "none", "summary": "<b>hi</b>"}], "events": []}
    out = report_html(up, tl, 3.2)
    assert "<script>x" not in out and "&lt;script&gt;" in out and "&lt;b&gt;hi" in out


def test_bodycam_jobs_use_close_view_and_keep_every_window(tmp_path):
    from aequitas.web.jobs import JobManager as JM

    jm = JM.__new__(JM)
    store = EventStore(tmp_path / "j.db")
    store.upsert_camera(CameraCfg(camera_id="bodycam-x", name="Bodycam", source="bodycam"))
    jm.store = store
    cam = jm._camera_for({"camera_id": "bodycam-x", "start_ts": 0, "path": "a.mp4", "view": "close"})
    assert cam.view == "close"
    assert jm._camera_for({"camera_id": "bodycam-x", "start_ts": 0, "path": "a.mp4"}).view == "cctv"
    for i in range(5):
        store.record_window("bodycam-x", i * 4000, i * 4000 + 4000, "s", "none", None, 1, 1, 1, keep=3)
    assert len(store.windows_for("bodycam-x")) == 3  # the default keeps the newest; bodycam jobs raise the limit
