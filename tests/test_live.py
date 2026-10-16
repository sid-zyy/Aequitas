"""Live CCTV dashboard backend: looping demo feeds, stored windows, one locality at a time."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from aequitas.config import CameraCfg, Settings
from aequitas.ingest.frame_source import FrameSource
from aequitas.store.db import EventStore
from aequitas.web.live import LiveManager, Locality, load_localities


def _video(path, n=10, fps=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(n):
        vw.write(np.full((48, 64, 3), i * 20 % 255, np.uint8))
    vw.release()
    return path


def test_looping_file_keeps_going_with_increasing_timestamps(tmp_path):
    cam = CameraCfg(camera_id="c", name="C", source=str(_video(tmp_path / "v.mp4", n=10)), realtime=False, loop=True)
    src = FrameSource(cam, sample_fps=10).start()
    ts = []
    for f in src.frames():
        ts.append(f.ts_ms)
        if len(ts) >= 25:  # 2.5 passes over a 10-frame clip
            src.stop()
            break
    assert len(ts) == 25 and all(b > a for a, b in zip(ts, ts[1:]))


def test_non_looping_file_ends(tmp_path):
    cam = CameraCfg(camera_id="c", name="C", source=str(_video(tmp_path / "v.mp4", n=10)), realtime=False)
    assert len(list(FrameSource(cam, sample_fps=10).start().frames())) == 10


def test_windows_are_stored_newest_first_and_pruned(tmp_path):
    store = EventStore(tmp_path / "t.db")
    for i in range(8):
        store.record_window("cam", i * 4000, (i + 1) * 4000, f"s{i}", "none", None, 800, 40, 1200, keep=5)
    recent = store.recent_windows("cam", 10)
    assert [w["summary"] for w in recent] == ["s7", "s6", "s5", "s4", "s3"]


@pytest.fixture
def localities(tmp_path):
    _video(tmp_path / "footage" / "a1.mp4")
    return [Locality.model_validate({"id": "kor", "name": "Koramangala", "cameras": [
                {"id": "kor-1", "name": "Sony World", "file": str(tmp_path / "footage" / "a1.mp4")},
                {"id": "kor-2", "name": "Forum", "file": str(tmp_path / "footage" / "missing.mp4")}]}),
            Locality.model_validate({"id": "wf", "name": "Whitefield", "cameras": [
                {"id": "wf-1", "name": "ITPL", "file": str(tmp_path / "footage" / "nope.mp4")}]})]


def test_live_manager_state_and_missing_footage(tmp_path, localities):
    store = EventStore(tmp_path / "t.db")
    live = LiveManager(Settings(data_dir=tmp_path), store, localities)
    assert {c["camera_id"] for c in store.cameras()} == {"kor-1", "kor-2", "wf-1"}
    live.start("wf")  # no footage at all: an error, not a crash
    st = live.state("wf")
    assert not st["running"] and "No footage" in st["error"]
    store.record_window("kor-1", 0, 4000, "Two people cross the road.", "none", None, 800.0, 40.0, 2500.0)
    st = live.state("kor")
    assert [c["has_footage"] for c in st["cameras"]] == [True, False]
    assert st["cameras"][0]["latest"]["summary"] == "Two people cross the road."
    assert st["cameras"][0]["behind_live_s"] == 2.5
    with pytest.raises(KeyError):
        live.state("nowhere")


def test_selecting_a_locality_stops_the_previous_one(tmp_path, localities, monkeypatch):
    started, stopped = [], []

    class FakePipeline:
        def __init__(self, s, cams, store, run_id=None):
            assert s.analysis.mode == "periodic" and all(c.loop and c.realtime for c in cams)
            self.run_id = run_id

        def run(self):
            started.append(self.run_id)

        def stop(self):
            stopped.append(self.run_id)

    import aequitas.pipeline

    monkeypatch.setattr(aequitas.pipeline, "Pipeline", FakePipeline)
    localities[1].cameras[0].file = localities[0].cameras[0].file  # give Whitefield footage too
    live = LiveManager(Settings(data_dir=tmp_path), EventStore(tmp_path / "t.db"), localities)
    live.start("kor")
    live.start("kor")  # already running: no restart
    live.start("wf")
    assert stopped == ["live-kor"] and live.active == "wf"


def test_pausing_the_ai_stops_live_analysis_until_resumed(tmp_path, localities, monkeypatch):
    started, stopped = [], []

    class FakePipeline:
        def __init__(self, s, cams, store, run_id=None):
            self.run_id = run_id

        def run(self):
            started.append(self.run_id)

        def stop(self):
            stopped.append(self.run_id)

    import aequitas.pipeline

    monkeypatch.setattr(aequitas.pipeline, "Pipeline", FakePipeline)
    live = LiveManager(Settings(data_dir=tmp_path), EventStore(tmp_path / "t.db"), localities)
    live.start("kor")
    live.set_paused(True)
    assert stopped == ["live-kor"] and live.pipeline is None and live.active == "kor"
    live.start("kor")  # the dashboard re-opening while paused starts nothing
    st = live.state("kor")
    assert st["paused"] and not st["running"] and st["error"] is None
    live.set_paused(False)  # resuming restarts the open locality
    assert live.pipeline is not None and not live.state("kor")["paused"]


def test_live_api_needs_sign_in_and_serves_only_configured_footage(tmp_path, localities, monkeypatch):
    import yaml

    from aequitas.chat.answer import ChatService
    from aequitas.chat.parser import QuestionParser
    from aequitas.web import server as web
    from aequitas.web.jobs import JobManager

    monkeypatch.setattr(JobManager, "submit", lambda self, upload_id: None)
    cfg = tmp_path / "loc.yaml"
    cfg.write_text(yaml.safe_dump({"localities": [loc.model_dump() for loc in localities]}), encoding="utf-8")
    s = Settings(data_dir=tmp_path / "data", live={"localities_file": cfg}, auth={"required": False})
    c = TestClient(web.create_app(s, ChatService(EventStore(s.db_path), QuestionParser(None)), []))
    locs = c.get("/api/live/localities").json()
    assert [x["name"] for x in locs] == ["Koramangala", "Whitefield"] and locs[0]["cameras"][1]["has_footage"] is False
    assert c.get("/api/live/cameras/kor-1/video").headers["content-type"] == "video/mp4"
    assert c.get("/api/live/cameras/kor-2/video").status_code == 404
    assert c.get("/api/live/cameras/..%2F..%2Fsecrets/video").status_code == 404
    assert c.get("/api/live/kor").json()["locality"]["name"] == "Koramangala"
    assert c.get("/api/live/nowhere").status_code == 404
    assert c.get("/api/live/ai").json() == {"paused": False}
    assert c.post("/api/live/ai", json={"paused": True}).json() == {"paused": True}
    assert c.get("/api/live/kor").json()["paused"] is True

    s2 = Settings(data_dir=tmp_path / "data2", live={"localities_file": cfg})  # auth on (default)
    c2 = TestClient(web.create_app(s2, ChatService(EventStore(s2.db_path), QuestionParser(None)), []))
    assert c2.get("/api/live/localities").status_code == 401


def test_bangalore_config_is_valid():
    locs = load_localities("config/bangalore.yaml")
    city = [loc for loc in locs if not any(c.is_live for c in loc.cameras)]
    assert len(city) >= 6 and all(len(loc.cameras) >= 3 for loc in city)
    webcam = [c for loc in locs for c in loc.cameras if c.is_live]
    assert webcam and webcam[0].source == "browser" and webcam[0].view == "close" and webcam[0].window_s == 2.0
    ids = [c.id for loc in locs for c in loc.cameras]
    assert len(ids) == len(set(ids))


def test_webcam_camera_runs_live_close_range_and_streams_mjpeg(tmp_path, monkeypatch):
    made = []

    class FakePipeline:
        def __init__(self, s, cams, store, run_id=None):
            made.extend(cams)
            self.latest_frames = {}

        def run(self):
            pass

        def stop(self):
            pass

    import aequitas.pipeline

    monkeypatch.setattr(aequitas.pipeline, "Pipeline", FakePipeline)
    loc = Locality.model_validate({"id": "demo", "name": "Demo", "cameras": [
        {"id": "cam", "name": "Webcam", "source": "0", "view": "close", "window_s": 2.0}]})
    live = LiveManager(Settings(data_dir=tmp_path), EventStore(tmp_path / "t.db"), [loc])
    assert live.has_footage("cam")  # a live source is always offered
    live.start("demo")
    assert made[0].source == "0" and not made[0].loop and made[0].view == "close" and made[0].window_s == 2.0
    st = live.state("demo")
    assert st["cameras"][0]["video_url"].endswith("/mjpeg") and st["cameras"][0]["live_source"]
    live.pipeline.latest_frames["cam"] = np.zeros((10, 10, 3), np.uint8)
    assert live.latest_frame("cam").shape == (10, 10, 3)


def test_close_view_prompt_describes_gestures_without_identities():
    from aequitas.understand.prompts import CHOICES, summarize_prompt

    p = summarize_prompt(False, view="close")
    assert "hands and arms" in p and "Never describe faces" in p
    assert not any(c in p for c in CHOICES)
    assert "hands and arms" not in summarize_prompt(False)
