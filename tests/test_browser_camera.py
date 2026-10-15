"""Browser camera source: frames pushed over HTTP reach the pipeline; nothing opens a device."""

import time

import numpy as np

from aequitas.config import CameraCfg
from aequitas.ingest.frame_source import PUSH_SOURCES, FrameSource, push_frame, source_kind


def test_pushed_frames_reach_the_source_and_stop_unregisters():
    assert source_kind("browser") == "push" and source_kind("0") == "webcam"
    src = FrameSource(CameraCfg(camera_id="bcam", name="b", source="browser"), 4).start()
    for _ in range(50):
        if "bcam" in PUSH_SOURCES:
            break
        time.sleep(0.01)
    assert src.live and push_frame("bcam", np.zeros((8, 8, 3), np.uint8))
    assert push_frame("bcam", np.ones((8, 8, 3), np.uint8))  # live: newest wins, the older one is dropped
    frames = src.frames()
    f = next(frames)
    assert f.camera_id == "bcam" and f.image.max() == 1 and src.dropped == 1
    src.stop()
    assert next(frames, None) is None  # stream ends
    assert "bcam" not in PUSH_SOURCES and not push_frame("bcam", np.zeros((8, 8, 3), np.uint8))


def test_frame_endpoint_rejects_unknown_cameras_and_non_images(tmp_path):
    from fastapi.testclient import TestClient

    from aequitas.chat.answer import ChatService
    from aequitas.chat.parser import QuestionParser
    from aequitas.config import Settings
    from aequitas.store.db import EventStore
    from aequitas.web import server as web

    loc = tmp_path / "loc.yaml"
    loc.write_text("localities:\n  - id: demo\n    name: Demo\n    cameras:\n"
                   "      - {id: demo-webcam, name: Webcam, source: browser, view: close}\n", encoding="utf-8")
    s = Settings(data_dir=tmp_path / "d", auth={"required": False}, live={"localities_file": loc})
    c = TestClient(web.create_app(s, ChatService(EventStore(s.db_path), QuestionParser(None)), []))
    assert c.post("/api/live/cameras/nope/frame", content=b"x").status_code == 404
    assert c.post("/api/live/cameras/demo-webcam/frame", content=b"not a jpeg").status_code == 400
    import cv2

    ok, jpg = cv2.imencode(".jpg", np.zeros((16, 16, 3), np.uint8))
    r = c.post("/api/live/cameras/demo-webcam/frame", content=jpg.tobytes(), headers={"Content-Type": "image/jpeg"})
    assert r.status_code == 200 and r.json() == {"accepted": False}  # analysis not started
    st = c.get("/api/live/demo").json()
    assert st["cameras"][0]["browser_camera"] is True
