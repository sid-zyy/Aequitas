"""Background analysis of uploaded videos: one job at a time, through the normal pipeline."""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from datetime import datetime, timezone

import cv2

from aequitas.config import CameraCfg, Settings, Zones
from aequitas.store.db import EventStore

log = logging.getLogger(__name__)


def probe_video(path: str) -> tuple[int, float]:
    """(duration_ms, native_fps); (0, 0) if the file cannot be read as video."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return 0, 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        ok, _ = cap.read()
        if not ok or fps <= 0:
            return 0, 0.0
        return int(frames / fps * 1000), fps
    finally:
        cap.release()


class JobManager:
    def __init__(self, settings: Settings, store: EventStore):
        self.settings = settings
        self.store = store
        self._q: queue.Queue[str] = queue.Queue()
        self._live: dict[str, dict] = {}  # upload_id -> in-memory progress
        self._pipelines: dict[str, object] = {}
        self._cancelled: set[str] = set()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker, name="upload-jobs", daemon=True)
        self._thread.start()

    def submit(self, upload_id: str) -> None:
        self._q.put(upload_id)

    def cancel(self, upload_id: str) -> bool:
        with self._lock:
            self._cancelled.add(upload_id)
            p = self._pipelines.get(upload_id)
        if p is not None:
            p.stop()
            return True
        row = self.store.upload(upload_id)
        if row and row["status"] == "queued":
            self.store.update_upload(upload_id, status="cancelled", finished_at=int(time.time() * 1000))
            return True
        return False

    def progress(self, upload_id: str) -> dict:
        with self._lock:
            live = dict(self._live.get(upload_id, {}))
            p = self._pipelines.get(upload_id)
        if p is not None and live.get("phase") != "summarising":
            live["vlm_backlog"] = len(p.queue)
            live["phase"] = p.phase
        return live

    def _camera_for(self, row: dict) -> CameraCfg:
        cam = self.store.camera(row["camera_id"])
        start_iso = datetime.fromtimestamp(row["start_ts"] / 1000, tz=timezone.utc).isoformat()
        return CameraCfg(
            camera_id=cam["camera_id"], name=cam["name"], aliases=cam["aliases"], source=row["path"],
            realtime=False, start_time=start_iso, zones=Zones.model_validate(cam["zones"] or {}),
            view=row.get("view") or "cctv",
        )

    def _worker(self) -> None:
        while True:
            upload_id = self._q.get()
            try:
                self._run(upload_id)
            except Exception as e:
                log.exception("upload %s failed", upload_id)
                self.store.update_upload(upload_id, status="failed", error=str(e)[:500],
                                         finished_at=int(time.time() * 1000))
            finally:
                with self._lock:
                    self._live.pop(upload_id, None)
                    self._pipelines.pop(upload_id, None)

    def _run(self, upload_id: str) -> None:
        from aequitas.pipeline import Pipeline
        from aequitas.understand.vlm_client import make_client

        row = self.store.upload(upload_id)
        if row is None or row["status"] != "queued" or upload_id in self._cancelled:
            return
        s = self.settings.model_copy(deep=True)
        s.analysis.mode = row["mode"]
        if row["vlm_model"]:
            s.vlm.model = row["vlm_model"]
        if row.get("kind") == "bodycam":  # the review timeline needs every window of the recording
            s.analysis.keep_windows = 1_000_000

        use_vlm, note = True, None
        try:
            vlm_ok = make_client(s.vlm).ping()
        except ValueError:
            vlm_ok = False
        if not vlm_ok:
            if s.analysis.mode == "periodic":
                raise RuntimeError(f"VLM '{s.vlm.model}' is not available, and periodic mode needs it")
            use_vlm = False
            note = f"VLM '{s.vlm.model}' unavailable: events were recorded without AI verification"

        started_ms = int(time.time() * 1000)
        with self._lock:
            self._live[upload_id] = {"frames_done": 0}

        def on_frame(_camera_id: str, n: int) -> None:
            with self._lock:
                self._live[upload_id]["frames_done"] = n

        self.store.update_upload(upload_id, status="processing", error=note, started_at=started_ms)
        pipeline = Pipeline(s, [self._camera_for(row)], self.store, use_vlm=use_vlm, on_frame=on_frame,
                            run_id=upload_id)
        with self._lock:
            self._pipelines[upload_id] = pipeline
            if upload_id in self._cancelled:
                pipeline.stop()
        pipeline.run()

        cancelled = upload_id in self._cancelled
        if row.get("kind") == "bodycam" and use_vlm and not cancelled:
            from aequitas.web.bodycam import make_summary

            with self._lock:
                self._live[upload_id]["phase"] = "summarising"
            try:
                make_summary(s, self.store, self.store.upload(upload_id))
            except Exception:
                log.exception("recording summary failed for %s", upload_id)
        with self._lock:
            done = self._live.get(upload_id, {}).get("frames_done", 0)
        self.store.update_upload(
            upload_id, status="cancelled" if cancelled else "done", frames_done=done,
            events_found=self.store.count_events_since(row["camera_id"], started_ms),
            finished_at=int(time.time() * 1000),
        )


def frames_expected(duration_ms: int, sample_fps: float) -> int:
    return max(1, math.ceil(duration_ms / 1000 * sample_fps))
