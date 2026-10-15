"""One interface for RTSP, webcam and file sources. Reading runs in its own thread so a slow
consumer never stalls the stream: live sources keep only the newest frame, files are either
paced at native speed (simulated live) or read losslessly for evaluation."""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

import cv2

from aequitas.config import CameraCfg
from aequitas.types import Frame

log = logging.getLogger(__name__)
_END = object()
# source "browser": frames arrive over HTTP from the page's own camera (getUserMedia), so the browser asks the
# user for permission and the server never opens a device. camera_id -> the running FrameSource.
PUSH_SOURCES: dict[str, "FrameSource"] = {}
_push_lock = threading.Lock()


def push_frame(camera_id: str, image) -> bool:
    """Hand one frame (BGR) from the browser to the running pipeline. False if that camera is not running."""
    with _push_lock:
        src = PUSH_SOURCES.get(camera_id)
    if src is None or src._stop.is_set():
        return False
    src._put(Frame(camera_id, src.pushed, int(time.time() * 1000), time.perf_counter_ns(), image))
    src.pushed += 1
    return True


def source_kind(source: str) -> str:
    if source == "browser":
        return "push"
    if source.isdigit():
        return "webcam"
    if "://" in source:
        return "stream"
    return "file"


class FrameSource:
    def __init__(self, cam: CameraCfg, sample_fps: float, reconnect_s: float = 3.0):
        self.cam = cam
        self.sample_fps = sample_fps
        self.kind = source_kind(cam.source)
        self.reconnect_s = reconnect_s
        self.live = self.kind in ("webcam", "stream", "push") or cam.realtime
        # Live: newest frame wins. Offline file: lossless, back-pressured.
        self._q: queue.Queue = queue.Queue(maxsize=1 if self.live else 16)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"ingest-{cam.camera_id}", daemon=True)
        self.dropped = 0
        self.pushed = 0

    def start(self) -> "FrameSource":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def frames(self) -> Iterator[Frame]:
        while True:
            item = self._q.get()
            if item is _END:
                return
            yield item

    def _open(self) -> cv2.VideoCapture | None:
        src: str | int = int(self.cam.source) if self.kind == "webcam" else self.cam.source
        if self.kind == "file" and not Path(self.cam.source).exists():
            log.error("camera %s: file not found: %s", self.cam.camera_id, self.cam.source)
            return None
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            log.warning("camera %s: cannot open %s", self.cam.camera_id, self.cam.source)
            return None
        return cap

    def _put(self, frame: Frame) -> None:
        if self.live:
            try:
                self._q.get_nowait()  # discard the stale frame the consumer has not taken
                self.dropped += 1
            except queue.Empty:
                pass
            self._q.put_nowait(frame)
        else:
            while not self._stop.is_set():
                try:
                    self._q.put(frame, timeout=0.5)
                    return
                except queue.Full:
                    continue

    def _run(self) -> None:
        try:
            if self.kind == "file":
                self._run_file()
            elif self.kind == "push":
                self._run_push()
            else:
                self._run_live()
        except Exception:
            log.exception("camera %s: ingest crashed", self.cam.camera_id)
        finally:
            # The end marker must get through even when the live slot is occupied.
            while True:
                try:
                    self._q.put(_END, timeout=0.5)
                    break
                except queue.Full:
                    try:
                        self._q.get_nowait()
                    except queue.Empty:
                        pass

    def _run_push(self) -> None:
        with _push_lock:
            PUSH_SOURCES[self.cam.camera_id] = self
        try:
            self._stop.wait()
        finally:
            with _push_lock:
                if PUSH_SOURCES.get(self.cam.camera_id) is self:
                    del PUSH_SOURCES[self.cam.camera_id]

    def _run_file(self) -> None:
        cap = self._open()
        if cap is None:
            return
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if self.cam.start_time:
            start_ms = int(datetime.fromisoformat(self.cam.start_time).astimezone().timestamp() * 1000)
        else:
            start_ms = int(time.time() * 1000)
        step_ms = 1000.0 / self.sample_fps
        next_sample_ms = 0.0
        wall_start = time.perf_counter()
        index = 0
        emitted = 0
        offset_ms = 0.0  # looping: time of the current pass's first frame
        while not self._stop.is_set():
            ok = cap.grab()
            if not ok:
                if not (self.cam.loop and index > 0):
                    break
                # Demo feed: replay the clip; stream time keeps increasing like a live camera.
                offset_ms += index * 1000.0 / native_fps
                index = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            pos_ms = offset_ms + index * 1000.0 / native_fps
            index += 1
            if pos_ms + 1e-6 < next_sample_ms:
                continue
            next_sample_ms += step_ms
            ok, image = cap.retrieve()
            if not ok:
                continue
            if self.cam.realtime:
                delay = pos_ms / 1000.0 - (time.perf_counter() - wall_start)
                if delay > 0:
                    time.sleep(delay)
            self._put(Frame(self.cam.camera_id, emitted, start_ms + int(pos_ms), time.perf_counter_ns(), image))
            emitted += 1
        cap.release()
        log.info("camera %s: file finished after %d sampled frames", self.cam.camera_id, emitted)

    def _run_live(self) -> None:
        period = 1.0 / self.sample_fps
        emitted = 0
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                time.sleep(self.reconnect_s)
                continue
            next_t = time.perf_counter()
            while not self._stop.is_set():
                ok = cap.grab()  # always drain so the decoder does not fall behind
                if not ok:
                    log.warning("camera %s: stream lost, reconnecting", self.cam.camera_id)
                    break
                now = time.perf_counter()
                if now < next_t:
                    continue
                next_t = max(next_t + period, now)
                ok, image = cap.retrieve()
                if ok:
                    self._put(Frame(self.cam.camera_id, emitted, int(time.time() * 1000), time.perf_counter_ns(), image))
                    emitted += 1
            cap.release()
            if not self._stop.is_set():
                time.sleep(self.reconnect_s)
