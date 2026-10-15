"""Wires the stages together: one thread per camera (ingest -> detect/track -> triggers) and a
single VLM worker thread (verify -> fuse -> store), connected by a bounded job queue."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from minerva import RULES_VERSION
from minerva.config import CameraCfg, Settings, resolve_device
from minerva.ingest.frame_source import FrameSource
from minerva.ingest.ring_buffer import RingBuffer
from minerva.store.db import EventRecord, EventStore, Timings
from minerva.triggers.base import Scene
from minerva.triggers.engine import TriggerEngine
from minerva.types import Candidate
from minerva.understand.fusion import fuse
from minerva.understand.keyframes import encode_jpeg, make_grid

log = logging.getLogger(__name__)


@dataclass
class VlmJob:
    cand: Candidate
    grid_jpeg: bytes
    detect_ms: float
    trigger_ms: float
    capture_ns: int  # wall clock of the trigger frame
    enqueued_ns: int = field(default_factory=time.perf_counter_ns)
    end_ms: int = 0  # extended while the episode is ongoing
    event_uid: str | None = None
    last_flush_ms: int = 0


class JobQueue:
    """Bounded per camera. When a camera's backlog is full, the lowest-score candidate is dropped."""

    def __init__(self, per_camera: int, on_drop):
        self.per_camera = per_camera
        self.on_drop = on_drop
        self._jobs: list[VlmJob] = []
        self._cv = threading.Condition()

    def put(self, job: VlmJob) -> None:
        with self._cv:
            self._jobs.append(job)
            mine = [j for j in self._jobs if j.cand.camera_id == job.cand.camera_id]
            if len(mine) > self.per_camera:
                victim = min(mine, key=lambda j: (j.cand.score, -j.enqueued_ns))
                self._jobs.remove(victim)
                self.on_drop(victim, "vlm_queue_full")
            self._cv.notify()

    def get(self, timeout: float) -> VlmJob | None:
        with self._cv:
            if not self._jobs and not self._cv.wait(timeout):
                return None
            return self._jobs.pop(0) if self._jobs else None

    def __len__(self) -> int:
        with self._cv:
            return len(self._jobs)


class Pipeline:
    def __init__(self, settings: Settings, cameras: list[CameraCfg], store: EventStore, use_vlm: bool = True):
        from minerva.vision.detector import Detector
        from minerva.vision.pose import PoseEstimator

        self.s = settings
        self.cameras = cameras
        self.store = store
        self.device = resolve_device(settings.device)
        log.info("device: %s", self.device)
        self.detector = Detector(settings.detector, self.device)
        self.pose = PoseEstimator(settings.detector.pose_weights, settings.detector.backend, self.device)
        self._det_lock = threading.Lock()  # YOLO models are not thread-safe
        self._pose_lock = threading.Lock()
        self.vlm = None
        if use_vlm:
            from minerva.understand.vlm_client import VlmClient

            self.vlm = VlmClient(settings.vlm)
            if not self.vlm.ping():
                raise RuntimeError(
                    f"Ollama model '{settings.vlm.model}' not reachable at {settings.vlm.host}. "
                    f"Run `ollama pull {settings.vlm.model}` or pass --no-vlm."
                )
        self.queue = JobQueue(settings.vlm.max_queue_per_camera, self._on_drop)
        self._open: dict[tuple[str, tuple], VlmJob] = {}
        self._open_lock = threading.Lock()
        self._stop = threading.Event()
        self.keyframe_dir = settings.data_dir / "keyframes"
        self.versions = {
            "detector": settings.detector.weights,
            "vlm": settings.vlm.model if use_vlm else None,
            "rules": RULES_VERSION,
        }
        for cam in cameras:
            store.upsert_camera(cam)

    # ---------- camera side ----------

    def _camera_loop(self, cam: CameraCfg) -> None:
        from minerva.vision.detector import CameraTracker
        from minerva.vision.motion import MotionMeter

        source = FrameSource(cam, self.s.sample_fps).start()
        tracker = CameraTracker(self.s.tracker, self.s.detector, self.s.sample_fps)
        motion = MotionMeter()
        ring = RingBuffer(self.s.ring_buffer_s, self.s.ring_buffer_max_width)
        engine = TriggerEngine(self.s.triggers, cam)
        n = 0
        for frame in source.frames():
            if self._stop.is_set():
                source.stop()
                break
            t0 = time.perf_counter()
            with self._det_lock:
                dets = self.detector(frame.image)
            persons, objects = tracker.update(dets)
            detect_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            motion.update(frame.image, [p.box for p in persons])
            ring.push(frame.ts_ms, frame.image, {t.track_id: t.box for t in persons + objects})
            image = frame.image

            def pose_fn(track, image=image):
                with self._pose_lock:
                    return self.pose(image, track)

            h, w = image.shape[:2]
            scene = Scene(cam.camera_id, frame.ts_ms, w, h, persons, objects, motion.ratio, pose_fn)
            out = engine.step(scene)
            trigger_ms = (time.perf_counter() - t1) * 1000

            for cand in out.candidates:
                grid = make_grid(ring.pick(cand.onset_ms, cand.trigger_ms), cand, cam.zones, self.s.vlm.tile_px)
                job = VlmJob(cand, encode_jpeg(grid), detect_ms, trigger_ms, frame.wall_ns, end_ms=cand.trigger_ms)
                with self._open_lock:
                    self._open[(cam.camera_id, cand.key)] = job
                log.info("candidate %s/%s on %s (score %.2f, tracks %s)", cand.category.value,
                         cand.subtype or "-", cam.camera_id, cand.score, cand.track_ids)
                self.queue.put(job)
            for key in out.ongoing:
                self._extend(cam.camera_id, key, frame.ts_ms)
            n += 1
            if n % 200 == 0:
                log.info("%s: %d frames, %d persons, detect %.0f ms, ingest dropped %d",
                         cam.camera_id, n, len(persons), detect_ms, source.dropped)
        log.info("%s: stream ended after %d frames", cam.camera_id, n)

    def _extend(self, camera_id: str, key: tuple, ts_ms: int) -> None:
        with self._open_lock:
            job = self._open.get((camera_id, key))
        if job is None:
            return
        job.end_ms = max(job.end_ms, ts_ms)
        if job.event_uid and ts_ms - job.last_flush_ms >= 2000:
            job.last_flush_ms = ts_ms
            self.store.extend_end(job.event_uid, job.end_ms)

    # ---------- VLM side ----------

    def _on_drop(self, job: VlmJob, reason: str) -> None:
        c = job.cand
        log.warning("dropped %s candidate on %s: %s", c.category.value, c.camera_id, reason)
        self.store.record_dropped(c.camera_id, c.trigger_ms, c.category.value, c.score, reason)

    def _process(self, job: VlmJob) -> None:
        c = job.cand
        queue_ms = (time.perf_counter_ns() - job.enqueued_ns) / 1e6
        ans, vlm_ms = (None, None)
        if self.vlm is not None:
            ans, vlm_ms = self.vlm.classify(job.grid_jpeg)
        fused = fuse(c, ans, self.s.fusion)

        uid = uuid.uuid4().hex
        day = datetime.fromtimestamp(c.onset_ms / 1000, tz=timezone.utc).astimezone().strftime("%Y-%m-%d")
        path = self.keyframe_dir / c.camera_id / day / f"{uid}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(job.grid_jpeg)

        rec = EventRecord(
            event_uid=uid, camera_id=c.camera_id, start_ts=c.onset_ms, end_ts=max(job.end_ms, c.trigger_ms),
            category=c.category.value, subtype=c.subtype, status=fused.status.value,
            confidence=fused.confidence, trigger_score=c.score, vlm_category=fused.vlm_category,
            summary=fused.summary, zone=c.zone, track_ids=list(c.track_ids),
            keyframe_path=path.as_posix(), model_versions=self.versions,
        )
        timings = Timings(detect_ms=job.detect_ms, trigger_ms=job.trigger_ms, queue_ms=queue_ms, vlm_ms=vlm_ms)
        event_id = self.store.insert_event(rec, timings)
        proc_ms = (time.perf_counter_ns() - job.capture_ns) / 1e6
        self.store.finish_timings(event_id, proc_ms, proc_ms + (c.trigger_ms - c.onset_ms))
        job.event_uid = uid
        log.info("event %s: %s/%s %s conf=%.2f vlm=%s (%.1f s) - %s", event_id, c.category.value,
                 c.subtype or "-", fused.status.value, fused.confidence, fused.vlm_category,
                 (vlm_ms or 0) / 1000, fused.summary)

    def _vlm_loop(self, cameras_done: threading.Event) -> None:
        while True:
            job = self.queue.get(timeout=0.5)
            if job is None:
                if cameras_done.is_set() and len(self.queue) == 0:
                    return
                continue
            try:
                self._process(job)
            except Exception:
                log.exception("failed to process %s candidate", job.cand.category.value)

    # ---------- lifecycle ----------

    def run(self, max_seconds: float | None = None) -> None:
        cameras_done = threading.Event()
        cam_threads = [threading.Thread(target=self._camera_loop, args=(c,), name=f"cam-{c.camera_id}", daemon=True)
                       for c in self.cameras]
        vlm_thread = threading.Thread(target=self._vlm_loop, args=(cameras_done,), name="vlm", daemon=True)
        for t in cam_threads:
            t.start()
        vlm_thread.start()
        deadline = time.monotonic() + max_seconds if max_seconds else None
        try:
            while any(t.is_alive() for t in cam_threads):
                if deadline and time.monotonic() > deadline:
                    log.info("time limit reached, stopping cameras")
                    self._stop.set()
                time.sleep(0.5)
                with self._open_lock:
                    now_ns = time.perf_counter_ns()
                    stale = [k for k, j in self._open.items() if j.event_uid and now_ns - j.enqueued_ns > 600e9]
                    for k in stale:
                        del self._open[k]
        except KeyboardInterrupt:
            log.info("interrupted, finishing queued events (Ctrl+C again to abort)")
            self._stop.set()
        cameras_done.set()
        vlm_thread.join()
        log.info("pipeline stopped")
