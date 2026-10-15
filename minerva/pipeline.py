"""Wires the stages together: one thread per camera and a single VLM worker thread, connected by a
bounded job queue.

- triggered mode: ingest -> YOLO/ByteTrack -> rules -> candidate -> VLM check -> fusion -> store
- periodic mode (no YOLO): ingest -> window of frames -> VLM -> cross-window agreement -> store"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from minerva import RULES_VERSION
from minerva.config import CameraCfg, Settings, resolve_device
from minerva.ingest.frame_source import FrameSource
from minerva.ingest.ring_buffer import RingBuffer
from minerva.store.db import EventRecord, EventStore, Timings
from minerva.types import Candidate
from minerva.understand.fusion import fuse
from minerva.understand.keyframes import encode_jpeg, make_grid
from minerva.understand.periodic import WindowAggregator

log = logging.getLogger(__name__)


@dataclass
class VlmJob:
    camera_id: str
    grid_jpeg: bytes
    capture_ns: int  # wall clock of the last frame in the job
    cand: Candidate | None = None  # triggered mode
    window: tuple[int, int] | None = None  # periodic mode: (start_ms, end_ms)
    detect_ms: float | None = None
    trigger_ms: float | None = None
    enqueued_ns: int = field(default_factory=time.perf_counter_ns)
    end_ms: int = 0  # extended while a triggered episode is ongoing
    event_uid: str | None = None
    last_flush_ms: int = 0

    @property
    def score(self) -> float:
        return self.cand.score if self.cand else 0.0


class JobQueue:
    """Bounded per camera. When a camera's backlog is full, the lowest-score job is dropped."""

    def __init__(self, per_camera: int, on_drop):
        self.per_camera = per_camera
        self.on_drop = on_drop
        self._jobs: list[VlmJob] = []
        self._cv = threading.Condition()

    def put(self, job: VlmJob) -> None:
        with self._cv:
            self._jobs.append(job)
            mine = [j for j in self._jobs if j.camera_id == job.camera_id]
            if len(mine) > self.per_camera:
                victim = min(mine, key=lambda j: (j.score, -j.enqueued_ns))
                self._jobs.remove(victim)
                self.on_drop(victim, "vlm_queue_full")
            self._cv.notify()

    def get(self, timeout: float) -> VlmJob | None:
        with self._cv:
            if not self._jobs and not self._cv.wait(timeout):
                return None
            return self._jobs.pop(0) if self._jobs else None

    def pending(self, camera_id: str) -> int:
        with self._cv:
            return sum(j.camera_id == camera_id for j in self._jobs)

    def __len__(self) -> int:
        with self._cv:
            return len(self._jobs)


class Pipeline:
    def __init__(self, settings: Settings, cameras: list[CameraCfg], store: EventStore, use_vlm: bool = True):
        self.s = settings
        self.cameras = cameras
        self.store = store
        self.mode = settings.analysis.mode
        if self.mode == "periodic" and not use_vlm:
            raise ValueError("periodic mode has no YOLO triggers, so it needs the VLM")
        self.device = resolve_device(settings.device)
        log.info("mode: %s, device: %s", self.mode, self.device)

        self.detector = self.pose = None
        if self.mode == "triggered":
            from minerva.vision.detector import Detector
            from minerva.vision.pose import PoseEstimator

            self.detector = Detector(settings.detector, self.device)
            self.pose = PoseEstimator(settings.detector.pose_weights, settings.detector.backend, self.device)
        self._det_lock = threading.Lock()  # YOLO models are not thread-safe
        self._pose_lock = threading.Lock()

        self.vlm = None
        if use_vlm:
            from minerva.understand.vlm_client import make_client

            self.vlm = make_client(settings.vlm)
            if not self.vlm.ping():
                raise RuntimeError(
                    f"VLM '{settings.vlm.model}' not reachable at {settings.vlm.host} ({settings.vlm.backend}). "
                    f"For Ollama run `ollama pull {settings.vlm.model}`; or pass --no-vlm in triggered mode."
                )
        self.aggregator = WindowAggregator(settings.analysis.confirm_votes, int(settings.analysis.max_gap_s * 1000))
        self.queue = JobQueue(settings.vlm.max_queue_per_camera, self._on_drop)
        self._open: dict[tuple[str, tuple], VlmJob] = {}
        self._open_lock = threading.Lock()
        self._stop = threading.Event()
        self.keyframe_dir = settings.data_dir / "keyframes"
        self.versions = {
            "mode": self.mode,
            "detector": settings.detector.weights if self.mode == "triggered" else None,
            "vlm": f"{settings.vlm.backend}:{settings.vlm.model}" if use_vlm else None,
            "rules": RULES_VERSION if self.mode == "triggered" else None,
        }
        for cam in cameras:
            store.upsert_camera(cam)

    # ---------- camera side ----------

    def _camera_loop(self, cam: CameraCfg) -> None:
        source = FrameSource(cam, self.s.sample_fps).start()
        ring = RingBuffer(self.s.ring_buffer_s, self.s.ring_buffer_max_width)
        step = self._triggered_step(cam, ring) if self.mode == "triggered" else self._periodic_step(cam, ring)
        n = 0
        for frame in source.frames():
            if self._stop.is_set():
                source.stop()
                break
            step(frame)
            n += 1
            if n % 200 == 0:
                log.info("%s: %d frames, ingest dropped %d, VLM backlog %d",
                         cam.camera_id, n, source.dropped, self.queue.pending(cam.camera_id))
        if not self._stop.is_set() and hasattr(step, "flush"):
            step.flush()
        log.info("%s: stream ended after %d frames", cam.camera_id, n)

    def _triggered_step(self, cam: CameraCfg, ring: RingBuffer):
        from minerva.triggers.base import Scene
        from minerva.triggers.engine import TriggerEngine
        from minerva.vision.detector import CameraTracker
        from minerva.vision.motion import MotionMeter

        tracker = CameraTracker(self.s.tracker, self.s.detector, self.s.sample_fps)
        motion = MotionMeter()
        engine = TriggerEngine(self.s.triggers, cam)

        def step(frame) -> None:
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
            out = engine.step(Scene(cam.camera_id, frame.ts_ms, w, h, persons, objects, motion.ratio, pose_fn))
            trigger_ms = (time.perf_counter() - t1) * 1000

            for cand in out.candidates:
                grid = make_grid(ring.pick(cand.onset_ms, cand.trigger_ms), cand, cam.zones, self.s.vlm.tile_px)
                job = VlmJob(cam.camera_id, encode_jpeg(grid), frame.wall_ns, cand=cand, detect_ms=detect_ms,
                             trigger_ms=trigger_ms, end_ms=cand.trigger_ms)
                with self._open_lock:
                    self._open[(cam.camera_id, cand.key)] = job
                log.info("candidate %s/%s on %s (score %.2f, tracks %s)", cand.category.value,
                         cand.subtype or "-", cam.camera_id, cand.score, cand.track_ids)
                self.queue.put(job)
            for key in out.ongoing:
                self._extend(cam.camera_id, key, frame.ts_ms)

        return step

    def _periodic_step(self, cam: CameraCfg, ring: RingBuffer):
        window_ms = int(self.s.analysis.window_s * 1000)

        state = {"start": None, "last": None}

        def emit(end_ms: int, wall_ns: int) -> None:
            start = state["start"]
            if end_ms - start > 2 * window_ms:
                log.debug("%s: window stretched to %.1f s (VLM slower than real time)",
                          cam.camera_id, (end_ms - start) / 1000)
            grid = make_grid(ring.pick(start, end_ms), None, cam.zones, self.s.vlm.tile_px)
            self.queue.put(VlmJob(cam.camera_id, encode_jpeg(grid), wall_ns, window=(start, end_ms)))
            state["start"] = end_ms

        def step(frame) -> None:
            ring.push(frame.ts_ms, frame.image, {})
            state["last"] = (frame.ts_ms, frame.wall_ns)
            if state["start"] is None:
                state["start"] = frame.ts_ms
            # Queue a window only once the previous one has been taken: if the VLM is slower than
            # real time, windows grow longer instead of piling up.
            if frame.ts_ms - state["start"] >= window_ms and not self.queue.pending(cam.camera_id):
                emit(frame.ts_ms, frame.wall_ns)

        def flush() -> None:
            """At stream end, analyse the tail if it is at least a second long."""
            if state["last"] and state["start"] is not None and state["last"][0] - state["start"] >= 1000:
                emit(*state["last"])

        step.flush = flush
        return step

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
        category = job.cand.category.value if job.cand else "window"
        ts = job.cand.trigger_ms if job.cand else job.window[1]
        log.warning("dropped %s job on %s: %s", category, job.camera_id, reason)
        self.store.record_dropped(job.camera_id, ts, category, job.score, reason)

    def _save_keyframe(self, camera_id: str, uid: str, ts_ms: int, jpeg: bytes) -> str:
        day = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone().strftime("%Y-%m-%d")
        path = self.keyframe_dir / camera_id / day / f"{uid}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(jpeg)
        return path.as_posix()

    def _commit(self, rec: EventRecord, timings: Timings, capture_ns: int, onset_to_trigger_ms: int) -> int:
        event_id = self.store.insert_event(rec, timings)
        proc_ms = (time.perf_counter_ns() - capture_ns) / 1e6
        self.store.finish_timings(event_id, proc_ms, proc_ms + onset_to_trigger_ms)
        return event_id

    def _process_candidate(self, job: VlmJob, queue_ms: float) -> None:
        c = job.cand
        ans, vlm_ms = (None, None)
        if self.vlm is not None:
            ans, vlm_ms = self.vlm.classify(job.grid_jpeg, highlighted=True)
        fused = fuse(c, ans, self.s.fusion)
        uid = uuid.uuid4().hex
        rec = EventRecord(
            event_uid=uid, camera_id=c.camera_id, start_ts=c.onset_ms, end_ts=max(job.end_ms, c.trigger_ms),
            category=c.category.value, subtype=c.subtype, status=fused.status.value,
            confidence=fused.confidence, trigger_score=c.score, vlm_category=fused.vlm_category,
            summary=fused.summary, zone=c.zone, track_ids=list(c.track_ids),
            keyframe_path=self._save_keyframe(c.camera_id, uid, c.onset_ms, job.grid_jpeg),
            model_versions=self.versions,
        )
        timings = Timings(detect_ms=job.detect_ms, trigger_ms=job.trigger_ms, queue_ms=queue_ms, vlm_ms=vlm_ms)
        event_id = self._commit(rec, timings, job.capture_ns, c.trigger_ms - c.onset_ms)
        job.event_uid = uid
        log.info("event %s: %s/%s %s conf=%.2f vlm=%s (%.1f s) - %s", event_id, c.category.value,
                 c.subtype or "-", fused.status.value, fused.confidence, fused.vlm_category,
                 (vlm_ms or 0) / 1000, fused.summary)

    def _process_window(self, job: VlmJob, queue_ms: float) -> None:
        start, end = job.window
        ans, vlm_ms = self.vlm.classify(job.grid_jpeg, highlighted=False)
        d = self.aggregator.add(job.camera_id, start, end, ans)
        log.debug("%s window %.1f s: %s (%.1f s)", job.camera_id, (end - start) / 1000,
                  ans.category if ans else "error", vlm_ms / 1000)
        if d.action == "new":
            ev = d.event
            rec = EventRecord(
                event_uid=ev.event_uid, camera_id=job.camera_id, start_ts=start, end_ts=end,
                category=ev.category, subtype=None, status=d.status.value, confidence=d.confidence,
                trigger_score=0.0, vlm_category=ev.category, summary=d.summary, zone=None, track_ids=[],
                keyframe_path=self._save_keyframe(job.camera_id, ev.event_uid, start, job.grid_jpeg),
                model_versions=self.versions,
            )
            event_id = self._commit(rec, Timings(queue_ms=queue_ms, vlm_ms=vlm_ms), job.capture_ns, 0)
            log.info("event %s: %s %s conf=%.2f (%.1f s) - %s", event_id, ev.category, d.status.value,
                     d.confidence, vlm_ms / 1000, d.summary)
        elif d.action == "update":
            self.store.update_event(d.event.event_uid, d.event.end_ms, d.status.value, d.confidence)
            log.info("event %s… now %s conf=%.2f (%d windows)", d.event.event_uid[:8], d.status.value,
                     d.confidence, d.event.votes)

    def _vlm_loop(self, cameras_done: threading.Event) -> None:
        while True:
            job = self.queue.get(timeout=0.5)
            if job is None:
                if cameras_done.is_set() and len(self.queue) == 0:
                    return
                continue
            queue_ms = (time.perf_counter_ns() - job.enqueued_ns) / 1e6
            try:
                if job.cand is not None:
                    self._process_candidate(job, queue_ms)
                else:
                    self._process_window(job, queue_ms)
            except Exception:
                log.exception("failed to process job from %s", job.camera_id)

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
