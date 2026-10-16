"""Wires the stages together: one thread per camera and `vlm.concurrency` understanding workers,
connected by a bounded job queue. Understanding is two stages: [1] the VLM summarizes the key frames,
[2] a text-only call classifies the summary.

- triggered mode: ingest -> YOLO/ByteTrack -> rules -> candidate -> summarize -> classify -> fusion -> store
- periodic mode (no YOLO): ingest -> window -> summarize -> classify -> cross-window agreement -> store"""

from __future__ import annotations

import logging
from pathlib import Path
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from aequitas import RULES_VERSION
from aequitas.config import CameraCfg, Settings, resolve_device
from aequitas.ingest.frame_source import FrameSource
from aequitas.ingest.ring_buffer import RingBuffer
from aequitas.store.db import EventRecord, EventStore, Timings
from aequitas.types import Candidate
from aequitas.understand.fusion import fuse, fuse_window
from aequitas.understand.gate import should_skip
from aequitas.understand.keyframes import encode_jpeg, make_grid, window_detail, window_motion
from aequitas.understand.periodic import WindowAggregator
from aequitas.understand.stages import InOrder, Understanding, escalate, needs_escalation, understand

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
    seq: int | None = None  # periodic mode: order of this window within its camera
    view: str = "cctv"
    violence: float | None = None  # periodic: highest clip-model score in the window

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
            job = self._jobs.pop(0) if self._jobs else None
            self._cv.notify_all()  # wakes a file source waiting for room
            return job

    def wait_room(self, camera_id: str, limit: int, stop: threading.Event) -> None:
        """Blocks while this camera already has `limit` jobs waiting. File sources use it so every
        window gets analysed (backpressure) instead of being stretched or dropped."""
        with self._cv:
            while not stop.is_set() and sum(j.camera_id == camera_id for j in self._jobs) >= limit:
                self._cv.wait(0.5)

    def pending(self, camera_id: str) -> int:
        with self._cv:
            return sum(j.camera_id == camera_id for j in self._jobs)

    def __len__(self) -> int:
        with self._cv:
            return len(self._jobs)


class Pipeline:
    def __init__(self, settings: Settings, cameras: list[CameraCfg], store: EventStore, use_vlm: bool = True,
                 on_frame=None, run_id: str | None = None):
        self.on_frame = on_frame  # optional callback(camera_id, frames_processed), for progress reporting
        self.run_id = run_id  # tags stage_timings rows (the upload ID for web uploads)
        self.phase = "processing"  # -> "verifying" once streams end -> "done" once the VLM queue drains
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
            from aequitas.vision.detector import Detector
            from aequitas.vision.pose import PoseEstimator

            self.detector = Detector(settings.detector, self.device)
            self.pose = PoseEstimator(settings.detector.pose_weights, settings.detector.backend, self.device)
        self.violence_model = self.hazard_detector = None
        m = settings.models
        if m.violence_dir and not (Path(m.violence_dir) / "violence_clip.onnx").exists():
            # not in git (127 MB): a fresh clone runs without it rather than failing to start
            log.warning("violence model not found in %s (download violence_clip.zip from the Kaggle run "
                        "or the release); running without it", m.violence_dir)
        elif m.violence_dir:  # triggered: fight trigger; periodic: second signal per window
            from aequitas.vision.violence import ViolenceModel

            self.violence_model = ViolenceModel(m.violence_dir)
        if self.mode == "triggered" and m.hazard_dir:
            from aequitas.vision.hazards import HazardDetector

            self.hazard_detector = HazardDetector(m.hazard_dir, self.device, m.hazard_conf)
        self._model_lock = threading.Lock()
        self._det_lock = threading.Lock()  # YOLO models are not thread-safe
        self._pose_lock = threading.Lock()

        self.vlm = self.classifier = None
        if use_vlm:
            from aequitas.understand.vlm_client import make_client

            self.vlm = make_client(settings.vlm)
            if not self.vlm.ping():
                raise RuntimeError(
                    f"VLM '{settings.vlm.model}' not reachable at {settings.vlm.host} ({settings.vlm.backend}). "
                    f"For Ollama run `ollama pull {settings.vlm.model}`; or pass --no-vlm in triggered mode."
                )
            self.classifier = self.vlm
            if settings.classifier.overridden:
                ep = settings.classifier.endpoint(settings.vlm)
                self.classifier = make_client(ep)
                if not self.classifier.ping():
                    raise RuntimeError(f"classifier '{ep.model}' not reachable at {ep.host} ({ep.backend})")
        self.order = InOrder()
        self.last_summary: dict[str, str] = {}  # per camera, for analysis.label_context
        self.gate = None
        if self.mode == "periodic" and settings.analysis.gate_model:
            try:
                from aequitas.understand.gate import SceneGate

                self.gate = SceneGate(settings.analysis.gate_model, self.device)
            except Exception as e:  # missing model or transformers: run without the gate
                log.warning("scene gate %s not loaded (%s); every window goes to the VLM",
                            settings.analysis.gate_model, e)
        self.big = None  # cascade re-checker
        if use_vlm and settings.escalation.model:
            from aequitas.understand.vlm_client import make_client

            self.big = make_client(settings.escalation.endpoint(settings.vlm))
            if not self.big.ping():
                log.warning("escalation model %s not reachable; cascade off", self.big.model)
                self.big = None
        self.aggregator = WindowAggregator(settings.analysis.confirm_votes, int(settings.analysis.max_gap_s * 1000))
        # never smaller than the windows allowed in flight, or a file source could still drop one
        self.queue = JobQueue(max(settings.vlm.max_queue_per_camera, settings.vlm.concurrency), self._on_drop)
        self._open: dict[tuple[str, tuple], VlmJob] = {}
        self._open_lock = threading.Lock()
        self._stop = threading.Event()
        self.latest_frames: dict = {}  # camera_id -> newest BGR frame
        self.keyframe_dir = settings.data_dir / "keyframes"
        self.versions = {
            "mode": self.mode,
            "detector": self.detector.name if self.detector is not None else None,  # the one loaded, after any fallback
            "vlm": f"{settings.vlm.backend}:{settings.vlm.model}" if use_vlm else None,
            "classifier": f"{self.classifier.cfg.backend}:{self.classifier.model}" if self.classifier else None,
            "rules": RULES_VERSION if self.mode == "triggered" else None,
            "violence": self.violence_model.meta.get("trained_at") if self.violence_model else None,
            "hazards": self.hazard_detector.meta.get("trained_at") if self.hazard_detector else None,
        }
        for cam in cameras:
            store.upsert_camera(cam)

    # ---------- camera side ----------

    def _camera_loop(self, cam: CameraCfg) -> None:
        source = FrameSource(cam, self.s.sample_fps).start()
        ring = RingBuffer(self.s.ring_buffer_s, self.s.ring_buffer_max_width)
        step = (self._triggered_step(cam, ring) if self.mode == "triggered"
                else self._periodic_step(cam, ring, live=source.live))
        n = 0
        for frame in source.frames():
            if self._stop.is_set():
                source.stop()
                break
            step(frame)
            self.latest_frames[cam.camera_id] = frame.image  # live preview (webcam MJPEG)
            n += 1
            if self.on_frame is not None:
                self.on_frame(cam.camera_id, n)
            if n % 200 == 0:
                log.info("%s: %d frames, ingest dropped %d, VLM backlog %d",
                         cam.camera_id, n, source.dropped, self.queue.pending(cam.camera_id))
        if not self._stop.is_set() and hasattr(step, "flush"):
            step.flush()
        log.info("%s: stream ended after %d frames", cam.camera_id, n)

    def _triggered_step(self, cam: CameraCfg, ring: RingBuffer):
        from aequitas.triggers.base import Scene
        from aequitas.triggers.engine import TriggerEngine
        from aequitas.vision.detector import CameraTracker
        from aequitas.vision.motion import MotionMeter

        tracker = CameraTracker(self.s.tracker, self.s.detector, self.s.sample_fps)
        motion = MotionMeter()
        engine = TriggerEngine(self.s.triggers, cam)
        scorer = None
        v_threshold = 0.5
        if self.violence_model is not None:
            from aequitas.vision.violence import ViolenceScorer

            scorer = ViolenceScorer(self.violence_model, int(self.s.models.violence_stride_s * 1000))
            v_threshold = self.s.models.violence_threshold or self.violence_model.threshold

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

            violence, hazards = None, []
            with self._model_lock:  # trained models are shared across camera threads
                if scorer is not None:
                    violence = scorer.update(frame.ts_ms, image)
                if self.hazard_detector is not None:
                    hazards = self.hazard_detector(image)
            h, w = image.shape[:2]
            out = engine.step(Scene(cam.camera_id, frame.ts_ms, w, h, persons, objects, motion.ratio, pose_fn,
                                    violence, v_threshold, hazards))
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

    def _periodic_step(self, cam: CameraCfg, ring: RingBuffer, live: bool = True):
        """live: a camera stream that cannot wait, so windows stretch when the VLM is slow. A file
        (live=False) gets fixed-length windows and waits for the VLM instead, so none of it is skipped."""
        window_ms = int((cam.window_s or self.s.analysis.window_s) * 1000)

        state = {"start": None, "last": None, "last_vlm": None, "vmax": None}
        band_low = self.s.escalation.violence_band_low
        a = self.s.analysis
        scorer = None
        if self.violence_model is not None:
            from aequitas.vision.violence import ViolenceScorer

            scorer = ViolenceScorer(self.violence_model, int(self.s.models.violence_stride_s * 1000))

        def emit(end_ms: int, wall_ns: int) -> None:
            start = state["start"]
            if end_ms - start > 2 * window_ms:
                log.debug("%s: window stretched to %.1f s (VLM slower than real time)",
                          cam.camera_id, (end_ms - start) / 1000)
            frames = ring.pick(start, end_ms)
            if a.skip_blank_below > 0 and window_detail(frames) < a.skip_blank_below:
                # Black, covered or blank picture: nothing to see. Both models invent things on it (seen live:
                # black frames scored 0.96 on the violence model and Gemma described "two people pushing").
                self.store.record_stage(self.run_id, cam.camera_id, "blank", None, None, None, True)
                state["vmax"] = None
                state["start"] = end_ms
                return
            if a.skip_static_below > 0 and state["last_vlm"] is not None \
                    and end_ms - state["last_vlm"] < a.max_skip_s * 1000 \
                    and window_motion(frames) < a.skip_static_below:
                # Latency: nothing moved, so the previous understanding still holds; no VLM call.
                self.store.record_stage(self.run_id, cam.camera_id, "skipped", None, None, None, True)
                state["start"] = end_ms
                return
            if self.gate is not None and state["last_vlm"] is not None:
                with self._model_lock:
                    p = self.gate.score([f.image for f in frames[::max(1, len(frames) // 4)]][:4])
                if should_skip(p, state["vmax"], end_ms - state["last_vlm"], below=a.gate_below,
                               violence_band_low=band_low, max_skip_ms=a.max_skip_s * 1000):
                    # Coarse stage says ordinary scene: no VLM call (Flashback / VadCLIP idea).
                    self.store.record_stage(self.run_id, cam.camera_id, "gated", None, None, None, True)
                    state["vmax"] = None
                    state["start"] = end_ms
                    return
            state["last_vlm"] = end_ms
            grid = make_grid(frames, None, cam.zones, self.s.vlm.tile_px)
            self.queue.put(VlmJob(cam.camera_id, encode_jpeg(grid), wall_ns, window=(start, end_ms),
                                  seq=self.order.ticket(cam.camera_id), view=cam.view, violence=state["vmax"]))
            state["vmax"] = None
            state["start"] = end_ms

        def step(frame) -> None:
            ring.push(frame.ts_ms, frame.image, {})
            if scorer is not None:
                with self._model_lock:
                    v = scorer.update(frame.ts_ms, frame.image)
                if v is not None:
                    state["vmax"] = v if state["vmax"] is None else max(state["vmax"], v)
            state["last"] = (frame.ts_ms, frame.wall_ns)
            if state["start"] is None:
                state["start"] = frame.ts_ms
            if frame.ts_ms - state["start"] < window_ms:
                return
            if not live:
                # File: every window is analysed; reading waits while `concurrency` windows are queued.
                self.queue.wait_room(cam.camera_id, max(1, self.s.vlm.concurrency), self._stop)
                emit(frame.ts_ms, frame.wall_ns)
            elif not self.queue.pending(cam.camera_id):
                # Live: queue a window only once the previous one has been taken; if the VLM is slower
                # than real time, windows grow longer instead of piling up.
                emit(frame.ts_ms, frame.wall_ns)

        def flush() -> None:
            """At stream end, analyse the tail if it is at least a second long."""
            if state["last"] and state["start"] is not None and state["last"][0] - state["start"] >= 1000:
                if not live:  # files never drop a window, including the last one
                    self.queue.wait_room(cam.camera_id, max(1, self.s.vlm.concurrency), self._stop)
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
        if job.seq is not None:
            self.order.skip(job.camera_id, job.seq, self._apply_window)

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

    def _record_stage(self, job: VlmJob, kind: str, queue_ms: float, u: Understanding) -> None:
        try:
            self.store.record_stage(self.run_id, job.camera_id, kind, queue_ms, u.summarize_ms, u.classify_ms,
                                    u.answer is not None)
        except Exception:
            log.exception("failed to record stage timings")

    def _process_candidate(self, job: VlmJob, queue_ms: float) -> None:
        c = job.cand
        u = Understanding(None, None, None, None)
        if self.vlm is not None:
            u = understand(self.vlm, self.classifier, job.grid_jpeg, highlighted=True,
                           categories=self.s.categories)
            if self.big is not None and needs_escalation(u, None, 1.1):
                u = escalate(u, self.big, job.grid_jpeg, True, self.s.categories)
            self._record_stage(job, "candidate", queue_ms, u)
        fused = fuse(c, u.answer, self.s.fusion)
        if fused.summary is None and u.summary:  # summarized but not classified: keep the description
            fused.summary = u.summary
        uid = uuid.uuid4().hex
        rec = EventRecord(
            event_uid=uid, camera_id=c.camera_id, start_ts=c.onset_ms, end_ts=max(job.end_ms, c.trigger_ms),
            category=c.category.value, subtype=c.subtype, status=fused.status.value,
            confidence=fused.confidence, trigger_score=c.score, vlm_category=fused.vlm_category,
            summary=fused.summary, zone=c.zone, track_ids=list(c.track_ids),
            keyframe_path=self._save_keyframe(c.camera_id, uid, c.onset_ms, job.grid_jpeg),
            model_versions=self.versions,
        )
        timings = Timings(detect_ms=job.detect_ms, trigger_ms=job.trigger_ms, queue_ms=queue_ms, vlm_ms=u.vlm_ms,
                          summarize_ms=u.summarize_ms, classify_ms=u.classify_ms)
        event_id = self._commit(rec, timings, job.capture_ns, c.trigger_ms - c.onset_ms)
        job.event_uid = uid
        log.info("event %s: %s/%s %s conf=%.2f label=%s (summarize %.1f s, classify %.2f s) - %s", event_id,
                 c.category.value, c.subtype or "-", fused.status.value, fused.confidence, fused.vlm_category,
                 (u.summarize_ms or 0) / 1000, (u.classify_ms or 0) / 1000, fused.summary)

    def _process_window(self, job: VlmJob, queue_ms: float) -> None:
        """Understanding runs concurrently; results are applied in window order per camera."""
        # With several windows in flight the newest finished summary may be one window older; fine as context.
        previous = self.last_summary.get(job.camera_id) if self.s.analysis.label_context else None
        u = understand(self.vlm, self.classifier, job.grid_jpeg, highlighted=False,
                       categories=self.s.categories, view=job.view, previous=previous)
        if self.big is not None and needs_escalation(u, job.violence, self.s.escalation.violence_band_low):
            u = escalate(u, self.big, job.grid_jpeg, False, self.s.categories, job.view)
        self._record_stage(job, "window", queue_ms, u)
        self.order.finish(job.camera_id, job.seq, (job, u, queue_ms), self._apply_window)

    def _apply_window(self, item) -> None:
        job, u, queue_ms = item
        try:
            self._apply_window_result(job, u, queue_ms)
        except Exception:
            log.exception("failed to store window from %s", job.camera_id)

    def _apply_window_result(self, job: VlmJob, u: Understanding, queue_ms: float) -> None:
        start, end = job.window
        if u.summary:
            self.last_summary[job.camera_id] = u.summary
        threshold = (self.s.models.violence_threshold or self.violence_model.threshold) if self.violence_model else 1.0
        ans, source = fuse_window(u.answer, job.violence, threshold, self.s.models.violence_alone_threshold)
        if u.escalated:
            source = f"{source}+rechecked:{u.escalated}"
        self.store.record_window(job.camera_id, start, end, ans.summary if ans else u.summary,
                                 ans.category if ans else None, ans.confidence if ans else None, u.summarize_ms,
                                 u.classify_ms, (time.perf_counter_ns() - job.capture_ns) / 1e6,
                                 keep=self.s.analysis.keep_windows, violence=job.violence, source=source)
        alone = self.s.models.violence_alone_threshold
        also = frozenset({"fighting"}) if (self.s.analysis.clip_votes and alone is not None
                                           and job.violence is not None and job.violence >= max(alone, threshold)
                                           and "fighting" in self.s.categories) else frozenset()
        d = self.aggregator.add(job.camera_id, start, end, ans, also)
        log.debug("%s window %.1f s: %s (summarize %.1f s, classify %.2f s)", job.camera_id, (end - start) / 1000,
                  ans.category if ans else "error", (u.summarize_ms or 0) / 1000, (u.classify_ms or 0) / 1000)
        if d.action == "new":
            ev = d.event
            rec = EventRecord(
                event_uid=ev.event_uid, camera_id=job.camera_id, start_ts=start, end_ts=end,
                category=ev.category, subtype=None, status=d.status.value, confidence=d.confidence,
                trigger_score=0.0, vlm_category=ev.category, summary=d.summary, zone=None, track_ids=[],
                keyframe_path=self._save_keyframe(job.camera_id, ev.event_uid, start, job.grid_jpeg),
                model_versions=self.versions,
            )
            timings = Timings(queue_ms=queue_ms, vlm_ms=u.vlm_ms, summarize_ms=u.summarize_ms,
                              classify_ms=u.classify_ms)
            event_id = self._commit(rec, timings, job.capture_ns, 0)
            log.info("event %s: %s %s conf=%.2f (%.1f s) - %s", event_id, ev.category, d.status.value,
                     d.confidence, (u.vlm_ms or 0) / 1000, d.summary)
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

    def stop(self) -> None:
        """Stop reading new frames; jobs already queued for the VLM still finish."""
        self._stop.set()

    def run(self, max_seconds: float | None = None) -> None:
        cameras_done = threading.Event()
        cam_threads = [threading.Thread(target=self._camera_loop, args=(c,), name=f"cam-{c.camera_id}", daemon=True)
                       for c in self.cameras]
        n_workers = self.s.vlm.concurrency if self.vlm is not None else 1
        vlm_threads = [threading.Thread(target=self._vlm_loop, args=(cameras_done,), name=f"vlm-{i}", daemon=True)
                       for i in range(n_workers)]
        for t in cam_threads + vlm_threads:
            t.start()
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
        self.phase = "verifying"
        cameras_done.set()
        for t in vlm_threads:
            t.join()
        self.phase = "done"
        log.info("pipeline stopped")
