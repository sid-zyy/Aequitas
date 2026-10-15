"""YOLO detection (shared across cameras) and ByteTrack tracking (one tracker per camera and class group).

People and bags are tracked separately so a person box can never inherit a bag's ID."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import supervision as sv

from minerva.config import DetectorCfg, TrackerCfg
from minerva.types import TrackObs

log = logging.getLogger(__name__)

PERSON = 0
COCO_NAMES = {0: "person", 24: "backpack", 26: "handbag", 28: "suitcase"}


def load_yolo(weights: str, backend: str, device: str):
    from ultralytics import YOLO

    if backend == "openvino":
        export_dir = Path(weights).with_suffix("").name + "_openvino_model"
        if not Path(export_dir).exists():
            log.info("exporting %s to OpenVINO (one-off)", weights)
            YOLO(weights).export(format="openvino", half=False)
        return YOLO(export_dir, task="pose" if "pose" in weights else "detect")
    return YOLO(weights)


class Detector:
    def __init__(self, cfg: DetectorCfg, device: str):
        self.cfg = cfg
        self.device = device
        self.model = load_yolo(cfg.weights, cfg.backend, device)
        self.half = device.startswith("cuda")

    def __call__(self, image: np.ndarray) -> sv.Detections:
        kwargs = {"half": True} if self.half else {}  # 'half' is deprecated off-GPU and warns per call
        result = self.model.predict(
            image,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            classes=list(COCO_NAMES),
            device=self.device if self.cfg.backend == "torch" else None,
            verbose=False,
            **kwargs,
        )[0]
        return sv.Detections.from_ultralytics(result)


class CameraTracker:
    def __init__(self, cfg: TrackerCfg, det_cfg: DetectorCfg, fps: float):
        self.det_cfg = det_cfg

        def make() -> sv.ByteTrack:
            return sv.ByteTrack(
                track_activation_threshold=cfg.activation_threshold,
                lost_track_buffer=cfg.lost_track_buffer,
                minimum_matching_threshold=cfg.matching_threshold,
                frame_rate=max(1, round(fps)),
            )

        self.person_tracker = make()
        self.object_tracker = make()

    def update(self, dets: sv.Detections) -> tuple[list[TrackObs], list[TrackObs]]:
        if len(dets):
            is_person = dets.class_id == PERSON
            persons = self.person_tracker.update_with_detections(dets[is_person])
            objects = self.object_tracker.update_with_detections(dets[~is_person])
        else:
            persons = self.person_tracker.update_with_detections(dets)
            objects = self.object_tracker.update_with_detections(dets)
        return self._to_obs(persons, person=True), self._to_obs(objects, person=False)

    def _to_obs(self, dets: sv.Detections, person: bool) -> list[TrackObs]:
        out = []
        if dets.tracker_id is None:
            return out
        # Object IDs are offset so person and object IDs never collide in shared dicts.
        offset = 0 if person else 1_000_000
        for box, conf, cls, tid in zip(dets.xyxy, dets.confidence, dets.class_id, dets.tracker_id):
            obs = TrackObs(int(tid) + offset, COCO_NAMES.get(int(cls), str(cls)), float(conf), tuple(map(float, box)))
            if person and (obs.conf < self.det_cfg.min_person_conf or obs.h < self.det_cfg.min_person_height_px):
                continue
            out.append(obs)
        return out
