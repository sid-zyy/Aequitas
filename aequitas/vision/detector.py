"""YOLO detection (shared across cameras) and ByteTrack/OC-SORT tracking (Ultralytics; one tracker per camera and
class group).

People and bags are tracked separately so a person box can never inherit a bag's ID."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import supervision as sv

from aequitas.config import DetectorCfg, TrackerCfg
from aequitas.types import TrackObs

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


def Detector(cfg: DetectorCfg, device: str):
    """People and bags in a BGR frame -> sv.Detections with COCO-80 class ids (COCO_NAMES)."""
    return RFDetrDetector(cfg, device) if cfg.family == "rfdetr" else YoloDetector(cfg, device)


class YoloDetector:
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


# RF-DETR's predict() returns COCO category ids (1-90, with gaps); Aequitas uses the 80-class ids YOLO uses.
COCO91_TO_80 = {1: 0, 27: 24, 31: 26, 33: 28}  # person, backpack, handbag, suitcase (only the classes we keep)


class RFDetrDetector:
    def __init__(self, cfg: DetectorCfg, device: str, model=None):
        self.cfg = cfg
        if model is None:
            import rfdetr

            cls = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "large": "RFDETRLarge"}
            model = getattr(rfdetr, cls[cfg.rfdetr_size])(device="cuda" if device.startswith("cuda") else device)
        self.model = model

    def __call__(self, image: np.ndarray) -> sv.Detections:
        d = self.model.predict(image[:, :, ::-1].copy(), threshold=self.cfg.conf)  # BGR -> RGB
        cls = np.array([COCO91_TO_80.get(int(c), -1) for c in d.class_id], dtype=int)
        keep = cls >= 0
        return sv.Detections(xyxy=d.xyxy[keep].astype(np.float32), confidence=d.confidence[keep].astype(np.float32),
                             class_id=cls[keep])


class Boxes:
    """The attributes Ultralytics trackers read from a Results.boxes object, built from plain numpy arrays."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy, self.conf, self.cls = xyxy, conf, cls
        self.xywh = np.concatenate([(xyxy[:, :2] + xyxy[:, 2:]) / 2, xyxy[:, 2:] - xyxy[:, :2]], axis=1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> Boxes:
        return Boxes(self.xyxy[idx], self.conf[idx], self.cls[idx])


def make_tracker(cfg: TrackerCfg):
    """Ultralytics BYTETracker or OCSORT with Aequitas's thresholds; track_buffer is in frames."""
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.trackers.oc_sort import OCSORT
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml

    cls = {"bytetrack": BYTETracker, "ocsort": OCSORT}[cfg.algorithm]
    args = YAML.load(check_yaml(f"{cfg.algorithm}.yaml"))
    args.update(
        track_high_thresh=cfg.activation_threshold,  # first-pass matching and track start use the same value
        new_track_thresh=cfg.activation_threshold,
        track_low_thresh=min(args["track_low_thresh"], cfg.activation_threshold),
        track_buffer=cfg.lost_track_buffer,
        match_thresh=cfg.matching_threshold,
        fuse_score=cfg.fuse_score,
    )
    return cls(SimpleNamespace(**args))


class CameraTracker:
    def __init__(self, cfg: TrackerCfg, det_cfg: DetectorCfg, fps: float):
        # fps is unused: lost_track_buffer is already in frames (Ultralytics' track_buffer is too)
        self.det_cfg = det_cfg
        self.person_tracker = make_tracker(cfg)
        self.object_tracker = make_tracker(cfg)

    def update(self, dets: sv.Detections) -> tuple[list[TrackObs], list[TrackObs]]:
        is_person = dets.class_id == PERSON if len(dets) else np.zeros(0, bool)
        persons = self._track(self.person_tracker, dets[is_person] if len(dets) else dets)
        objects = self._track(self.object_tracker, dets[~is_person] if len(dets) else dets)
        return self._to_obs(persons, person=True), self._to_obs(objects, person=False)

    @staticmethod
    def _track(tracker, dets: sv.Detections) -> list[tuple[int, np.ndarray, float, int]]:
        """-> (track id, detection box, confidence, class) for each track matched this frame."""
        n = len(dets)
        xyxy = dets.xyxy.astype(np.float32) if n else np.zeros((0, 4), np.float32)
        conf = dets.confidence.astype(np.float32) if n and dets.confidence is not None else np.ones(n, np.float32)
        cls = dets.class_id.astype(np.float32) if n and dets.class_id is not None else np.zeros(n, np.float32)
        out = tracker.update(Boxes(xyxy, conf, cls))
        # Each row is [x1, y1, x2, y2, track_id, score, cls, idx]; report the detection's own box (idx), not the
        # Kalman estimate, as supervision did.
        return [(int(r[4]), xyxy[int(r[7])], float(conf[int(r[7])]), int(cls[int(r[7])])) for r in out]

    def _to_obs(self, tracks: list[tuple[int, np.ndarray, float, int]], person: bool) -> list[TrackObs]:
        out = []
        # Object IDs are offset so person and object IDs never collide in shared dicts.
        offset = 0 if person else 1_000_000
        for tid, box, conf, cls in tracks:
            obs = TrackObs(tid + offset, COCO_NAMES.get(cls, str(cls)), conf, tuple(map(float, box)))
            if person and (obs.conf < self.det_cfg.min_person_conf or obs.h < self.det_cfg.min_person_height_px):
                continue
            out.append(obs)
        return out
