"""Torso angle from YOLO pose, run only on the crop of a person-down candidate."""

from __future__ import annotations

import math

import numpy as np

from aequitas.types import PoseResult, TrackObs

# COCO keypoint indices
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12


def torso_angle(kpts_xy: np.ndarray, kpts_conf: np.ndarray) -> PoseResult | None:
    """Angle between the shoulder-midpoint -> hip-midpoint vector and vertical, in degrees."""
    idx = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
    conf = float(np.min(kpts_conf[idx]))
    shoulder = (kpts_xy[L_SHOULDER] + kpts_xy[R_SHOULDER]) / 2
    hip = (kpts_xy[L_HIP] + kpts_xy[R_HIP]) / 2
    dx, dy = hip - shoulder
    if abs(dx) + abs(dy) < 1e-3:
        return None
    return PoseResult(math.degrees(math.atan2(abs(dx), abs(dy))), conf)


class PoseEstimator:
    def __init__(self, weights: str, backend: str, device: str, pad_frac: float = 0.15):
        from aequitas.vision.detector import load_yolo

        self.model = load_yolo(weights, backend, device)
        self.device = device if backend == "torch" else None
        self.pad_frac = pad_frac

    def __call__(self, image: np.ndarray, track: TrackObs) -> PoseResult | None:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = track.box
        pad = self.pad_frac * max(x2 - x1, y2 - y1)
        x1, y1 = max(0, int(x1 - pad)), max(0, int(y1 - pad))
        x2, y2 = min(w, int(x2 + pad)), min(h, int(y2 + pad))
        if x2 - x1 < 16 or y2 - y1 < 16:
            return None
        res = self.model.predict(image[y1:y2, x1:x2], imgsz=320, device=self.device, verbose=False)[0]
        if res.keypoints is None or len(res.keypoints) == 0 or res.keypoints.conf is None:
            return None
        best = int(res.boxes.conf.argmax()) if res.boxes is not None and len(res.boxes) else 0
        xy = res.keypoints.xy[best].cpu().numpy()
        conf = res.keypoints.conf[best].cpu().numpy()
        return torso_angle(xy, conf)
