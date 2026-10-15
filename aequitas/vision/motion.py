"""Cheap motion energy: mean absolute frame difference inside a box, relative to how much
people normally move on this camera (rolling median over the last minute)."""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from aequitas.types import Box


class MotionMeter:
    def __init__(self, work_width: int = 320, history: int = 240, floor: float = 3.0):
        self.work_width = work_width
        self.floor = floor
        self._prev: np.ndarray | None = None
        self._diff: np.ndarray | None = None
        self._scale = 1.0
        self._samples: deque[float] = deque(maxlen=history)

    def update(self, image: np.ndarray, person_boxes: list[Box]) -> None:
        h, w = image.shape[:2]
        self._scale = self.work_width / w
        gray = cv2.cvtColor(cv2.resize(image, (self.work_width, max(1, int(h * self._scale)))), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        self._diff = cv2.absdiff(gray, self._prev) if self._prev is not None and self._prev.shape == gray.shape else None
        self._prev = gray
        for box in person_boxes:
            e = self.energy(box)
            if e is not None:
                self._samples.append(e)

    def energy(self, box: Box) -> float | None:
        if self._diff is None:
            return None
        s = self._scale
        x1, y1, x2, y2 = (int(round(v * s)) for v in box)
        roi = self._diff[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        return float(roi.mean()) if roi.size else None

    @property
    def baseline(self) -> float:
        return max(self.floor, float(np.median(self._samples))) if self._samples else self.floor

    def ratio(self, box: Box) -> float:
        e = self.energy(box)
        return 0.0 if e is None else e / self.baseline
