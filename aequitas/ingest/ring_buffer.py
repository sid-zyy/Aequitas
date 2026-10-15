"""Per-camera buffer of recent frames and their track boxes, used to build keyframe grids."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from aequitas.types import Box


@dataclass
class BufferedFrame:
    ts_ms: int
    image: np.ndarray  # BGR, downscaled to max_width
    scale: float  # buffered pixels / original pixels
    boxes: dict[int, Box]  # track_id -> box in ORIGINAL pixel coordinates (persons and objects)


class RingBuffer:
    def __init__(self, seconds: float, max_width: int = 960):
        self.span_ms = int(seconds * 1000)
        self.max_width = max_width
        self._frames: deque[BufferedFrame] = deque()

    def push(self, ts_ms: int, image: np.ndarray, boxes: dict[int, Box]) -> None:
        h, w = image.shape[:2]
        scale = min(1.0, self.max_width / w)
        small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()
        self._frames.append(BufferedFrame(ts_ms, small, scale, dict(boxes)))
        while self._frames and ts_ms - self._frames[0].ts_ms > self.span_ms:
            self._frames.popleft()

    def __len__(self) -> int:
        return len(self._frames)

    def pick(self, onset_ms: int, trigger_ms: int, n: int = 4) -> list[BufferedFrame]:
        """n frames spread evenly from onset to trigger. If onset is older than the buffer,
        the window starts at the oldest buffered frame."""
        if not self._frames:
            return []
        frames = list(self._frames)
        lo = max(onset_ms, frames[0].ts_ms)
        hi = max(lo, min(trigger_ms, frames[-1].ts_ms))
        targets = [lo + (hi - lo) * i / max(1, n - 1) for i in range(n)]
        return [min(frames, key=lambda f: abs(f.ts_ms - t)) for t in targets]
