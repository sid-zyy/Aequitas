"""Runs the violence clip classifier trained by kaggle/violence_clip.py (ONNX + metadata.json).

The model was trained on clips of `clip_len` frames at `clip_fps`. Frames are picked from a short
per-camera buffer by timestamp, so the pipeline's own sample rate does not have to match exactly."""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


class ViolenceModel:
    def __init__(self, model_dir: str | Path):
        import onnxruntime as ort

        d = Path(model_dir)
        self.meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        onnx = next(d.glob("*.onnx"))
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(onnx), providers=providers)
        self.input = self.session.get_inputs()[0].name
        m = self.meta
        self.clip_len, self.clip_fps = int(m["clip_len"]), float(m["clip_fps"])
        self.short, self.crop = int(m["short_side"]), int(m["crop"])
        self.mean = np.array(m["mean"], np.float32)
        self.std = np.array(m["std"], np.float32)
        self.temperature = float(m.get("temperature", 1.0))
        self.threshold = float(m.get("threshold", 0.5))
        log.info("violence model: %s (%d frames @ %g fps, threshold %.2f)", onnx.name, self.clip_len,
                 self.clip_fps, self.threshold)

    def prepare(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Resize the short side and centre-crop: the same as validation preprocessing in training."""
        h, w = frame_bgr.shape[:2]
        s = self.short / min(h, w)
        img = cv2.resize(frame_bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        y, x = (h - self.crop) // 2, (w - self.crop) // 2
        return img[y:y + self.crop, x:x + self.crop, ::-1]  # BGR -> RGB

    def score(self, frames_rgb: list[np.ndarray]) -> float:
        clip = np.stack(frames_rgb).astype(np.float32) / 255.0
        clip = (clip - self.mean) / self.std
        x = clip.transpose(3, 0, 1, 2)[None]  # 1,C,T,H,W
        logits = self.session.run(None, {self.input: np.ascontiguousarray(x)})[0][0] / self.temperature
        e = np.exp(logits - logits.max())
        return float(e[1] / e.sum())


class ViolenceScorer:
    """Per camera: keeps recent prepared frames and scores the last clip every `stride_ms`."""

    def __init__(self, model: ViolenceModel, stride_ms: int = 500):
        self.model = model
        self.stride_ms = stride_ms
        self.span_ms = int(1000 * model.clip_len / model.clip_fps)
        self._buf: deque[tuple[int, np.ndarray]] = deque()
        self._last_run: int | None = None
        self.last_score: float | None = None

    def update(self, ts_ms: int, frame_bgr: np.ndarray) -> float | None:
        self._buf.append((ts_ms, self.model.prepare(frame_bgr)))
        while self._buf and ts_ms - self._buf[0][0] > self.span_ms + 1000:
            self._buf.popleft()
        if ts_ms - self._buf[0][0] < 0.75 * self.span_ms:
            return self.last_score  # not enough history yet
        if self._last_run is not None and ts_ms - self._last_run < self.stride_ms:
            return self.last_score
        self._last_run = ts_ms
        step = 1000.0 / self.model.clip_fps
        targets = [ts_ms - (self.model.clip_len - 1 - i) * step for i in range(self.model.clip_len)]
        frames = [min(self._buf, key=lambda f: abs(f[0] - t))[1] for t in targets]
        self.last_score = self.model.score(frames)
        return self.last_score
