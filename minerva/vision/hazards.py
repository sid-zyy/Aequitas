"""Runs the hazard detector trained by kaggle/hazard_detector.py: fire, smoke, knife, gun, stick.

It runs next to the COCO person/bag detector. Results go into the Scene for rules and, later, the
per-frame timeline (docs/ROADMAP.md step 2)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from minerva.types import Box

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Hazard:
    label: str
    conf: float
    box: Box


class HazardDetector:
    def __init__(self, model_dir: str | Path, device: str, conf: float = 0.35):
        from ultralytics import YOLO

        d = Path(model_dir)
        self.meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        weights = d / "hazard_detector.pt"
        if not weights.exists():
            weights = next(d.glob("*.onnx"))
        self.model = YOLO(str(weights), task="detect")
        self.names = self.meta["classes"]
        self.imgsz = int(self.meta.get("imgsz", 640))
        self.device = device if weights.suffix == ".pt" else None
        self.conf = conf
        log.info("hazard detector: %s classes=%s", weights.name, self.names)

    def __call__(self, image: np.ndarray) -> list[Hazard]:
        r = self.model.predict(image, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        if r.boxes is None:
            return []
        return [Hazard(self.names[int(c)], float(s), tuple(map(float, b)))
                for b, s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
