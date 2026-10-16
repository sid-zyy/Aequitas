"""Detector factory and the RF-DETR adapter with a fake model (no weights needed)."""

import numpy as np
import supervision as sv

from aequitas.config import DetectorCfg
from aequitas.vision.detector import RFDetrDetector


class FakeRF:
    """Stands in for rfdetr.RFDETR*: predict() returns COCO category ids (person = 1, backpack = 27, ...)."""

    def __init__(self, xyxy, conf, coco_ids):
        self.out = sv.Detections(xyxy=np.array(xyxy, float), confidence=np.array(conf, float),
                                 class_id=np.array(coco_ids, int))
        self.seen = None

    def predict(self, image, threshold):
        self.seen = (image, threshold)
        return self.out


def test_rfdetr_maps_coco91_ids_and_drops_other_classes():
    fake = FakeRF([[0, 0, 10, 20], [1, 1, 5, 5], [2, 2, 6, 6], [3, 3, 7, 7], [4, 4, 8, 8], [5, 5, 9, 9]],
                  [0.9, 0.8, 0.7, 0.6, 0.5, 0.4], [1, 27, 31, 33, 3, 62])  # person, backpack, handbag, suitcase, car, chair
    d = RFDetrDetector(DetectorCfg(family="rfdetr"), "cpu", model=fake)(np.zeros((20, 30, 3), np.uint8))
    assert d.class_id.tolist() == [0, 24, 26, 28]
    assert d.confidence.tolist() == np.float32([0.9, 0.8, 0.7, 0.6]).tolist()
    assert d.xyxy.shape == (4, 4) and d.xyxy.dtype == np.float32


def test_rfdetr_gets_rgb_and_the_configured_confidence():
    fake = FakeRF(np.zeros((0, 4)), [], [])
    img = np.zeros((4, 4, 3), np.uint8)
    img[..., 0] = 255  # blue in BGR
    d = RFDetrDetector(DetectorCfg(family="rfdetr", conf=0.3), "cpu", model=fake)(img)
    seen, thr = fake.seen
    assert thr == 0.3
    assert (seen[..., 2] == 255).all() and (seen[..., 0] == 0).all()
    assert len(d) == 0


def test_detector_name_for_versions():
    assert DetectorCfg().name() == "yolo11n.pt"
    assert DetectorCfg(family="rfdetr", rfdetr_size="medium").name() == "rfdetr-medium"
