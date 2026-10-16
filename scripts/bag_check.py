"""Can the detector see the left bag? CAVIAR LeftBag / LeftBag_PickedUp label the bag as an object whose role is
"leaving object" (LeftBag: 16x22 px, frames 945-1354). Abandoned-object detection needs it as backpack/handbag/suitcase.

    .venv/Scripts/python scripts/bag_check.py

Per detector (aequitas.vision.detector.Detector, so the same class mapping the app uses): over the frames where the bag
is labelled, sampled at 4 fps, the share with a bag-class box overlapping it (IoU >= 0.3: the bag is tiny) at detector
confidence >= 0.10 and >= 0.25, and the best bag-class confidence seen there. Write-up: docs/DETECTION.md.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from aequitas.config import DetectorCfg
from aequitas.vision.detector import Detector

BAGS = (24, 26, 28)
SETUPS = {
    "yolo11n 640 (in use)": DetectorCfg(conf=0.05),
    "yolo11m 960": DetectorCfg(weights="yolo11m.pt", imgsz=960, conf=0.05),
    "yolo26m 960": DetectorCfg(weights="yolo26m.pt", imgsz=960, conf=0.05),
    "rfdetr small": DetectorCfg(family="rfdetr", rfdetr_size="small", conf=0.05),
    "rfdetr medium": DetectorCfg(family="rfdetr", rfdetr_size="medium", conf=0.05),
}


def bag_frames(xml: Path) -> dict[int, tuple[float, float, float, float]]:
    out = {}
    for fr in ET.parse(xml).getroot().iter("frame"):
        for o in fr.find("objectlist").iter("object"):
            if o.find(".//role").text.strip() == "leaving object":
                b = {k: float(v) for k, v in o.find("box").attrib.items()}
                out[int(fr.attrib["number"])] = (b["xc"] - b["w"] / 2, b["yc"] - b["h"] / 2,
                                                 b["xc"] + b["w"] / 2, b["yc"] + b["h"] / 2)
    return out


def iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9)


def main() -> None:
    clips = []
    for name in ("LeftBag", "LeftBag_PickedUp"):
        mpg, xml = Path(f"data/videos/caviar/{name}.mpg"), Path(f"data/videos/caviar/{name}.xml")
        if not (mpg.exists() and xml.exists()):
            print(f"{name}: clip or labels missing, skipped")
            continue
        gt = bag_frames(xml)
        cap, frames, i = cv2.VideoCapture(str(mpg)), [], 0
        step = round((cap.get(cv2.CAP_PROP_FPS) or 25) / 4)
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i in gt and i % step == 0:
                frames.append((gt[i], f))
            i += 1
        clips.append((name, frames))
        print(f"{name}: bag labelled in {len(gt)} frames; {len(frames)} at 4 fps; size {gt[min(gt)][2] - gt[min(gt)][0]:.0f}"
              f"x{gt[min(gt)][3] - gt[min(gt)][1]:.0f} px")
    print("\n| Detector | " + " | ".join(f"{n}: hit @0.10 | hit @0.25 | best conf" for n, _ in clips) + " |")
    print("|---|" + "---|---|---|" * len(clips))
    for label, cfg in SETUPS.items():
        det = Detector(cfg, "cuda:0")
        cells = []
        for _, frames in clips:
            hits10 = hits25 = 0
            best = 0.0
            for box, f in frames:
                d = det(f)
                s = [float(c) for b, c, k in zip(d.xyxy, d.confidence, d.class_id) if k in BAGS and iou(b, box) >= 0.3]
                m = max(s, default=0.0)
                best = max(best, m)
                hits10 += m >= 0.10
                hits25 += m >= 0.25
            cells += [f"{hits10 / len(frames):.2f}", f"{hits25 / len(frames):.2f}", f"{best:.2f}"]
        print(f"| {label} | " + " | ".join(cells) + " |", flush=True)
        del det


if __name__ == "__main__":
    main()
