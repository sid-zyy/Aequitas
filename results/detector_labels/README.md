# Detector check on real violence CCTV (hand-verified people)

Made by `scripts/detector_label_check.py`; write-up in docs/DETECTION.md. `labels.json` holds the labels, `proposals.json` every
detector's boxes. The frame images (`full/`, `sheets/`) are local only.

Hand-verified people on real violence CCTV: 33 frames labelled, 118 people (7 missed by every detector); 9 crowded or blurred frames not labelled. Recall / precision at person confidence 0.4 (production) unless noted. People per group: all labelled 118, clear only 46, faculty 37, fight dataset 81.

| Detector | All: R@0.4 | P@0.4 | R@0.25 | P@0.25 | Clear only: R | P | Faculty: R | P | Fight dataset: R | P |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo11n 640 (today) | 0.47 | 0.86 | 0.53 | 0.70 | 0.61 | 0.82 | 0.49 | 0.86 | 0.46 | 0.86 |
| yolo11m 640 | 0.48 | 0.90 | 0.56 | 0.84 | 0.54 | 0.89 | 0.49 | 0.95 | 0.48 | 0.89 |
| yolo11m 960 | 0.42 | 0.86 | 0.52 | 0.81 | 0.54 | 0.96 | 0.27 | 1.00 | 0.48 | 0.83 |
| yolo26m 640 | 0.53 | 0.87 | 0.60 | 0.84 | 0.65 | 0.91 | 0.41 | 0.94 | 0.58 | 0.85 |
| yolo26l 960 | 0.53 | 0.85 | 0.58 | 0.81 | 0.65 | 0.91 | 0.51 | 0.95 | 0.54 | 0.81 |
| rtdetr-l 640 (Ultralytics) | 0.67 | 0.84 | 0.75 | 0.60 | 0.76 | 0.90 | 0.81 | 0.88 | 0.60 | 0.82 |
| rfdetr small (512) | 0.70 | 0.87 | 0.81 | 0.66 | 0.78 | 0.90 | 0.84 | 0.94 | 0.64 | 0.84 |
| rfdetr medium (576) | 0.72 | 0.93 | 0.81 | 0.67 | 0.78 | 0.95 | 0.81 | 1.00 | 0.68 | 0.90 |
| d-fine-m obj2coco | 0.69 | 0.68 | 0.85 | 0.39 | 0.74 | 0.74 | 0.78 | 0.74 | 0.65 | 0.65 |
