"""Violence clip model on held-out real CCTV: the Surveillance Camera Fight Dataset (Akti et al. 2019, MIT; 150 fight
+ 150 noFight YouTube CCTV clips, ~2 s each). The model was trained on SCVD, RLVS and RWF-2000, not on these.

    .venv/Scripts/python scripts/violence_survfight.py [--dir data/videos/survfight]

Frames are sampled at the model's clip_fps (4, the pipeline's rate). The model scores 8-frame windows (2 s), stride
2 frames; a clip's score is its highest window, as a periodic window takes the max (pipeline._periodic_step). Clips
shorter than 8 sampled frames reuse the nearest frame, as ViolenceScorer does. The threshold is the one calibrated on
the training run's validation set (metadata.json); 0.9 is models.violence_alone_threshold. Neither is tuned here. Write-up: docs/DETECTION.md.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from aequitas.config import ModelsCfg
from aequitas.vision.violence import ViolenceModel


def clip_score(model: ViolenceModel, path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    src = cap.get(cv2.CAP_PROP_FPS) or 25
    frames, t, i = [], 0.0, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i / src >= t:
            frames.append(model.prepare(f))
            t += 1 / model.clip_fps
        i += 1
    n = model.clip_len
    if len(frames) < n:  # nearest frame for the missing positions
        frames = [frames[min(len(frames) - 1, round(k * (len(frames) - 1) / (n - 1)))] for k in range(n)]
    return max(model.score(frames[s:s + n]) for s in range(0, len(frames) - n + 1, 2))


def auc(pos: list[float], neg: list[float]) -> float:
    p, q = np.array(pos)[:, None], np.array(neg)[None]
    return float(((p > q) + 0.5 * (p == q)).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/videos/survfight")
    ap.add_argument("--model", default="models/violence_clip")
    a = ap.parse_args()
    model = ViolenceModel(a.model)
    scores, t0 = {}, time.perf_counter()
    for kind in ("fight", "noFight"):
        scores[kind] = [clip_score(model, p) for p in sorted(Path(a.dir, kind).glob("*.mp4"))]
    secs = time.perf_counter() - t0
    nf, nn = len(scores["fight"]), len(scores["noFight"])
    print(f"Violence clip model on the Surveillance Camera Fight Dataset ({nf} fight, {nn} noFight), "
          f"ROC AUC {auc(scores['fight'], scores['noFight']):.3f}")
    # calibrated threshold (fight trigger, votes) and the existing 'clip model alone' threshold; neither tuned here
    for label, thr in (("calibrated", model.threshold), ("alone (models.violence_alone_threshold)",
                                                         ModelsCfg().violence_alone_threshold)):
        tp = sum(s >= thr for s in scores["fight"])
        fp = sum(s >= thr for s in scores["noFight"])
        prec, rec = tp / max(tp + fp, 1), tp / nf
        print(f"  {thr:.2f} {label}: accuracy {(tp + nn - fp) / (nf + nn):.3f}, fight recall {rec:.3f}, "
              f"false alarms {fp}/{nn} ({fp / nn:.3f}), precision {prec:.3f}, F1 {2 * prec * rec / max(prec + rec, 1e-9):.3f}")
    print(f"  {1000 * secs / (nf + nn):.0f} ms per clip including decoding (CPU ONNX)")


if __name__ == "__main__":
    main()
