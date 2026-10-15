"""Scores held-out clips (the faculty's samples) with the trained violence clip model.

    .venv/Scripts/python scripts/eval_violence.py --model models/violence_clip --videos data/footage/faculty

Label per clip from its name: v* / w* = violent (violence / weaponized), anything else = not violent.
Prints each clip's highest 2 s score, the verdict at the model's calibrated threshold, and the accuracy.
These clips were excluded from training (kaggle/violence_clip.py EXCLUDE_STEMS)."""

import argparse
from pathlib import Path

import cv2

from aequitas.vision.violence import ViolenceModel, ViolenceScorer


def clip_scores(model: ViolenceModel, path: Path, sample_fps: float = 4.0) -> list[float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    scorer, scores, i = ViolenceScorer(model, stride_ms=500), [], 0
    step = max(1, round(fps / sample_fps))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            s = scorer.update(int(i * 1000 / fps), frame)
            if s is not None:
                scores.append(s)
        i += 1
    cap.release()
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/violence_clip")
    ap.add_argument("--videos", default="data/footage/faculty")
    args = ap.parse_args()
    model = ViolenceModel(args.model)
    rows, correct = [], 0
    for p in sorted(Path(args.videos).glob("*")):
        if p.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
            continue
        gold = p.stem[:1].lower() in {"v", "w"}
        scores = clip_scores(model, p)
        top = max(scores) if scores else 0.0
        pred = top >= model.threshold
        correct += pred == gold
        rows.append((p.name, gold, top, pred))
        print(f"{p.name:20} gold={'violent' if gold else 'normal ':8} max score {top:.2f} -> "
              f"{'violent' if pred else 'normal'} {'OK' if pred == gold else 'MISS'}")
    if rows:
        print(f"\n{correct}/{len(rows)} correct at threshold {model.threshold:.2f} (n={len(rows)}: indicative only)")


if __name__ == "__main__":
    main()
