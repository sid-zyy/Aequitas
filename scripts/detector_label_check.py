"""Score person detectors on real violence CCTV with hand-verified boxes (no ground truth ships with this footage).

    .venv/Scripts/python scripts/detector_label_check.py propose    # run every detector in MODELS, save boxes
    .venv/Scripts/python scripts/detector_label_check.py sheets     # candidates + numbered crop sheets
    (label results/detector_labels/labels.json; full-frame views help: results/detector_labels/full/)
    .venv/Scripts/python scripts/detector_label_check.py overview   # verified boxes drawn, to count missed people
    .venv/Scripts/python scripts/detector_label_check.py score

Frames: the faculty sample clips (data/footage/faculty/*.mp4, data/footage/*_converted*.avi not already there) at 1/3
and 2/3 of each clip, and one middle frame from a fixed random sample of the Surveillance Camera Fight Dataset
(Akti et al. 2019, MIT, data/videos/survfight: real CCTV from YouTube, fight and noFight).

Candidates: the union of every detector's boxes at confidence >= PROPOSE_CONF, merged at IoU >= 0.5. labels.json maps
frame -> {"people": [candidates that are one person's visible extent], "missed": people no candidate covers,
"uncertain": judgement calls} or {"skip": reason} (crowded or blurred frames that can't be labelled reliably).

score: recall (verified people found at IoU >= 0.5; missed people count against every detector) and precision per
detector at person confidence 0.4 (production: config/settings.yaml) and 0.25. Both are >= PROPOSE_CONF, so every
scored box was a candidate someone judged. Reported for all labelled frames, the clear ones only, the faculty clips
and the fight dataset. Write-up: docs/DETECTION.md.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from detector_compare import CANDIDATES, ap, iou  # noqa: E402

OUT = Path("results/detector_labels")
MODELS = ["yolo11n 640 (today)", "yolo11m 640", "yolo11m 960", "yolo26m 640", "yolo26l 960",
          "rtdetr-l 640 (Ultralytics)", "rfdetr small (512)", "rfdetr medium (576)", "d-fine-m obj2coco"]
PROPOSE_CONF = 0.25  # the lower scored operating point: every scored box gets checked


def frames() -> list[tuple[str, np.ndarray]]:
    out = []

    def grab(path: Path, fracs):
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for f in fracs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * f))
            ok, img = cap.read()
            if ok:
                out.append((f"{path.parent.name}/{path.stem}@{f:.2f}", img))

    fac = sorted(Path("data/footage/faculty").glob("*.mp4"))
    have = {p.stem for p in fac}
    extra = [p for p in sorted(Path("data/footage").glob("*_converted*.avi")) if p.stem.split("_")[0] not in have]
    for p in fac + extra:
        grab(p, (1 / 3, 2 / 3))
    rng = random.Random(0)
    for kind, k in (("fight", 20), ("noFight", 10)):
        for p in rng.sample(sorted(Path(f"data/videos/survfight/{kind}").glob("*.mp4")), k):
            grab(p, (0.5,))
    return out


def merge(boxes: list[np.ndarray]) -> list[np.ndarray]:
    keep: list[np.ndarray] = []
    for b in sorted(boxes, key=lambda x: -(x[2] - x[0]) * (x[3] - x[1])):
        if not keep or iou([b], keep).max() < 0.5:
            keep.append(b)
    return keep


def propose() -> None:
    import torch

    fr = frames()
    print(f"{len(fr)} frames", flush=True)
    dets: dict[str, list] = {}
    for name in MODELS:
        print(name, flush=True)
        det = CANDIDATES[name]()
        dets[name] = [tuple(x.tolist() for x in det(img)) for _, img in fr]
        del det
        torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"frames": [k for k, _ in fr], "detections": dets}, open(OUT / "proposals.json", "w"))
    print(f"wrote {OUT}/proposals.json", flush=True)


def sheets() -> None:
    """Candidates (union of every detector's boxes at >= PROPOSE_CONF, merged at IoU >= 0.5) as numbered crops for
    labelling, 30 per sheet, plus nothing else: the full-frame check for missed people is `overview`."""
    P = json.load(open(OUT / "proposals.json"))
    fr = dict(frames())
    cands, cells = {}, []
    for i, key in enumerate(P["frames"]):
        pool = [np.array(b) for d in P["detections"].values() for b, sc in zip(*d[i]) if sc >= PROPOSE_CONF]
        boxes = merge(pool)
        cands[key] = [list(map(float, b)) for b in boxes]
        img = fr[key]
        H, W = img.shape[:2]
        for j, b in enumerate(boxes):
            x1, y1, x2, y2 = b
            pw, ph = 0.3 * (x2 - x1) + 8, 0.3 * (y2 - y1) + 8
            cx1, cy1, cx2, cy2 = int(max(0, x1 - pw)), int(max(0, y1 - ph)), int(min(W, x2 + pw)), int(min(H, y2 + ph))
            crop = img[cy1:cy2, cx1:cx2].copy()
            f = min(170 / max(1, crop.shape[1]), 150 / max(1, crop.shape[0]))
            crop = cv2.resize(crop, None, fx=f, fy=f)
            cv2.rectangle(crop, (int((x1 - cx1) * f), int((y1 - cy1) * f)), (int((x2 - cx1) * f), int((y2 - cy1) * f)),
                          (0, 255, 255), 1)
            cell = np.full((172, 180, 3), 40, np.uint8)
            cell[20:20 + crop.shape[0], 5:5 + crop.shape[1]] = crop
            cv2.putText(cell, f"{i}.{j}", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cells.append(cell)
    d = OUT / "sheets"
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.jpg"):
        old.unlink()
    for n in range(0, len(cells), 30):
        group = cells[n:n + 30] + [np.zeros_like(cells[0])] * (-len(cells[n:n + 30]) % 6)
        cv2.imwrite(str(d / f"crops_{n // 30:02d}.jpg"), np.vstack([np.hstack(group[r:r + 6]) for r in range(0, len(group), 6)]))
    P["candidates"] = cands
    json.dump(P, open(OUT / "proposals.json", "w"))
    print(f"{sum(map(len, cands.values()))} candidates on {(len(cells) + 29) // 30} sheets", flush=True)


def overview() -> None:
    """Each frame with the verified people drawn, to count people no detector found ("missed")."""
    P = json.load(open(OUT / "proposals.json"))
    L = json.load(open(OUT / "labels.json"))
    fr = dict(frames())
    d = OUT / "overview"
    d.mkdir(parents=True, exist_ok=True)
    for i, key in enumerate(P["frames"]):
        img = fr[key].copy()
        f = max(1.0, 900 / img.shape[1])
        img = cv2.resize(img, None, fx=f, fy=f)
        for j in L.get(key, {}).get("people", []):
            x1, y1, x2, y2 = (int(v * f) for v in P["candidates"][key][j])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{i} {key}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(str(d / f"{i:02d}.jpg"), img)


def score() -> None:
    P = json.load(open(OUT / "proposals.json"))
    L = json.load(open(OUT / "labels.json"))
    keys = [k for k in P["frames"] if "people" in L.get(k, {})]
    fight = ("fight/", "noFight/")
    groups = {
        "all labelled": keys,
        "clear only": [k for k in keys if not L[k]["uncertain"]],
        "faculty": [k for k in keys if not k.startswith(fight)],
        "fight dataset": [k for k in keys if k.startswith(fight)],
    }

    def pr(name: str, ks: list[str], conf: float) -> tuple[float, float]:
        tp = fp = tot = 0
        for k in ks:
            i = P["frames"].index(k)
            g = [P["candidates"][k][j] for j in L[k]["people"]]
            tot += len(g) + L[k]["missed"]
            b, s = np.array(P["detections"][name][i][0]).reshape(-1, 4), np.array(P["detections"][name][i][1])
            b = b[s >= conf][np.argsort(-s[s >= conf])]
            m = iou(b, g) if len(g) else np.zeros((len(b), 0))
            used: set[int] = set()
            for r in range(len(b)):
                row = [m[r, j] if j not in used else -1 for j in range(m.shape[1])]
                j = int(np.argmax(row)) if row else -1
                if j >= 0 and row[j] >= 0.5:
                    used.add(j)
                    tp += 1
                else:
                    fp += 1
        return tp / max(tot, 1), tp / max(tp + fp, 1)

    def people(ks):
        return sum(len(L[k]["people"]) + L[k]["missed"] for k in ks)

    skipped = [k for k in P["frames"] if "skip" in L.get(k, {})]
    head = (f"\nHand-verified people on real violence CCTV: {len(keys)} frames labelled, {people(keys)} people "
            f"({sum(L[k]['missed'] for k in keys)} missed by every detector); {len(skipped)} crowded or blurred frames not "
            "labelled. Recall / precision at person confidence 0.4 (production) unless noted. People per group: "
            + ", ".join(f"{g} {people(ks)}" for g, ks in groups.items()) + ".\n\n"
            "| Detector | All: R@0.4 | P@0.4 | R@0.25 | P@0.25 | Clear only: R | P | Faculty: R | P | Fight dataset: R | P |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n")
    body = ""
    for name in P["detections"]:
        cells = [*pr(name, groups["all labelled"], 0.4), *pr(name, groups["all labelled"], 0.25)]
        for g in ("clear only", "faculty", "fight dataset"):
            cells += pr(name, groups[g], 0.4)
        body += f"| {name} | " + " | ".join(f"{v:.2f}" for v in cells) + " |\n"
    print(head + body)
    with open(OUT / "README.md", "a", encoding="utf-8") as f:
        f.write(head + body)


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("cmd", choices=["propose", "sheets", "overview", "score"])
    {"propose": propose, "sheets": sheets, "overview": overview, "score": score}[ap_.parse_args().cmd]()
