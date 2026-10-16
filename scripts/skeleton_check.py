"""Run YOLO pose + STGCN++ (aequitas/vision/skeleton.py) over video files in 4 s windows at 4 fps, the way the
periodic pipeline sees a camera, and print the top NTU actions and the incident score per window.

    .venv/Scripts/python scripts/skeleton_check.py data/footage/incidents/punching.mp4 data/footage/faculty/v001.mp4

    .venv/Scripts/python scripts/skeleton_check.py --record 20     # film 20 s from camera 0 first, then check it

--record saves the clip to data/eval/webcam_<time>.mp4 (a countdown first: --countdown, default 3 s). Close the browser tab that uses
the webcam before recording; Windows lets one program use the camera at a time.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from aequitas.vision.skeleton import NTU120, PoseSequence, SkeletonActionModel, incident_from_probs


def windows(path: str, fps: float, window_s: float):
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, round(src_fps / fps))
    per_win = int(round(window_s * fps))
    buf, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            buf.append(frame)
            if len(buf) == per_win:
                yield buf
                buf = []
        i += 1
    if len(buf) >= fps:  # tail of at least a second
        yield buf


def record(seconds: float, camera: int, countdown: int = 3) -> str:
    out = Path("data/eval") / f"webcam_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(camera)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"camera {camera} gave no picture (privacy switch, or another program has it)")
    for k in range(countdown, 0, -1):
        print(f"recording in {k}...", flush=True)
        time.sleep(1)
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
    t0, n = time.time(), 0
    while time.time() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            break
        if n < (time.time() - t0) * 15:  # about 15 fps whatever the camera gives
            writer.write(frame)
            n += 1
    writer.release()
    cap.release()
    print(f"saved {out} ({n} frames)")
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--record", type=float, default=0, help="seconds to film from the local camera first")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--countdown", type=int, default=3, help="seconds before recording starts")
    ap.add_argument("--fps", type=float, default=4)
    ap.add_argument("--window", type=float, default=4)
    ap.add_argument("--models", default="models/skeleton")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.record:
        a.videos.append(record(a.record, a.camera, a.countdown))
    if not a.videos:
        ap.error("give video files or --record SECONDS")
    model = SkeletonActionModel(a.models, a.device)
    pose = PoseSequence(device=a.device)
    for path in a.videos:
        print(f"\n## {path}")
        for w, frames in enumerate(windows(path, a.fps, a.window)):
            t0 = time.perf_counter()
            for f in frames:
                pose.add(f)
            t1 = time.perf_counter()
            got = pose.take()
            if got is None:
                print(f"  w{w:02d} nobody")
                continue
            kp, sc, wh = got
            p = model.probs(kp, sc, wh)
            t2 = time.perf_counter()
            cat, pc, _ = incident_from_probs(p)
            people = int((sc.mean(-1) > 0.3).any(1).sum())
            top = ", ".join(f"{NTU120[i]} {p[i]:.2f}" for i in np.argsort(-p)[:3])
            print(f"  w{w:02d} people={people} {cat or '-':12} {pc:.2f} | {top} | pose {1000 * (t1 - t0) / len(frames):.0f} ms/frame, "
                  f"action {1000 * (t2 - t1):.0f} ms")


if __name__ == "__main__":
    main()
