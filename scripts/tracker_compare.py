"""Compare person trackers on CAVIAR's hand-labelled tracks, with identical YOLO detections for every tracker.

    .venv/Scripts/python -m pip install motmetrics deep-sort-realtime   # evaluation only
    bash scripts/get_caviar.sh   # clips; ground truth: data/videos/caviar/<clip>.xml (see the CAVIAR page)
    .venv/Scripts/python scripts/tracker_compare.py --fps 4

Trackers: Aequitas's CameraTracker (aequitas/vision/detector.py: Ultralytics ByteTrack or OC-SORT with our
thresholds), supervision ByteTrack as Aequitas used it before (while supervision < 0.31 has it), Ultralytics
ByteTrack, BoT-SORT (camera-motion compensation, no ReID), OC-SORT, and DeepSORT (deep-sort-realtime, MobileNetV2
appearance embeddings). Metrics (py-motmetrics, IoU >= 0.5): IDF1 (identity kept over time: what loitering and
tailgating need), MOTA, ID switches, recall, precision; tracker time per frame with detections (detection
excluded, embedder included). Results are appended to data/reports/tracker_compare.md. Write-up: docs/TRACKING.md.
"""

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import cv2
import motmetrics as mm
import numpy as np
import supervision as sv

from aequitas.config import DetectorCfg, TrackerCfg
from aequitas.vision.detector import Boxes, CameraTracker


def load_gt(xml: Path) -> dict[int, list[tuple[int, tuple[float, float, float, float]]]]:
    gt = {}
    for fr in ET.parse(xml).getroot().iter("frame"):
        objs = []
        for o in fr.iter("object"):
            b = o.find("box").attrib
            w, h, xc, yc = (float(b[k]) for k in ("w", "h", "xc", "yc"))
            objs.append((int(o.attrib["id"]), (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)))
        gt[int(fr.attrib["number"])] = objs
    return gt


def make_trackers(fps: float, cfg: TrackerCfg, device: str):
    from deep_sort_realtime.deepsort_tracker import DeepSort
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.trackers.oc_sort import OCSORT
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml

    buffer_s = cfg.lost_track_buffer / 4  # Aequitas sets frames at 4 fps; keep the same seconds at any fps
    rate = max(1, round(fps))

    def ul(cls, yaml_name, **over):
        args = YAML.load(check_yaml(yaml_name))
        # same thresholds as Aequitas's config for a fair comparison
        args.update(track_high_thresh=cfg.activation_threshold, new_track_thresh=cfg.activation_threshold,
                    track_buffer=int(buffer_s * rate), match_thresh=cfg.matching_threshold, **over)
        return cls(SimpleNamespace(**args))  # track_buffer is in frames (max_frames_lost)

    frames_cfg = cfg.model_copy(update={"lost_track_buffer": int(buffer_s * rate)})
    no_filter = DetectorCfg(min_person_conf=0, min_person_height_px=0)  # tracker only, like the other rows

    def aequitas(**over):
        tr = CameraTracker(frames_cfg.model_copy(update=over), no_filter, fps)
        return ("aq", tr)

    old = {}
    if hasattr(sv, "ByteTrack"):  # removed in supervision 0.31; Aequitas used it (with activation - 0.1) until 2026-10
        old["ByteTrack (supervision, old Aequitas)"] = ("sv", sv.ByteTrack(
            track_activation_threshold=max(0.05, cfg.activation_threshold - 0.1), lost_track_buffer=int(buffer_s * rate),
            minimum_matching_threshold=cfg.matching_threshold, frame_rate=rate))
    return {
        **old,
        "CameraTracker bytetrack (Aequitas)": aequitas(algorithm="bytetrack"),
        "CameraTracker bytetrack, fuse_score off": aequitas(algorithm="bytetrack", fuse_score=False),
        "CameraTracker ocsort": aequitas(algorithm="ocsort"),
        "ByteTrack (Ultralytics)": ("ul", ul(BYTETracker, "bytetrack.yaml")),
        "BoT-SORT (GMC, no ReID)": ("ul", ul(BOTSORT, "botsort.yaml", with_reid=False)),
        "OC-SORT": ("ul", ul(OCSORT, "ocsort.yaml")),
        "DeepSORT (MobileNetV2 ReID)": ("ds", DeepSort(max_age=int(buffer_s * rate), n_init=2,
                                                       embedder="mobilenet", half=True,
                                                       embedder_gpu=device.startswith("cuda"))),
    }


def run_tracker(kind, tr, frame, xyxy, conf):
    """-> list of (track id, (x1, y1, x2, y2))"""
    if kind == "sv":
        d = sv.Detections(xyxy=xyxy.astype(np.float32), confidence=conf.astype(np.float32),
                          class_id=np.zeros(len(conf), int))
        out = tr.update_with_detections(d)
        return [] if out.tracker_id is None else [(int(t), tuple(b)) for b, t in zip(out.xyxy, out.tracker_id)]
    if kind == "aq":
        d = sv.Detections(xyxy=xyxy.astype(np.float32), confidence=conf.astype(np.float32),
                          class_id=np.zeros(len(conf), int))
        return [(t.track_id, t.box) for t in tr.update(d)[0]]
    if kind == "ul":
        out = tr.update(Boxes(xyxy, conf, np.zeros(len(conf))), frame)
        return [(int(r[4]), tuple(r[:4])) for r in out]
    raw = [([float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], float(c), 0) for b, c in zip(xyxy, conf)]
    tracks = tr.update_tracks(raw, frame=frame)
    return [(int(t.track_id), tuple(t.to_ltrb())) for t in tracks if t.is_confirmed() and t.time_since_update == 0]


def iou_dist(a, b):
    if not len(a) or not len(b):
        return np.empty((len(a), len(b)))
    a, b = np.asarray(a, float)[:, None], np.asarray(b, float)[None]  # x1, y1, x2, y2
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])  # noqa: E731
    iou = inter / np.maximum(area(a) + area(b) - inter, 1e-9)
    d = 1 - iou  # motmetrics' iou_matrix (it uses np.asfarray, gone in NumPy 2): no match below IoU 0.5
    d[iou < 0.5] = np.nan
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/videos/caviar")
    ap.add_argument("--fps", type=float, default=4, help="frames per second fed to the trackers (pipeline: 4)")
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.10)  # eval_caviar_v2.yaml
    ap.add_argument("--activation", type=float, default=0.15)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="data/reports/tracker_compare.md")
    a = ap.parse_args()
    from ultralytics import YOLO

    yolo = YOLO(a.weights)
    cfg = TrackerCfg(activation_threshold=a.activation)
    clips = sorted(p for p in Path(a.dir).glob("*.mpg") if p.with_suffix(".xml").exists())
    accs: dict[str, list] = {}
    times: dict[str, list[float]] = {}
    names = []
    for clip in clips:
        gt = load_gt(clip.with_suffix(".xml"))
        cap = cv2.VideoCapture(str(clip))
        src = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, round(src / a.fps))
        frames, dets = [], []
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i % step == 0 and i in gt:
                r = yolo.predict(f, imgsz=a.imgsz, conf=a.conf, classes=[0], device=0 if a.device == "cuda" else "cpu",
                                 verbose=False)[0]
                frames.append((i, f))
                dets.append((r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()))
            i += 1
        trackers = make_trackers(a.fps, cfg, a.device)
        names = list(trackers)
        for name, (kind, tr) in trackers.items():
            acc = mm.MOTAccumulator(auto_id=True)
            spent = []
            for (fi, f), (xyxy, conf) in zip(frames, dets):
                t0 = time.perf_counter()
                hyp = run_tracker(kind, tr, f, xyxy, conf)
                if len(conf):  # time only frames with detections (DeepSORT embeds each one)
                    spent.append(time.perf_counter() - t0)
                g = gt.get(fi, [])
                acc.update([o for o, _ in g], [h for h, _ in hyp], iou_dist([b for _, b in g], [b for _, b in hyp]))
            accs.setdefault(name, []).append(acc)
            times.setdefault(name, []).extend(spent)
        print(f"{clip.stem}: {len(frames)} frames at {src / step:.1f} fps, "
              f"{sum(len(d[1]) for d in dets) / max(1, len(dets)):.1f} detections per frame", flush=True)

    mh = mm.metrics.create()
    metrics = ["idf1", "mota", "num_switches", "recall", "precision", "mostly_tracked", "num_unique_objects"]
    rows = []
    for name in names:
        s = mh.compute_many(accs[name], metrics=metrics, names=[c.stem for c in clips], generate_overall=True)
        o = s.loc["OVERALL"]
        rows.append((name, o.idf1, o.mota, int(o.num_switches), o.recall, o.precision,
                     f"{int(o.mostly_tracked)}/{int(o.num_unique_objects)}", 1000 * float(np.mean(times[name]))))
    head = (f"Tracker comparison on CAVIAR ({len(clips)} clips: {', '.join(c.stem for c in clips)}), {a.fps:g} fps, "
            f"{a.weights} imgsz {a.imgsz} conf {a.conf}, activation {a.activation}, IoU >= 0.5\n\n"
            "| Tracker | IDF1 | MOTA | ID switches | Recall | Precision | Mostly tracked | ms/frame with detections (mean) |\n"
            "|---|---|---|---|---|---|---|---|\n")
    body = "".join(f"| {n} | {i:.3f} | {m:.3f} | {s} | {r:.3f} | {p:.3f} | {mt} | {t:.2f} |\n"
                   for n, i, m, s, r, p, mt, t in rows)
    print("\n" + head + body)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(head + body + "\n")


if __name__ == "__main__":
    main()
