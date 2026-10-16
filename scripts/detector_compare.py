"""Compare person detectors on the RTX 4050: accuracy on CAVIAR's hand-labelled boxes, agreement on the faculty
sample clips, latency, GPU memory, and what the tracker makes of each one's detections.

    .venv/Scripts/python -m pip install rfdetr motmetrics   # evaluation only
    .venv/Scripts/python scripts/detector_compare.py                        # every candidate
    .venv/Scripts/python scripts/detector_compare.py --only yolo26 rfdetr   # names containing these

Detectors (COCO-pretrained, person class only, no fine-tuning): Ultralytics YOLO11 / YOLO26 / RT-DETR, RF-DETR
(rfdetr), D-FINE and DEIMv2 (Hugging Face transformers). Each runs on the GPU in FP16 (RF-DETR in FP32, its predict default), batch 1, at a confidence floor of
0.01, so AP covers the whole curve.

- CAVIAR (8 clips, 4 fps, 384x288, tiny high-angle people): AP50 and AP30 against the hand-labelled boxes, recall
  at the eval config's detector confidence (0.10) for people under and over 30 px tall (median 22 px), and IDF1 / ID switches with Aequitas's
  CameraTracker (bytetrack, activation 0.15) on each detector's boxes.
- Faculty samples (data/footage/faculty, 5 clips, 4 fps, 1280x720): no labels. Pseudo-labels are the boxes on which
  the two strongest models (D-FINE-X Objects365->COCO and RF-DETR-L) agree (IoU >= 0.5, both >= 0.5 confidence). AP50
  against them measures agreement with strong models, not accuracy, and favours those two (marked *).
- Latency: wall time per frame for the whole call (resize, inference, post-processing, copy back), mean and p95 over
  the faculty frames after warm-up, CUDA synchronised. GPU memory: peak allocated by PyTorch during the run.

Results are appended to data/reports/detector_compare.md. Write-up: docs/DETECTION.md.
"""

import argparse
import gc
import os
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from aequitas.config import DetectorCfg, TrackerCfg  # noqa: E402
from aequitas.vision.detector import CameraTracker  # noqa: E402

FLOOR = 0.01


# ---- detectors: frame (BGR) -> (xyxy float32 [N, 4], conf float32 [N]) for people ----

class Ultra:
    def __init__(self, weights: str, imgsz: int):
        from ultralytics import YOLO

        self.m, self.imgsz = YOLO(weights), imgsz

    def __call__(self, f):
        r = self.m.predict(f, imgsz=self.imgsz, conf=FLOOR, classes=[0], half=True, device=0, verbose=False,
                           max_det=300)[0]
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()


class RF:
    def __init__(self, cls: str, resolution: int | None = None):
        import rfdetr

        kw = {"resolution": resolution} if resolution else {}
        self.m = getattr(rfdetr, cls)(**kw)
        try:
            self.m.optimize_for_inference(dtype=torch.float16)
        except Exception as e:  # noqa: BLE001  (older rfdetr / unsupported: run unoptimised and say so)
            print(f"  {cls}: optimize_for_inference failed ({e}); running unoptimised")
        # predict() returns COCO category ids (person = 1), while class_names is a 0-based list
        assert self.m.class_names[0] == "person"
        self.person = 1

    def __call__(self, f):
        d = self.m.predict(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), threshold=FLOOR)
        keep = d.class_id == self.person
        return d.xyxy[keep].astype(np.float32), d.confidence[keep].astype(np.float32)


class HF:
    def __init__(self, repo: str):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.proc = AutoImageProcessor.from_pretrained(repo)
        self.m = AutoModelForObjectDetection.from_pretrained(repo, dtype=torch.float16).cuda().eval()
        self.person = next(int(k) for k, v in self.m.config.id2label.items() if v == "person")

    @torch.inference_mode()
    def __call__(self, f):
        h, w = f.shape[:2]
        x = self.proc(images=cv2.cvtColor(f, cv2.COLOR_BGR2RGB), return_tensors="pt").to("cuda")
        x["pixel_values"] = x["pixel_values"].half()
        out = self.m(**x)
        out.logits, out.pred_boxes = out.logits.float(), out.pred_boxes.float()
        r = self.proc.post_process_object_detection(out, threshold=FLOOR, target_sizes=[(h, w)])[0]
        keep = r["labels"] == self.person
        return r["boxes"][keep].cpu().numpy().astype(np.float32), r["scores"][keep].cpu().numpy().astype(np.float32)


CANDIDATES = {
    # name: (factory, note)
    "yolo11n 640 (today)": lambda: Ultra("yolo11n.pt", 640),
    "yolo11m 960": lambda: Ultra("yolo11m.pt", 960),
    "yolo26n 640": lambda: Ultra("yolo26n.pt", 640),
    "yolo26s 640": lambda: Ultra("yolo26s.pt", 640),
    "yolo26s 960": lambda: Ultra("yolo26s.pt", 960),
    "yolo26m 640": lambda: Ultra("yolo26m.pt", 640),
    "yolo26m 960": lambda: Ultra("yolo26m.pt", 960),
    "yolo26l 960": lambda: Ultra("yolo26l.pt", 960),
    "rtdetr-l 640 (Ultralytics)": lambda: Ultra("rtdetr-l.pt", 640),
    "rfdetr nano (384)": lambda: RF("RFDETRNano"),
    "rfdetr small (512)": lambda: RF("RFDETRSmall"),
    "rfdetr medium (576)": lambda: RF("RFDETRMedium"),
    "rfdetr large (704)*": lambda: RF("RFDETRLarge"),
    "d-fine-s obj2coco": lambda: HF("ustc-community/dfine-small-obj2coco"),
    "d-fine-m obj2coco": lambda: HF("ustc-community/dfine-medium-obj2coco"),
    "d-fine-l obj2coco": lambda: HF("ustc-community/dfine-large-obj2coco-e25"),
    "d-fine-x obj2coco*": lambda: HF("ustc-community/dfine-xlarge-obj2coco"),
    "deimv2-s dinov3": lambda: HF("harshaljanjani/DEIMv2_DINOv3_S_COCO_Transformers"),
    "deimv2-l dinov3": lambda: HF("plainsight-ai/deimv2-dinov3-l-coco"),
}
REFERENCES = ("d-fine-x obj2coco*", "rfdetr large (704)*")


# ---- data ----

def caviar(folder: Path, fps: float):
    import xml.etree.ElementTree as ET

    clips = []
    for mpg in sorted(p for p in folder.glob("*.mpg") if p.with_suffix(".xml").exists()):
        gt = {}
        for fr in ET.parse(mpg.with_suffix(".xml")).getroot().iter("frame"):
            boxes = []
            for o in fr.iter("object"):
                b = {k: float(v) for k, v in o.find("box").attrib.items()}
                boxes.append((int(o.attrib["id"]), (b["xc"] - b["w"] / 2, b["yc"] - b["h"] / 2,
                                                    b["xc"] + b["w"] / 2, b["yc"] + b["h"] / 2)))
            gt[int(fr.attrib["number"])] = boxes
        clips.append((mpg.stem, read(mpg, fps, keep=gt), gt))
    return clips


def read(path: Path, fps: float, keep=None):
    cap = cv2.VideoCapture(str(path))
    step = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 25) / fps))
    out, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            return out
        if i % step == 0 and (keep is None or i in keep):
            out.append((i, f))
        i += 1


# ---- metrics ----

def iou(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float)[:, None], np.asarray(b, float)[None]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])  # noqa: E731
    return iw * ih / np.maximum(area(a) + area(b) - iw * ih, 1e-9)


def ap(frames_gt, frames_det, thr):
    """VOC-style all-point AP over all frames: frames_gt[i] = boxes, frames_det[i] = (boxes, scores)."""
    recs, n_gt = [], sum(len(g) for g in frames_gt)
    for fi, (g, (b, s)) in enumerate(zip(frames_gt, frames_det)):
        recs += [(float(sc), fi, j) for j, sc in enumerate(s)]
    recs.sort(reverse=True)
    used = [np.zeros(len(g), bool) for g in frames_gt]
    ious = [iou(b, g) for g, (b, _) in zip(frames_gt, frames_det)]
    tp = np.zeros(len(recs))
    for k, (_, fi, j) in enumerate(recs):
        if not len(frames_gt[fi]):
            continue
        row = np.where(used[fi], -1, ious[fi][j])
        m = int(np.argmax(row))
        if row[m] >= thr:
            used[fi][m], tp[k] = True, 1
    if not n_gt:
        return float("nan")
    ctp = np.cumsum(tp)
    rec, prec = ctp / n_gt, ctp / np.arange(1, len(tp) + 1)
    prec = np.maximum.accumulate(np.concatenate([prec, [0]])[::-1])[::-1][:-1]
    return float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))


def recall_by_height(frames_gt, frames_det, conf, thr=0.5, split=30):
    hit, tot = np.zeros(2), np.zeros(2)
    for g, (b, s) in zip(frames_gt, frames_det):
        if not len(g):
            continue
        m = iou(b[s >= conf], g)
        for j, box in enumerate(g):
            k = int(box[3] - box[1] >= split)
            tot[k] += 1
            if m.shape[0] and m[:, j].max() >= thr:
                hit[k] += 1
    return hit / np.maximum(tot, 1), tot


def idf1(clips, dets, conf=0.10, activation=0.15):
    import motmetrics as mm

    accs = []
    for (name, frames, gt), d in zip(clips, dets):
        tr = CameraTracker(TrackerCfg(activation_threshold=activation), DetectorCfg(min_person_conf=0,
                                                                                    min_person_height_px=0), 4)
        acc = mm.MOTAccumulator(auto_id=True)
        for (fi, _), (b, s) in zip(frames, d):
            k = s >= conf
            sd = sv.Detections(xyxy=b[k], confidence=s[k], class_id=np.zeros(int(k.sum()), int))
            hyp = [(t.track_id, t.box) for t in tr.update(sd)[0]]
            g = gt.get(fi, [])
            dist = 1 - iou([x for _, x in g], [x for _, x in hyp])
            dist[dist > 0.5] = np.nan
            acc.update([o for o, _ in g], [h for h, _ in hyp], dist)
        accs.append(acc)
    s = mm.metrics.create().compute_many(accs, metrics=["idf1", "num_switches"], generate_overall=True,
                                         names=[c[0] for c in clips])
    return float(s.loc["OVERALL", "idf1"]), int(s.loc["OVERALL", "num_switches"])


def pseudo_labels(ref_a, ref_b, conf=0.5, thr=0.5):
    out = []
    for (ba, sa), (bb, sb) in zip(ref_a, ref_b):
        ba, bb = ba[sa >= conf], bb[sb >= conf]
        m = iou(ba, bb)
        out.append([ba[i] for i in range(len(ba)) if m.shape[1] and m[i].max() >= thr])
    return out


# ---- run ----

def run(det, frames):
    out, times = [], []
    for _, f in frames:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = det(f)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        out.append(r)
    return out, times


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--caviar", default="data/videos/caviar")
    ap_.add_argument("--faculty", default="data/footage/faculty")
    ap_.add_argument("--fps", type=float, default=4)
    ap_.add_argument("--only", nargs="*", default=None)
    ap_.add_argument("--out", default="data/reports/detector_compare.md")
    a = ap_.parse_args()

    cav = caviar(Path(a.caviar), a.fps)
    fac = [f for p in sorted(Path(a.faculty).glob("*.mp4")) for f in read(p, a.fps)]
    cav_gt = [[b for _, b in gt.get(fi, [])] for _, frames, gt in cav for fi, _ in frames]
    print(f"CAVIAR: {len(cav)} clips, {len(cav_gt)} frames, {sum(map(len, cav_gt))} labelled people; "
          f"faculty: {len(fac)} frames", flush=True)

    names = [n for n in CANDIDATES if not a.only or any(o in n for o in a.only)]
    for r in REFERENCES:  # needed for the faculty pseudo-labels
        if r not in names:
            names.append(r)
    res = {}
    for name in names:
        print(f"{name} ...", flush=True)
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            det = CANDIDATES[name]()
            warm = fac[:10]
            run(det, warm)
            fac_d, t = run(det, fac)
            cav_d = [run(det, frames)[0] for _, frames, _ in cav]
            mem = torch.cuda.max_memory_allocated() / 2**20
        except Exception as e:  # noqa: BLE001  (report and go on: one broken model shouldn't stop the run)
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        finally:
            det = None
            gc.collect()
        flat = [x for d in cav_d for x in d]
        rec, tot = recall_by_height(cav_gt, flat, conf=0.10)
        i1, sw = idf1(cav, cav_d)
        res[name] = dict(fac=fac_d, ms=1000 * np.mean(t), p95=1000 * np.percentile(t, 95), mem=mem,
                         ap50=ap(cav_gt, flat, 0.5), ap30=ap(cav_gt, flat, 0.3), rec_small=rec[0], rec_big=rec[1],
                         idf1=i1, sw=sw)
        r = res[name]
        print(f"  CAVIAR AP50 {r['ap50']:.3f} AP30 {r['ap30']:.3f} recall@0.10 <30px {rec[0]:.3f} >=30px {rec[1]:.3f} "
              f"IDF1 {i1:.3f} sw {sw} | {r['ms']:.1f} ms (p95 {r['p95']:.1f}) {mem:.0f} MiB", flush=True)

    pl = pseudo_labels(res[REFERENCES[0]]["fac"], res[REFERENCES[1]]["fac"]) if all(r in res for r in REFERENCES) else None
    head = (f"\nDetector comparison, RTX 4050 laptop, FP16 batch 1, person class. CAVIAR: {len(cav)} clips, {len(cav_gt)} "
            f"frames at {a.fps:g} fps, {sum(map(len, cav_gt))} labelled boxes (<30 px tall: {int(tot[0])}). Faculty: "
            f"{len(fac)} frames, {sum(map(len, pl)) if pl else 0} pseudo-labelled people (* = a pseudo-label source).\n\n"
            "| Detector | CAVIAR AP50 | AP30 | Recall@0.10 <30 px | >=30 px | IDF1 | ID sw | Faculty AP50 (pseudo) "
            "| ms/frame mean | p95 | GPU MiB |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
    body = ""
    for name, r in res.items():
        fa = ap(pl, r["fac"], 0.5) if pl else float("nan")
        body += (f"| {name} | {r['ap50']:.3f} | {r['ap30']:.3f} | {r['rec_small']:.3f} | {r['rec_big']:.3f} | "
                 f"{r['idf1']:.3f} | {r['sw']} | {fa:.3f} | {r['ms']:.1f} | {r['p95']:.1f} | {r['mem']:.0f} |\n")
    print(head + body)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(head + body)


if __name__ == "__main__":
    main()
