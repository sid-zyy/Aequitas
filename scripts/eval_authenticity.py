"""Measures the citizen-portal "AI-generated" frame detector on real and AI video.

Per clip: the per-frame P(AI) of a Hugging Face image AI-detector on the same sampled frames the app uses
(aequitas.authenticity.sample_frames), and its median (the app's frame score). Per detector: score
distribution per set, ROC AUC (real vs AI), and false/true-positive rates at the app's thresholds.

Sets (data/ is gitignored, see results/authenticity/README.md for where the clips come from):
  real   all clips under data/footage except faculty/ (Pexels stock, incidents, staged candidates, demo)
         plus a fixed sample of data/videos/survfight (real CCTV, fight and noFight)
  ai     data/videos/ai_kling (Kling) and data/videos/ai_bmbench (CogVideoX, HunyuanVideo, Mochi, AnimateDiff)
  faculty  data/footage/faculty: REPORTED ONLY, never used to pick a model or a threshold (held-out test)

Scores are cached per detector in results/authenticity/scores/<model>.json, so re-runs only add new clips.

  .venv/Scripts/python scripts/eval_authenticity.py --models Organika/sdxl-detector umm-maybe/AI-image-detector
  .venv/Scripts/python scripts/eval_authenticity.py --report-only
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aequitas import authenticity as A  # noqa: E402

OUT = ROOT / "results" / "authenticity"
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
DEFAULT_MODELS = ["Organika/sdxl-detector", "umm-maybe/AI-image-detector",
                  "Ateeqq/ai-vs-human-image-detector", "haywoodsloan/ai-image-detector-deploy"]


def _videos(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in VIDEO_EXT) if d.exists() else []


def clip_sets(survfight_every: int = 7) -> dict[str, list[tuple[str, Path]]]:
    footage, videos = ROOT / "data" / "footage", ROOT / "data" / "videos"
    real = [("stock/cctv", p) for p in _videos(footage)
            if "faculty" not in p.relative_to(footage).parts and p.parent != footage]  # root *.avi = faculty copies
    sf = videos / "survfight"
    for sub in ("fight", "noFight"):
        real += [("survfight", p) for p in sorted((sf / sub).glob("*.mp4"))[::survfight_every]]
    ai = [("kling", p) for p in _videos(videos / "ai_kling")]
    gen = {}
    meta = videos / "ai_bmbench_dl" / "split.json"
    if meta.exists():
        gen = {Path(r["video"]).name: r["model_name"].split("/")[-1] for r in json.loads(meta.read_text())}
    ai += [(gen.get(p.name, "bmbench"), p) for p in _videos(videos / "ai_bmbench")]
    faculty = [("faculty", p) for p in _videos(footage / "faculty")]
    return {"real": real, "ai": ai, "faculty": faculty}


def _cache_path(model: str, crop_bottom: float = 0.0) -> Path:
    suffix = f"__crop{int(crop_bottom * 100)}" if crop_bottom else ""
    return OUT / "scores" / (model.replace("/", "__") + suffix + ".json")


def score_model(model: str, sets: dict, frames: int, crop_bottom: float = 0.0) -> dict:
    """crop_bottom: drop this fraction of each frame's bottom (removes generator watermarks such as Kling's
    bottom-right logo, to check the detector is not just reading the watermark)."""
    cache_file = _cache_path(model, crop_bottom)
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    clf = None
    for name, clips in sets.items():
        for source, path in clips:
            key = path.relative_to(ROOT).as_posix()
            if key in cache:
                continue
            clf = clf or A.frame_classifier(model)
            t0 = time.perf_counter()
            imgs = A.sample_frames(path, frames)
            if crop_bottom:
                imgs = [im.crop((0, 0, im.width, round(im.height * (1 - crop_bottom)))) for im in imgs]
            probs = [A.ai_probability(clf(img)) for img in imgs]
            cache[key] = {"set": name, "source": source, "per_frame": [round(p, 4) for p in probs],
                          "median": round(statistics.median(probs), 4) if probs else None,
                          "seconds": round(time.perf_counter() - t0, 2)}
            print(f"{model:45s} {name:8s} {key:70s} {cache[key]['median']}", flush=True)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cache, indent=1))
    return cache


def auc(neg: list[float], pos: list[float]) -> float:
    """P(score of a random AI clip > score of a random real clip), ties count half (Mann-Whitney)."""
    if not neg or not pos:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def rate(xs: list[float], t: float) -> float:
    return sum(x >= t for x in xs) / len(xs) if xs else float("nan")


def threshold_for_fpr(neg: list[float], max_fpr: float) -> float:
    """Lowest threshold whose false-positive rate on `neg` is <= max_fpr."""
    for t in sorted(set(neg)) + [1.01]:
        t2 = t + 1e-9
        if rate(neg, t2) <= max_fpr:
            return round(t2, 4)
    return 1.01


def cross_validated_tpr(neg: list[float], pos: list[float], max_fpr: float) -> tuple[float, float]:
    """2-fold: pick the threshold on one half (alternate clips), measure FPR/TPR on the other half."""
    fprs, tprs = [], []
    for k in (0, 1):
        t = threshold_for_fpr(neg[k::2], max_fpr)
        fprs.append(rate(neg[1 - k::2], t))
        tprs.append(rate(pos[1 - k::2], t))
    return statistics.mean(fprs), statistics.mean(tprs)


def _dist(xs: list[float]) -> str:
    if not xs:
        return "-"
    q = statistics.quantiles(xs, n=4, method="inclusive") if len(xs) > 1 else [xs[0]] * 3
    return f"n={len(xs)} min {min(xs):.2f} q1 {q[0]:.2f} med {q[1]:.2f} q3 {q[2]:.2f} max {max(xs):.2f}"


def report(model: str, cache: dict, cfg: A.AuthenticityCfg) -> dict:
    by = {"real": [], "ai": [], "faculty": []}
    by_source: dict[str, list[float]] = {}
    for v in cache.values():
        if v["median"] is None:
            continue
        by[v["set"]].append(v["median"])
        by_source.setdefault(f"{v['set']}/{v['source']}", []).append(v["median"])
    neg, pos = by["real"], by["ai"]
    r = {"model": model, "auc": round(auc(neg, pos), 3),
         "at_uncertain": {"t": cfg.uncertain_at, "fpr": rate(neg, cfg.uncertain_at), "tpr": rate(pos, cfg.uncertain_at)},
         "at_likely_ai": {"t": cfg.likely_ai_at, "fpr": rate(neg, cfg.likely_ai_at), "tpr": rate(pos, cfg.likely_ai_at)},
         "real_at_100pct": sum(x >= 0.995 for x in neg), "n_real": len(neg), "n_ai": len(pos)}
    for fpr in (0.05, 0.10):
        t = threshold_for_fpr(neg, fpr)
        cv_fpr, cv_tpr = cross_validated_tpr(neg, pos, fpr)
        r[f"fpr_le_{int(fpr * 100)}"] = {"t_in_sample": t, "tpr_in_sample": rate(pos, t), "cv_fpr": cv_fpr, "cv_tpr": cv_tpr}
    r["faculty"] = {k: v["median"] for k, v in cache.items() if v["set"] == "faculty"}
    print(f"\n=== {model}   AUC real-vs-AI {r['auc']}")
    for k, xs in sorted(by_source.items()):
        print(f"  {k:28s} {_dist(xs)}")
    for k in ("at_uncertain", "at_likely_ai"):
        print(f"  t={r[k]['t']:.2f}: FPR real {r[k]['fpr']:.0%}  TPR ai {r[k]['tpr']:.0%}")
    print(f"  real clips scored 100 %: {r['real_at_100pct']}/{len(neg)}")
    for fpr in (5, 10):
        x = r[f"fpr_le_{fpr}"]
        print(f"  FPR<={fpr}%: t={x['t_in_sample']:.3f} TPR {x['tpr_in_sample']:.0%} in-sample; "
              f"2-fold CV FPR {x['cv_fpr']:.0%} TPR {x['cv_tpr']:.0%}")
    print("  faculty (held out, not used for choices): "
          + ", ".join(f"{Path(k).stem} {v:.2f}" for k, v in sorted(r["faculty"].items())))
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--frames", type=int, default=A.AuthenticityCfg().frames)
    ap.add_argument("--report-only", action="store_true", help="use cached scores, run no model")
    ap.add_argument("--crop-bottom", type=float, default=0.0, help="drop this fraction of each frame's bottom")
    args = ap.parse_args()
    sets = clip_sets()
    print({k: len(v) for k, v in sets.items()})
    cfg = A.AuthenticityCfg()
    summary = []
    for m in args.models:
        cp = _cache_path(m, args.crop_bottom)
        cache = (json.loads(cp.read_text()) if cp.exists() else {}) if args.report_only else (
            score_model(m, sets, args.frames, args.crop_bottom))
        if cache:
            summary.append({**report(m, cache, cfg), "crop_bottom": args.crop_bottom})
            if not args.report_only:
                A._CLASSIFIERS.clear()  # free GPU memory before the next detector
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass
    name = f"summary_crop{int(args.crop_bottom * 100)}.json" if args.crop_bottom else "summary.json"
    (OUT / name).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
