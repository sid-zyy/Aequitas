"""AI-likelihood for citizen videos: flags obviously AI-generated ones with a percentage and the reasons.

This is evidence for a human, not proof. No open detector reliably recognises every video generator,
and real phone videos can be flagged. Signals, strongest first:

1. provenance: a C2PA manifest (Sora, Veo, Adobe... embed one) that declares AI generation -> ~99 %
2. generator fingerprints in the file's metadata (tool names written by generators/editors) -> ~90 %
3. a frame classifier (an image AI-detector run on sampled frames) -> its mean probability
4. camera metadata (phone make/model, GPS) -> evidence of a real capture, lowers the score

Re-uploads through messaging apps strip metadata, so (1), (2) and (4) are often absent; then the
frame classifier decides. With no classifier and no metadata, the result says "not enough signal"."""

from __future__ import annotations

import re
import statistics
import time
from pathlib import Path

from pydantic import BaseModel

VERSION = "1"
DISCLAIMER = ("AI-likelihood is an automated estimate to help prioritise review, not proof. Real videos can be "
              "flagged, and new AI generators can be missed. Verify with the citizen and the original file.")

# Tool names written into files by AI video generators or AI-heavy editors. Only distinctive names
# (short ones like "Veo" alone would match random compressed bytes), searched only inside readable
# text runs of the file, as metadata is.
GENERATOR_MARKERS = [
    "OpenAI Sora", "Sora 2", "Google Veo", "Veo 2", "Veo 3", "RunwayML", "Runway Gen", "Kling AI", "Hailuo AI",
    "MiniMax", "Dream Machine", "Synthesia", "HeyGen", "Midjourney", "Stable Video Diffusion", "AnimateDiff",
    "ComfyUI", "HunyuanVideo", "Wan2.1", "Wan2.2", "CogVideoX", "PixVerse", "Pika Labs", "Haiper", "Adobe Firefly",
]
AI_SOURCE_TYPES = ["trainedAlgorithmicMedia", "compositeWithTrainedAlgorithmicMedia", "algorithmicMedia"]
CAMERA_MARKERS = [b"com.apple.quicktime.make", b"com.apple.quicktime.model", b"com.android.version",
                  b"com.android.manufacturer", b"\xa9mak", b"\xa9mod", b"\xa9xyz", b"com.apple.quicktime.location"]
_AI_LABEL = re.compile(r"(fake|artificial|ai|generated|synthetic|sdxl)", re.I)


class AuthenticityCfg(BaseModel):
    # Image AI-detector from Hugging Face run on sampled frames (transformers image-classification).
    # Default: Organika/sdxl-detector (CC-BY-NC: fine for a demo, replace before any commercial use).
    # None = metadata checks only.
    frame_model: str | None = "Organika/sdxl-detector"
    frames: int = 8
    # Verdict thresholds on the final likelihood. Strict on purpose: the frame detector gave 64 % (per frame
    # 0.33-0.97) on a re-encoded video of a real photo, so only high scores are called likely AI.
    likely_ai_at: float = 0.85
    uncertain_at: float = 0.5
    # The frame detector alone never says "likely AI": on 22 real stock / CCTV clips (data/footage, 2026-10-16)
    # it gave 5 of them 100 % (accident, fire, punching, flyover, an arrest). Image detectors are trained on
    # stills; compressed video frames look "synthetic" to them. Metadata / C2PA evidence can still say likely AI.
    frame_alone_max_verdict: str = "uncertain"


def _read_edges(path: Path, n: int = 4 * 1024**2) -> bytes:
    """Metadata lives in the moov/udta boxes, at the start or the end of an MP4."""
    size = path.stat().st_size
    with path.open("rb") as f:
        head = f.read(n)
        if size > 2 * n:
            f.seek(size - n)
            return head + f.read(n)
        return head + f.read()


def metadata_signals(data: bytes) -> dict:
    text = "\n".join(s.decode("ascii") for s in re.findall(rb"[\x20-\x7e]{8,}", data))  # like `strings`
    generators = sorted({m for m in GENERATOR_MARKERS
                         if re.search(rf"(?<![A-Za-z0-9]){re.escape(m)}(?![A-Za-z0-9])", text, re.I)})
    ai_source = sorted({t for t in AI_SOURCE_TYPES if t in text})
    camera = sorted({m.decode("latin-1").replace("\xa9", "(c)") for m in CAMERA_MARKERS if m in data})
    return {"c2pa_present": b"c2pa" in data or b"jumb" in data, "ai_source_type": ai_source,
            "generators": generators, "camera_metadata": camera}


def c2pa_signal(path: Path) -> dict | None:
    """Reads a C2PA manifest with the c2pa-python library, if installed. None = library not available."""
    try:
        import c2pa  # type: ignore
    except ImportError:
        return None
    try:
        with c2pa.Reader(str(path)) as reader:
            manifest = reader.json()
    except Exception as e:  # no manifest, or unsupported/invalid
        return {"manifest": False, "detail": str(e)[:120]}
    return {"manifest": True, "ai_declared": any(t in manifest for t in AI_SOURCE_TYPES),
            "generator": (re.findall(r'"claim_generator"\s*:\s*"([^"]+)"', manifest) or [None])[0]}


def sample_frames(path: Path, n: int):
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frames = []
    for k in range(n):
        if total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((k + 0.5) * total / n))
        ok, img = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


_CLASSIFIERS: dict[str, object] = {}


def frame_classifier(model_id: str):
    if model_id not in _CLASSIFIERS:
        from transformers import pipeline

        try:
            import torch

            device = 0 if torch.cuda.is_available() else -1  # GPU: ~10x faster than the CPU default
        except ImportError:
            device = -1
        _CLASSIFIERS[model_id] = pipeline("image-classification", model=model_id, device=device)
    return _CLASSIFIERS[model_id]


def ai_probability(predictions: list[dict]) -> float:
    """P(AI) from one image's predictions, whatever the model calls its classes."""
    ai = [p["score"] for p in predictions if _AI_LABEL.search(p["label"]) and "real" not in p["label"].lower()]
    if ai:
        return float(max(ai))
    real = [p["score"] for p in predictions if "real" in p["label"].lower() or "human" in p["label"].lower()]
    return float(1 - max(real)) if real else 0.5


def combine(meta: dict, c2pa: dict | None, frame_p: float | None, cfg: AuthenticityCfg) -> tuple[float | None, list[str]]:
    reasons = []
    if (c2pa and c2pa.get("ai_declared")) or meta["ai_source_type"]:
        reasons.append("Content credentials (C2PA) declare the video AI-generated"
                       + (f" by {c2pa['generator']}" if c2pa and c2pa.get("generator") else ""))
        return 0.99, reasons
    if meta["generators"]:
        reasons.append(f"File metadata names an AI video tool: {', '.join(meta['generators'])}")
        return max(0.9, frame_p or 0), reasons
    if frame_p is None:
        if meta["camera_metadata"]:
            reasons.append("Phone camera metadata present (make/model or location); no frame check ran")
            return 0.1, reasons
        reasons.append("No metadata and no frame check: not enough signal")
        return None, reasons
    p = frame_p
    reasons.append(f"Frame detector: {p * 100:.0f} % median AI probability over sampled frames")
    if meta["camera_metadata"]:
        p *= 0.6
        reasons.append("Phone camera metadata present: lowers the likelihood")
    else:
        reasons.append("No camera metadata (common for forwarded videos, also for AI ones)")
    return round(min(1.0, p), 4), reasons


def analyze_video(path: str | Path, cfg: AuthenticityCfg | None = None) -> dict:
    cfg = cfg or AuthenticityCfg()
    path = Path(path)
    t0 = time.perf_counter()
    meta = metadata_signals(_read_edges(path))
    c2 = c2pa_signal(path)
    frame_p, frame_detail = None, None
    if cfg.frame_model and not meta["ai_source_type"]:
        try:
            clf = frame_classifier(cfg.frame_model)
            probs = [ai_probability(clf(img)) for img in sample_frames(path, cfg.frames)]
            if probs:
                frame_p = round(statistics.median(probs), 4)  # robust to a few wild frames
                frame_detail = {"model": cfg.frame_model, "per_frame": [round(x, 3) for x in probs]}
        except Exception as e:  # transformers missing, download failed, unreadable video...
            frame_detail = {"model": cfg.frame_model, "error": str(e)[:200]}
    likelihood, reasons = combine(meta, c2, frame_p, cfg)
    frame_only = frame_p is not None and not (meta["ai_source_type"] or meta["generators"]
                                              or (c2 and c2.get("ai_declared")))
    if likelihood is None:
        verdict = "not_enough_signal"
    elif likelihood >= cfg.likely_ai_at and frame_only and cfg.frame_alone_max_verdict == "uncertain":
        verdict = "uncertain"
        reasons.append("Only the frame detector points to AI, and it also flags real compressed video: "
                       "needs a human look")
    elif likelihood >= cfg.likely_ai_at:
        verdict = "likely_ai"
    elif likelihood >= cfg.uncertain_at:
        verdict = "uncertain"
    else:
        verdict = "no_ai_signs"
    return {
        "ai_likelihood": likelihood,
        "ai_likelihood_pct": round(likelihood * 100) if likelihood is not None else None,
        "verdict": verdict, "reasons": reasons, "metadata": meta, "c2pa": c2, "frames": frame_detail,
        "disclaimer": DISCLAIMER, "version": VERSION, "seconds": round(time.perf_counter() - t0, 2),
    }
