"""AI-likelihood for citizen videos: flags obviously AI-generated ones with a percentage and the reasons.

This is evidence for a human, not proof. No open detector reliably recognises every video generator,
and real phone videos can be flagged. Signals, strongest first:

1. provenance: a C2PA manifest (Sora, Veo, Adobe... embed one) that declares AI generation -> ~99 %
2. generator fingerprints in the file's metadata (tool names written by generators/editors) -> ~90 %
3. camera metadata (phone make/model, GPS) -> evidence of a real capture -> ~10 %
4. a frame classifier (an image AI-detector run on sampled frames) -> a check, NOT a percentage

Only (1)-(3) give a percentage. Re-uploads through messaging apps strip metadata, so they are often absent; then
only the frame check speaks, and it is reported as words ("real", or "AI-like frames" above a measured
threshold): image detectors are trained on stills, and their scores on compressed video are uncalibrated (they give
real stock video 100 %). Measured in results/authenticity/README.md (scripts/eval_authenticity.py)."""

from __future__ import annotations

import re
import statistics
import time
from pathlib import Path

from pydantic import BaseModel

VERSION = "2"
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
    # Image AI-detector from Hugging Face run on sampled frames (transformers image-classification). None =
    # metadata checks only. Default: haywoodsloan/ai-image-detector-deploy (SwinV2, Apache-2.0, 781 MB). Measured on
    # 80 real and 50 AI clips (results/authenticity/README.md): 1/80 real clips >= 0.85 (Organika/sdxl-detector,
    # the old default: 30/80), 34/40 Kling clips caught, 0/10 open-model clips (CogVideoX, Hunyuan, Mochi...).
    frame_model: str | None = "haywoodsloan/ai-image-detector-deploy"
    frames: int = 8
    # The GPU is shared with the VLM (6 GB laptop): use it only when this much VRAM is free (peak ~0.85 GB for
    # the default model; 8 frames take ~0.3 s on the GPU, ~3 s on the CPU).
    gpu_min_free_gb: float = 1.5
    # Verdict thresholds on the final likelihood (C2PA / metadata evidence).
    likely_ai_at: float = 0.85
    uncertain_at: float = 0.5
    # The frame detector alone is never shown as a percentage: image detectors' scores are uncalibrated on video
    # (Organika gave 22 of 80 real stock / CCTV clips >= 99.5 %). A frame-only result says "AI-like frames"
    # (verdict uncertain) when the median frame score reaches frame_flag_at, else "real" (a low score
    # misses most open-model generators). None = always real. Measured on non-faculty clips only.
    frame_flag_at: float | None = 0.85
    frame_flag_note: str = "in our test this flagged 1 of 80 real clips and 34 of 50 AI clips"

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


def _device(gpu_min_free_gb: float) -> int:
    """0 = first GPU (~10x faster) when it has the free memory, else -1 = CPU (the VLM keeps the GPU)."""
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] >= gpu_min_free_gb * 1024**3:
            return 0
    except Exception:  # torch missing, or the driver cannot report memory
        pass
    return -1


def frame_classifier(model_id: str, gpu_min_free_gb: float = 1.5):
    if model_id not in _CLASSIFIERS:
        from transformers import pipeline

        # float32: transformers 5 loads the checkpoint's own half precision by default, which fails on the CPU
        # ("Float and Half") whenever the GPU is busy with the VLM and the detector falls back
        _CLASSIFIERS[model_id] = pipeline("image-classification", model=model_id, device=_device(gpu_min_free_gb),
                                          dtype="float32")
    return _CLASSIFIERS[model_id]


def ai_probability(predictions: list[dict]) -> float:
    """P(AI) from one image's predictions, whatever the model calls its classes."""
    ai = [p["score"] for p in predictions if _AI_LABEL.search(p["label"]) and "real" not in p["label"].lower()]
    if ai:
        return float(max(ai))
    real = [p["score"] for p in predictions if "real" in p["label"].lower() or "human" in p["label"].lower()]
    return float(1 - max(real)) if real else 0.5


def combine(meta: dict, c2pa: dict | None, frame_p: float | None,
            cfg: AuthenticityCfg) -> tuple[float | None, list[str], str]:
    """(likelihood, reasons, basis). basis says which evidence decided: provenance, metadata, camera, frames or
    none. With basis "frames" the likelihood is None: the frame score is reported as a check, not a percentage."""
    reasons = []
    if (c2pa and c2pa.get("ai_declared")) or meta["ai_source_type"]:
        reasons.append("Content credentials (C2PA) declare the video AI-generated"
                       + (f" by {c2pa['generator']}" if c2pa and c2pa.get("generator") else ""))
        return 0.99, reasons, "provenance"
    if meta["generators"]:
        reasons.append(f"File metadata names an AI video tool: {', '.join(meta['generators'])}")
        return 0.9, reasons, "metadata"
    if meta["camera_metadata"]:
        reasons.append("Phone camera metadata present (make/model or location): points to a real capture")
        if frame_p is not None:
            reasons.append("The frame detector also ran; its score is not used (uncalibrated on video)")
        return 0.1, reasons, "camera"
    if frame_p is None:
        reasons.append("No metadata and no frame check: not enough signal")
        return None, reasons, "none"
    if cfg.frame_flag_at is not None and frame_p >= cfg.frame_flag_at:
        reasons.append("Frame check: AI-like frames, needs a human look"
                       + (f" ({cfg.frame_flag_note})" if cfg.frame_flag_note else ""))
    else:
        reasons.append("Frame check: no AI signs found, treated as real")  # prefer a missed fake to a real video flagged
    # the raw frame score stays in the result's "frame_score", not in the text the officer reads
    reasons.append("Only an image AI-detector ran, on sampled frames. On real stock and CCTV video it scores as "
                   "high as on AI video, so its score is not shown as a percentage")
    reasons.append("No camera metadata (common for forwarded videos, also for AI ones)")
    return None, reasons, "frames"


def analyze_video(path: str | Path, cfg: AuthenticityCfg | None = None) -> dict:
    cfg = cfg or AuthenticityCfg()
    path = Path(path)
    t0 = time.perf_counter()
    meta = metadata_signals(_read_edges(path))
    c2 = c2pa_signal(path)
    frame_p, frame_detail = None, None
    if cfg.frame_model and not meta["ai_source_type"]:
        try:
            clf = frame_classifier(cfg.frame_model, cfg.gpu_min_free_gb)
            probs = [ai_probability(clf(img)) for img in sample_frames(path, cfg.frames)]
            if probs:
                frame_p = round(statistics.median(probs), 4)  # robust to a few wild frames
                frame_detail = {"model": cfg.frame_model, "per_frame": [round(x, 3) for x in probs]}
        except Exception as e:  # transformers missing, download failed, unreadable video...
            frame_detail = {"model": cfg.frame_model, "error": str(e)[:200]}
    likelihood, reasons, basis = combine(meta, c2, frame_p, cfg)
    if basis == "frames":
        flagged = cfg.frame_flag_at is not None and frame_p >= cfg.frame_flag_at
        verdict = "uncertain" if flagged else "no_ai_signs"
    elif likelihood is None:
        verdict = "not_enough_signal"
    elif likelihood >= cfg.likely_ai_at:
        verdict = "likely_ai"
    elif likelihood >= cfg.uncertain_at:
        verdict = "uncertain"
    else:
        verdict = "no_ai_signs"
    return {
        "ai_likelihood": likelihood,
        "ai_likelihood_pct": round(likelihood * 100) if likelihood is not None else None,
        "verdict": verdict, "basis": basis, "headline": reasons[0], "frame_score": frame_p, "reasons": reasons,
        "metadata": meta, "c2pa": c2, "frames": frame_detail,
        "disclaimer": DISCLAIMER, "version": VERSION, "seconds": round(time.perf_counter() - t0, 2),
    }
