"""Zero-shot scene gate (coarse stage before the VLM).

Idea from Flashback (2025) and VadCLIP (AAAI 2024): an image-text model (SigLIP) compares each frame with
short descriptions of normal scenes and of incidents. A window that clearly looks normal skips the VLM, which
then only explains the candidates. On our 72 saved demo windows (scripts/bench_variants.py set) it took 17 ms
per window and, at 0.3, skipped 28 of 33 normal windows and none of the 39 incident windows. That threshold was
chosen on those windows, so treat it as a demo setting; the VLM still looks at least every analysis.max_skip_s.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

NORMAL = ["normal street traffic with cars and motorbikes", "people walking on a busy street",
          "people standing and talking", "an empty road", "people sitting in a room", "a shop front with customers",
          "a crowded market street", "a train station platform with people"]
INCIDENT = ["two people fighting", "a man punching another man", "police officers arresting a man",
            "police handcuffing a person", "a car on fire", "flames and thick smoke", "a person lying on the ground",
            "a car crash", "a person holding a gun", "a person holding a knife", "a crowd running away in panic",
            "people throwing stones", "a man attacking a woman"]


def should_skip(p_incident: float | None, violence: float | None, since_vlm_ms: float | None, *,
                below: float, violence_band_low: float, max_skip_ms: float) -> bool:
    """Skip the VLM only when every signal agrees the window is ordinary."""
    if p_incident is None or since_vlm_ms is None:  # no score, or the camera's first window
        return False
    if violence is not None and violence >= violence_band_low:  # the clip model sees something
        return False
    if since_vlm_ms >= max_skip_ms:  # the VLM looks at least this often anyway
        return False
    return p_incident < below


class SceneGate:
    def __init__(self, model: str, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = "cuda" if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.proc = AutoProcessor.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model, dtype=dtype).to(self.device).eval()
        with torch.no_grad():
            t = self.proc(text=NORMAL + INCIDENT, padding="max_length", return_tensors="pt").to(self.device)
            emb = self.model.get_text_features(**t)
            emb = getattr(emb, "pooler_output", emb)
            self.text = emb / emb.norm(dim=-1, keepdim=True)
        log.info("scene gate: %s on %s", model, self.device)

    def score(self, images_bgr: list) -> float | None:
        """Highest incident probability over the frames (BGR numpy images). None if no frames."""
        if not images_bgr:
            return None
        from PIL import Image

        torch = self.torch
        pil = [Image.fromarray(im[:, :, ::-1]) for im in images_bgr]
        with torch.no_grad():
            px = self.proc(images=pil, return_tensors="pt")["pixel_values"].to(self.device, self.model.dtype)
            emb = self.model.get_image_features(pixel_values=px)
            emb = getattr(emb, "pooler_output", emb)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            logits = (emb @ self.text.T).float() * self.model.logit_scale.exp().float() + self.model.logit_bias.float()
            p = logits.softmax(-1)[:, len(NORMAL):].sum(-1)
        return float(p.max())
