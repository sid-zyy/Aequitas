"""Quantizes a Gemma 4 checkpoint with llm-compressor for the latency/accuracy study (faculty
requirement R5). The output is a compressed-tensors folder that vLLM loads with no --quantization flag.

    python scripts/quantize/quantize.py --scheme w4a16 --model google/gemma-4-E2B-it --out models/e2b-w4a16

Schemes:
  fp8    FP8_DYNAMIC: per-channel FP8 weights, dynamic per-token FP8 activations. No calibration.
         Needs compute capability >= 8.9 to run fast (Ada/Hopper).
  w8a8   INT8 weights and activations: SmoothQuant (moves activation outliers into weights) + GPTQ.
  w4a16  4-bit weights (GPTQ, group size 128), BF16 activations.

Always kept in BF16, following the published Gemma 4 recipes:
  - vision and audio towers and their projections: quantizing them hurts image understanding most
  - token embeddings and the per-layer embeddings (PLE): reported to break output when quantized
  - lm_head
On Gemma 4 E2B the PLE alone is ~2.8 B of the 5.1 B parameters, so 4-bit checkpoints stay ~7 GiB.

Calibration uses text only (the language backbone is what is quantized). By default it uses
Aequitas-style prompts plus a general chat set. Check the installed llm-compressor's Gemma 4 example
if an API name below has moved; this was written against the 2026 API (oneshot, QuantizationModifier,
GPTQModifier, SmoothQuantModifier)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

IGNORE = [
    "lm_head",
    "re:.*embed.*",  # embed_tokens, embed_tokens_per_layer (PLE), embed_vision, embed_audio
    "re:.*per_layer_model_projection.*",
    "re:.*vision_tower.*",
    "re:.*audio_tower.*",
    "re:.*multi_modal_projector.*",
    "re:.*router",
]

AEQUITAS_CALIB = [
    "A security camera clip was described as follows. Description: Two people walk through the door one "
    "right behind the other. Choose the ONE category that best matches the description.",
    "Describe what happens in the scene: a man walks across the hall, stops near a bench and sits down.",
    "Answer the user's question using ONLY the events listed below. [#12] 14:02 camera Gate, fighting: "
    "two men push each other. Question: were there any fights today?",
]


def recipe(scheme: str):
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier

    if scheme == "fp8":
        return QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=IGNORE)
    if scheme == "w4a16":
        return GPTQModifier(targets="Linear", scheme="W4A16", ignore=IGNORE)
    if scheme == "w8a8":
        try:
            from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
        except ImportError:  # moved in newer releases
            from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
        return [SmoothQuantModifier(smoothing_strength=0.8),
                GPTQModifier(targets="Linear", scheme="W8A8", ignore=IGNORE)]
    raise ValueError(scheme)


def calibration(processor, n: int, max_len: int, dataset: str):
    from datasets import Dataset, load_dataset

    texts = list(AEQUITAS_CALIB)
    if dataset != "none" and n > len(texts):
        ds = load_dataset(dataset, split=f"train_sft[:{n - len(texts)}]")
        for row in ds:
            texts.append(processor.apply_chat_template(row["messages"], tokenize=False))
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    enc = [tok(t, max_length=max_len, truncation=True, add_special_tokens=False) for t in texts[:n]]
    return Dataset.from_list([{"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]} for e in enc])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, choices=["fp8", "w8a8", "w4a16"])
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=256, help="calibration samples (w8a8/w4a16)")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k", help="'none' = Aequitas prompts only")
    args = ap.parse_args()

    from llmcompressor import oneshot
    from transformers import AutoModelForImageTextToText, AutoProcessor

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype="auto")
    kwargs = {}
    if args.scheme != "fp8":
        kwargs = {"dataset": calibration(processor, args.samples, args.max_len, args.dataset),
                  "max_seq_length": args.max_len, "num_calibration_samples": args.samples}
    oneshot(model=model, recipe=recipe(args.scheme), **kwargs)

    out = Path(args.out)
    model.save_pretrained(out, save_compressed=True)
    processor.save_pretrained(out)
    size_gb = sum(f.stat().st_size for f in out.glob("*.safetensors")) / 2**30
    meta = {"base": args.model, "scheme": args.scheme, "ignore": IGNORE, "samples": kwargs.get("num_calibration_samples"),
            "dataset": args.dataset if args.scheme != "fp8" else None, "size_gib": round(size_gb, 2),
            "minutes": round((time.time() - t0) / 60, 1)}
    (out / "aequitas_quant.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
