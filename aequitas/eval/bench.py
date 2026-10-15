"""`aequitas bench`: latency of the three faculty tasks against any OpenAI-compatible server (vLLM, or
Ollama's /v1 endpoint), at several concurrency levels.

Per request it measures TTFT, time per output token (TPOT), output tokens/s and total time, by
streaming every response. Per level it reports p50/p95 and wall time for the whole batch, and, for
vLLM, the speculative-decoding acceptance rate read from /metrics before and after the level.

Every summarize request gets a different image (Ollama caches identical prompts + images, which made
calls look ~10x faster than reality in M0)."""

from __future__ import annotations

import base64
import json
import logging
import random
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from aequitas.eval.report import percentile
from aequitas.understand.prompts import CHOICES, SummaryAnswer, classify_text_prompt, summarize_prompt
from aequitas.understand.vlm_client import is_local_host

log = logging.getLogger(__name__)

TASKS = ("summarize", "classify", "chat")

_SUMMARIES = [
    "Two people walk through the doorway one right behind the other.",
    "A man falls near the bench and lies on the ground without moving.",
    "Two men push and hit each other in the middle of the corridor.",
    "A crowd forms near the gate and then suddenly scatters.",
    "A person leaves a backpack next to the pillar and walks away.",
    "People walk across the hall; nothing unusual happens.",
]


@dataclass
class Sample:
    task: str
    ok: bool
    total_ms: float
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    out_tokens: int | None = None
    error: str | None = None


def synthetic_grid(i: int, size: int = 896, seed: int = 0) -> bytes:
    """A different 2x2 'scene' per call: random people-like boxes on a noisy floor."""
    rng = np.random.default_rng(seed * 100_003 + i)
    img = rng.integers(60, 120, (size, size, 3), dtype=np.uint8)
    half = size // 2
    for q in range(4):
        ox, oy = (q % 2) * half, (q // 2) * half
        for _ in range(rng.integers(1, 5)):
            x, y = rng.integers(ox + 20, ox + half - 80), rng.integers(oy + 20, oy + half - 160)
            color = tuple(int(c) for c in rng.integers(0, 255, 3))
            cv2.rectangle(img, (int(x), int(y)), (int(x) + 50, int(y) + 140), color, -1)
        cv2.putText(img, f"t-{3 - q}s", (ox + 10, oy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def _chat_prompt(i: int) -> str:
    facts = "\n".join(f"[#{100 + k}] 2026-10-15 14:{10 + k}:00, camera Gate, {s}"
                      for k, s in enumerate(_SUMMARIES))
    q = ["What happened at the gate this afternoon?", "Were there any falls?", "Summarise the fights."][i % 3]
    return ("Answer the user's question using ONLY the events listed below. Refer to events by their "
            f"reference like [#12]. Write 1-3 sentences.\n\nEvents:\n{facts}\n\nQuestion: {q} (request {i})")


def build_request(task: str, model: str, i: int, seed: int, max_tokens: int) -> dict:
    if task == "summarize":
        b64 = base64.b64encode(synthetic_grid(i, seed=seed)).decode("ascii")
        content = [{"type": "text", "text": summarize_prompt(highlighted=False)},
                   {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]
        extra = {"response_format": {"type": "json_schema", "json_schema": {
            "name": "SummaryAnswer", "schema": SummaryAnswer.model_json_schema()}}}
    elif task == "classify":
        content = classify_text_prompt(f"{_SUMMARIES[i % len(_SUMMARIES)]} (clip {seed}-{i})")
        extra = {"structured_outputs": {"choice": CHOICES}}
    elif task == "chat":
        content = _chat_prompt(i)
        extra = {}
    else:
        raise ValueError(task)
    return {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0,
            "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True}, **extra}


def run_one(http, base: str, body: dict, task: str) -> Sample:
    """Streams one request; TTFT = first content token, TPOT = (last - first) / (tokens - 1)."""
    t0 = time.perf_counter()
    first = last = None
    pieces = 0
    usage_tokens = None
    try:
        with http.stream("POST", f"{base}/chat/completions", json=body) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage_tokens = chunk["usage"].get("completion_tokens")
                choices = chunk.get("choices") or []
                if choices and (choices[0].get("delta") or {}).get("content"):
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    pieces += 1
    except Exception as e:
        return Sample(task, False, (time.perf_counter() - t0) * 1000, error=str(e)[:200])
    total = (time.perf_counter() - t0) * 1000
    n = usage_tokens or pieces
    ttft = (first - t0) * 1000 if first else None
    tpot = ((last - first) * 1000 / (n - 1)) if first and n and n > 1 else None
    return Sample(task, first is not None, total, ttft, tpot, n)


# ---------- vLLM /metrics ----------

_SPEC_METRICS = {
    "drafts": "vllm:spec_decode_num_drafts",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens",
}


def parse_spec_metrics(text: str) -> dict[str, float]:
    """Sums the speculative-decoding counters over all label sets. Missing = not a vLLM server, or
    speculation is off."""
    out: dict[str, float] = {}
    for key, name in _SPEC_METRICS.items():
        pat = re.compile(rf"^{re.escape(name)}(?:_total)?(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)$", re.M)
        vals = [float(v) for v in pat.findall(text)]
        if vals:
            out[key] = sum(vals)
    return out


def acceptance(before: dict, after: dict) -> dict | None:
    d = {k: after.get(k, 0) - before.get(k, 0) for k in _SPEC_METRICS}
    if not d["draft_tokens"]:
        return None
    return {"acceptance_rate": round(d["accepted_tokens"] / d["draft_tokens"], 4),
            "mean_accepted_per_draft": round(d["accepted_tokens"] / d["drafts"], 3) if d["drafts"] else None,
            **{k: int(v) for k, v in d.items()}}


def _metrics(http, root: str) -> dict:
    try:
        r = http.get(f"{root}/metrics", timeout=5)
        return parse_spec_metrics(r.text) if r.status_code == 200 else {}
    except Exception:
        return {}


# ---------- the run ----------

def _stats(vals: list[float]) -> dict:
    if not vals:
        return {}
    return {"p50": round(percentile(vals, .5), 1), "p95": round(percentile(vals, .95), 1),
            "mean": round(statistics.mean(vals), 1)}


def summarize_level(samples: list[Sample], wall_ms: float) -> dict:
    ok = [s for s in samples if s.ok]
    toks = sum(s.out_tokens or 0 for s in ok)
    return {
        "requests": len(samples), "ok": len(ok), "wall_ms": round(wall_ms, 1),
        "requests_per_s": round(len(ok) / (wall_ms / 1000), 3) if wall_ms else None,
        "output_tokens_per_s": round(toks / (wall_ms / 1000), 1) if wall_ms else None,
        "total_ms": _stats([s.total_ms for s in ok]),
        "ttft_ms": _stats([s.ttft_ms for s in ok if s.ttft_ms is not None]),
        "tpot_ms": _stats([s.tpot_ms for s in ok if s.tpot_ms is not None]),
        "errors": sorted({s.error for s in samples if s.error})[:3],
    }


def run_bench(host: str, model: str, tasks: list[str], levels: list[int], requests: int,
              label: str = "", out_dir: Path = Path("data/reports"), allow_remote: bool = False,
              max_tokens: dict | None = None) -> dict:
    import httpx

    if not allow_remote and not is_local_host(host):
        raise ValueError(f"{host} is not local; Aequitas only benchmarks local servers")
    root = host.rstrip("/").removesuffix("/v1")
    base = root + "/v1"
    max_tokens = {"summarize": 160, "classify": 16, "chat": 160, **(max_tokens or {})}
    seed = random.randrange(1_000_000)  # fresh images every run (cache trap)
    http = httpx.Client(timeout=600)
    report = {"label": label, "host": host, "model": model, "started": datetime.now().isoformat(timespec="seconds"),
              "requests_per_level": requests, "max_tokens": max_tokens, "results": {}}

    # warm-up: model load, CUDA graphs, prefix cache for the shared prompt text
    for task in tasks:
        run_one(http, base, build_request(task, model, -1, seed, max_tokens[task]), task)

    next_id = 0  # request index: every request gets its own image and text
    for task in tasks:
        report["results"][task] = {}
        for c in levels:
            before = _metrics(http, root)
            ids = range(next_id, next_id + requests)
            next_id += requests
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=c) as pool:
                samples = list(pool.map(
                    lambda i: run_one(http, base, build_request(task, model, i, seed, max_tokens[task]), task), ids))
            wall = (time.perf_counter() - t0) * 1000
            level = summarize_level(samples, wall)
            level["speculation"] = acceptance(before, _metrics(http, root))
            report["results"][task][str(c)] = level
            log.info("%s c=%d: %d/%d ok, wall %.1f s, TTFT p50 %s ms, TPOT p50 %s ms", task, c, level["ok"],
                     requests, wall / 1000, level["ttft_ms"].get("p50"), level["tpot_ms"].get("p50"))

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"bench-{re.sub(r'[^A-Za-z0-9_.-]+', '_', label) + '-' if label else ''}{datetime.now():%Y%m%d-%H%M%S}"
    (out_dir / f"{stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / f"{stem}.md").write_text(markdown(report), encoding="utf-8")
    report["_path"] = str(out_dir / f"{stem}.md")
    return report


def markdown(report: dict) -> str:
    lines = [f"# Bench {report['label'] or ''}".rstrip(), "",
             f"Model `{report['model']}` at `{report['host']}`, {report['started']}, "
             f"{report['requests_per_level']} requests per level.", "",
             "| task | conc. | ok | wall s | req/s | out tok/s | total p50 ms | TTFT p50 ms | TPOT p50 ms | spec. accept |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for task, levels in report["results"].items():
        for c, r in levels.items():
            spec = r.get("speculation")
            lines.append(
                f"| {task} | {c} | {r['ok']}/{r['requests']} | {r['wall_ms'] / 1000:.1f} | {r['requests_per_s']} | "
                f"{r['output_tokens_per_s']} | {r['total_ms'].get('p50', '-')} | {r['ttft_ms'].get('p50', '-')} | "
                f"{r['tpot_ms'].get('p50', '-')} | {spec['acceptance_rate'] if spec else '-'} |")
    return "\n".join(lines) + "\n"

