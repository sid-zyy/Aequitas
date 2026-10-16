"""Model-independent LLM/VLM clients. The same interface serves the VLM check and the chatbot, and
either can point at Ollama or at any local OpenAI-compatible server (llama.cpp, vLLM, LM Studio).

Messages use one neutral shape: {"role": ..., "content": str, "images": [jpeg bytes]}."""

from __future__ import annotations

import base64
import ipaddress
import json
import logging
import math
import os
import time
from collections.abc import Iterator
from urllib.parse import urlparse

from pydantic import BaseModel, ValidationError

from aequitas.config import LlmEndpoint
from aequitas.understand.prompts import (
    CHOICES,
    SummaryAnswer,
    VlmAnswer,
    classify_prompt,
    classify_text_prompt,
    label_model,
    summarize_prompt,
)
from aequitas.taxonomy import choices as label_choices

log = logging.getLogger(__name__)


def is_local_host(url: str) -> bool:
    """Loopback or private-network addresses only: a GPU box on the LAN is fine, a cloud API is not."""
    host = urlparse(url if "://" in url else f"http://{url}").hostname or ""
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private


class LlmClient:
    def __init__(self, cfg: LlmEndpoint):
        if not cfg.allow_remote and not is_local_host(cfg.host):
            raise ValueError(
                f"{cfg.host} is not a local address. Aequitas only uses local inference; "
                "set allow_remote: true if this host is your own server."
            )
        self.cfg = cfg
        self.model = cfg.model

    def _chat(self, messages: list[dict], schema: dict, name: str,
              max_tokens: int | None = None) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def ping(self) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def structured(self, messages: list[dict], model_cls: type[BaseModel], retries: int = 1,
                   max_tokens: int | None = None) -> BaseModel | None:
        schema = model_cls.model_json_schema()
        for attempt in range(retries + 1):
            try:
                raw = self._chat(messages, schema, model_cls.__name__, max_tokens or self.cfg.max_tokens)
                return model_cls.model_validate_json(raw)
            except ValidationError as e:
                log.warning("invalid structured output (attempt %d): %s", attempt + 1, e.errors()[:2])
            except Exception as e:  # connection errors, timeouts
                log.warning("%s call failed (attempt %d): %s", self.cfg.backend, attempt + 1, e)
        return None

    def classify(self, grid_jpeg: bytes, highlighted: bool = True) -> tuple[VlmAnswer | None, float]:
        """Legacy single call: category and summary from the VLM at once (used by `aequitas spike`)."""
        t0 = time.perf_counter()
        ans = self.structured(
            [{"role": "user", "content": classify_prompt(highlighted), "images": [grid_jpeg]}], VlmAnswer
        )
        return ans, (time.perf_counter() - t0) * 1000

    # ---------- the two pipeline stages ----------

    def summarize(self, grid_jpeg: bytes, highlighted: bool = True,
                  view: str = "cctv") -> tuple[SummaryAnswer | None, float]:
        """Stage 1: the VLM describes the key-frame grid."""
        t0 = time.perf_counter()
        ans = self.structured(
            [{"role": "user", "content": summarize_prompt(highlighted, self.cfg.concise_summary, view,
                                                           self.cfg.guided_checks),
              "images": [grid_jpeg]}],
            SummaryAnswer,
            max_tokens=self.cfg.summary_max_tokens + (48 if self.cfg.guided_checks else 0),
        )
        return ans, (time.perf_counter() - t0) * 1000

    def _choose(self, prompt: str, choices: list[str]) -> tuple[str, float | None]:
        """One label from `choices` and its probability if the backend reports it. The default uses
        a JSON-schema enum, which every backend supports but which gives no confidence."""
        ans = self.structured([{"role": "user", "content": prompt}], label_model(tuple(choices)))
        if ans is None:
            raise ValueError("no valid label")
        return ans.category, None

    def classify_text(self, summary: str, evidence: str = "", categories: dict[str, str] | None = None,
                      previous: str | None = None) -> tuple[str | None, float | None, float]:
        """Stage 2: label the summary text against `categories` (default: the police list).
        Returns (label, confidence, ms); label None on failure."""
        t0 = time.perf_counter()
        choices = label_choices(categories) if categories else CHOICES
        label, conf = None, None
        try:
            label, conf = self._choose(classify_text_prompt(summary, evidence, categories, previous), choices)
        except Exception as e:
            log.warning("%s classify call failed: %s", self.cfg.backend, e)
        if label not in choices:
            label, conf = None, None
        return label, conf, (time.perf_counter() - t0) * 1000


class OllamaClient(LlmClient):
    def __init__(self, cfg: LlmEndpoint):
        super().__init__(cfg)
        import ollama

        self.client = ollama.Client(host=cfg.host, timeout=cfg.timeout_s)
        self.last_stats: dict = {}  # token counts and durations of the last call (benchmarks read these)

    def _options(self, num_predict: int) -> dict:
        opts = {"temperature": 0, "num_ctx": self.cfg.num_ctx, "num_predict": num_predict}
        if self.cfg.draft_num_predict is not None:
            opts["draft_num_predict"] = self.cfg.draft_num_predict
        if self.cfg.repeat_penalty is not None:
            opts["repeat_penalty"] = self.cfg.repeat_penalty
        return opts

    def _chat(self, messages: list[dict], schema: dict, name: str, max_tokens: int | None = None) -> str:
        msgs = [{k: v for k, v in m.items() if k != "images" or v} for m in messages]
        kwargs = {"think": self.cfg.think} if self.cfg.think is not None else {}
        resp = self.client.chat(
            model=self.model,
            messages=msgs,
            format=schema,
            options=self._options(max_tokens or self.cfg.max_tokens),
            keep_alive=self.cfg.keep_alive,
            **kwargs,
        )
        self.last_stats = {k: resp.get(k) for k in ("prompt_eval_count", "prompt_eval_duration", "eval_count",
                                                    "eval_duration", "total_duration")}
        return resp["message"]["content"]

    def stream_text(self, messages: list[dict], max_tokens: int = 256) -> Iterator[str]:
        """Free-text answer, yielded piece by piece as the model generates it."""
        kwargs = {"think": self.cfg.think} if self.cfg.think is not None else {}
        for chunk in self.client.chat(
            model=self.model, messages=messages, stream=True, keep_alive=self.cfg.keep_alive,
            options=self._options(max_tokens), **kwargs,
        ):
            if chunk.get("done"):
                self.last_stats = {k: chunk.get(k) for k in ("prompt_eval_count", "prompt_eval_duration",
                                                             "eval_count", "eval_duration", "total_duration")}
            delta = chunk["message"]["content"]
            if delta:
                yield delta

    def ping(self) -> bool:
        try:
            names = {m["model"] for m in self.client.list()["models"]}
            return self.model in names or f"{self.model}:latest" in names
        except Exception:
            return False


def openai_messages(messages: list[dict]) -> list[dict]:
    """Neutral messages -> OpenAI chat format, with images as data URLs."""
    out = []
    for m in messages:
        images = m.get("images") or []
        if images:
            parts = [{"type": "text", "text": m["content"]}]
            for img in images:
                b64 = base64.b64encode(img).decode("ascii")
                parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            out.append({"role": m["role"], "content": parts})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def openai_payload(model: str, messages: list[dict], schema: dict, name: str, max_tokens: int | None = None) -> dict:
    """Chat-completions request with a JSON-schema response format. Not 'strict': local servers
    differ in what strict mode accepts, and pydantic validates anyway. max_tokens caps decode time."""
    body = {
        "model": model,
        "messages": openai_messages(messages),
        "temperature": 0,
        "response_format": {"type": "json_schema", "json_schema": {"name": name, "schema": schema}},
    }
    if max_tokens:
        body["max_tokens"] = max_tokens
    return body


def choice_payload(model: str, prompt: str, choices: list[str]) -> dict:
    """vLLM structured outputs: the decoder may only emit one of `choices` (vLLM >= 0.12 field name;
    the old guided_choice was removed). logprobs give the label's probability."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 16,
        "logprobs": True,
        "structured_outputs": {"choice": choices},
    }


def sse_deltas(lines) -> Iterator[str]:
    """Text pieces from an OpenAI-style chat-completions event stream."""
    for line in lines:
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            choices = json.loads(data).get("choices") or []
        except json.JSONDecodeError:
            continue
        delta = (choices[0].get("delta") or {}).get("content") if choices else None
        if delta:
            yield delta


def label_confidence(choice: dict) -> float | None:
    """exp(sum of token log-probs) of the generated label: the model's probability of the whole label."""
    tokens = ((choice.get("logprobs") or {}).get("content")) or []
    if not tokens:
        return None
    return round(min(1.0, math.exp(sum(t.get("logprob", 0.0) for t in tokens))), 4)


class OpenAICompatClient(LlmClient):
    def __init__(self, cfg: LlmEndpoint):
        super().__init__(cfg)
        import httpx

        headers = {}
        if cfg.api_key_env and os.environ.get(cfg.api_key_env):
            headers["Authorization"] = f"Bearer {os.environ[cfg.api_key_env]}"
        base = cfg.host.rstrip("/")
        self.base = base if base.endswith("/v1") else base + "/v1"
        self.http = httpx.Client(timeout=cfg.timeout_s, headers=headers)

    def _chat(self, messages: list[dict], schema: dict, name: str, max_tokens: int | None = None) -> str:
        r = self.http.post(f"{self.base}/chat/completions", json=openai_payload(self.model, messages, schema, name,
                                                                         max_tokens or self.cfg.max_tokens))
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def _choose(self, prompt: str, choices: list[str]) -> tuple[str, float | None]:
        if self.cfg.choice_decoding:
            r = self.http.post(f"{self.base}/chat/completions", json=choice_payload(self.model, prompt, choices))
            r.raise_for_status()
            choice = r.json()["choices"][0]
            label = (choice["message"]["content"] or "").strip().strip('"')
            if label in choices:
                return label, label_confidence(choice)
            # a server that ignores structured_outputs answers in free text: fall back to the schema
            log.warning("choice decoding returned %r; falling back to a JSON-schema enum", label[:40])
        return super()._choose(prompt, choices)

    def stream_text(self, messages: list[dict], max_tokens: int = 256) -> Iterator[str]:
        """Free-text answer over server-sent events (OpenAI streaming format)."""
        body = {"model": self.model, "messages": openai_messages(messages), "temperature": 0,
                "max_tokens": max_tokens, "stream": True}
        with self.http.stream("POST", f"{self.base}/chat/completions", json=body) as r:
            r.raise_for_status()
            for delta in sse_deltas(r.iter_lines()):
                yield delta

    def ping(self) -> bool:
        try:
            r = self.http.get(f"{self.base}/models")
            r.raise_for_status()
            ids = {m.get("id") for m in r.json().get("data", [])}
            return not ids or self.model in ids  # some servers serve one model and list nothing useful
        except Exception:
            return False


def make_client(cfg: LlmEndpoint) -> LlmClient:
    return OllamaClient(cfg) if cfg.backend == "ollama" else OpenAICompatClient(cfg)


def VlmClient(cfg: LlmEndpoint, model: str | None = None) -> LlmClient:
    """Backwards-compatible helper: a client for `cfg`, optionally with another model."""
    return make_client(cfg.model_copy(update={"model": model}) if model else cfg)
