"""Model-independent LLM/VLM clients. The same interface serves the VLM check and the chatbot, and
either can point at Ollama or at any local OpenAI-compatible server (llama.cpp, vLLM, LM Studio).

Messages use one neutral shape: {"role": ..., "content": str, "images": [jpeg bytes]}."""

from __future__ import annotations

import base64
import ipaddress
import logging
import os
import time
from urllib.parse import urlparse

from pydantic import BaseModel, ValidationError

from minerva.config import LlmEndpoint
from minerva.understand.prompts import VlmAnswer, classify_prompt

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
                f"{cfg.host} is not a local address. Minerva only uses local inference; "
                "set allow_remote: true if this host is your own server."
            )
        self.cfg = cfg
        self.model = cfg.model

    def _chat(self, messages: list[dict], schema: dict, name: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def ping(self) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def structured(self, messages: list[dict], model_cls: type[BaseModel], retries: int = 1) -> BaseModel | None:
        schema = model_cls.model_json_schema()
        for attempt in range(retries + 1):
            try:
                raw = self._chat(messages, schema, model_cls.__name__)
                return model_cls.model_validate_json(raw)
            except ValidationError as e:
                log.warning("invalid structured output (attempt %d): %s", attempt + 1, e.errors()[:2])
            except Exception as e:  # connection errors, timeouts
                log.warning("%s call failed (attempt %d): %s", self.cfg.backend, attempt + 1, e)
        return None

    def classify(self, grid_jpeg: bytes, highlighted: bool = True) -> tuple[VlmAnswer | None, float]:
        t0 = time.perf_counter()
        ans = self.structured(
            [{"role": "user", "content": classify_prompt(highlighted), "images": [grid_jpeg]}], VlmAnswer
        )
        return ans, (time.perf_counter() - t0) * 1000


class OllamaClient(LlmClient):
    def __init__(self, cfg: LlmEndpoint):
        super().__init__(cfg)
        import ollama

        self.client = ollama.Client(host=cfg.host, timeout=cfg.timeout_s)

    def _chat(self, messages: list[dict], schema: dict, name: str) -> str:
        msgs = [{k: v for k, v in m.items() if k != "images" or v} for m in messages]
        kwargs = {"think": self.cfg.think} if self.cfg.think is not None else {}
        resp = self.client.chat(
            model=self.model,
            messages=msgs,
            format=schema,
            options={"temperature": 0, "num_ctx": self.cfg.num_ctx},
            keep_alive=self.cfg.keep_alive,
            **kwargs,
        )
        return resp["message"]["content"]

    def ping(self) -> bool:
        try:
            names = {m["model"] for m in self.client.list()["models"]}
            return self.model in names or f"{self.model}:latest" in names
        except Exception:
            return False


def openai_payload(model: str, messages: list[dict], schema: dict, name: str) -> dict:
    """Chat-completions request with images as data URLs and a JSON-schema response format.
    Not 'strict': local servers differ in what strict mode accepts, and pydantic validates anyway."""
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
    return {
        "model": model,
        "messages": out,
        "temperature": 0,
        "response_format": {"type": "json_schema", "json_schema": {"name": name, "schema": schema}},
    }


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

    def _chat(self, messages: list[dict], schema: dict, name: str) -> str:
        r = self.http.post(f"{self.base}/chat/completions", json=openai_payload(self.model, messages, schema, name))
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

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
