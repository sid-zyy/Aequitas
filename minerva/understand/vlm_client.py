"""Thin wrapper over a local Ollama server. Used for both the VLM check and the chatbot's
question parsing, so only one heavy model is ever loaded."""

from __future__ import annotations

import logging
import time

from pydantic import BaseModel, ValidationError

from minerva.config import VlmCfg
from minerva.understand.prompts import VlmAnswer, classify_prompt

log = logging.getLogger(__name__)


class VlmClient:
    def __init__(self, cfg: VlmCfg, model: str | None = None):
        import ollama

        self.cfg = cfg
        self.model = model or cfg.model
        self.client = ollama.Client(host=cfg.host, timeout=cfg.timeout_s)

    def _chat(self, messages: list[dict], schema: dict) -> str:
        resp = self.client.chat(
            model=self.model,
            messages=messages,
            format=schema,
            options={"temperature": 0, "num_ctx": self.cfg.num_ctx},
            keep_alive=self.cfg.keep_alive,
        )
        return resp["message"]["content"]

    def structured(self, messages: list[dict], model_cls: type[BaseModel], retries: int = 1) -> BaseModel | None:
        schema = model_cls.model_json_schema()
        for attempt in range(retries + 1):
            try:
                raw = self._chat(messages, schema)
                return model_cls.model_validate_json(raw)
            except ValidationError as e:
                log.warning("invalid structured output (attempt %d): %s", attempt + 1, e.errors()[:2])
            except Exception as e:  # connection errors, timeouts
                log.warning("Ollama call failed (attempt %d): %s", attempt + 1, e)
        return None

    def classify(self, grid_jpeg: bytes) -> tuple[VlmAnswer | None, float]:
        t0 = time.perf_counter()
        ans = self.structured(
            [{"role": "user", "content": classify_prompt(), "images": [grid_jpeg]}], VlmAnswer
        )
        return ans, (time.perf_counter() - t0) * 1000

    def ping(self) -> bool:
        try:
            names = {m["model"] for m in self.client.list()["models"]}
            return self.model in names or f"{self.model}:latest" in names
        except Exception:
            return False
