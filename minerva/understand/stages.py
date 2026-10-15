"""Runs the two understanding stages for one key-frame grid: [1] the VLM summarizes it, [2] a
text-only call classifies the summary. Each stage is timed on its own (faculty requirement R5)."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from minerva.understand.prompts import VlmAnswer


@dataclass
class Understanding:
    answer: VlmAnswer | None  # None if either stage failed
    summary: str | None  # kept even when classification failed
    summarize_ms: float | None
    classify_ms: float | None

    @property
    def vlm_ms(self) -> float | None:
        """Both stages together, the old single 'vlm' timing."""
        parts = [t for t in (self.summarize_ms, self.classify_ms) if t is not None]
        return sum(parts) if parts else None


def understand(vlm, classifier, grid_jpeg: bytes, highlighted: bool) -> Understanding:
    s, summarize_ms = vlm.summarize(grid_jpeg, highlighted=highlighted)
    if s is None:
        return Understanding(None, None, summarize_ms, None)
    label, conf, classify_ms = classifier.classify_text(s.summary, s.visible_evidence)
    summary = s.summary.strip()
    if label is None:
        return Understanding(None, summary, summarize_ms, classify_ms)
    ans = VlmAnswer(category=label, summary=summary, visible_evidence=s.visible_evidence, confidence=conf)
    return Understanding(ans, summary, summarize_ms, classify_ms)


class InOrder:
    """Lets concurrent workers finish out of order while results are applied in submission order,
    per key. Periodic mode needs this: WindowAggregator assumes each camera's windows arrive in time
    order. Dropped jobs must call skip() so later results are not held back forever."""

    def __init__(self):
        self._lock = threading.Lock()
        self._next_seq: dict[str, int] = {}  # key -> next sequence number to hand out
        self._next_apply: dict[str, int] = {}
        self._done: dict[str, dict[int, object]] = {}
        self._applying: set[str] = set()

    def ticket(self, key: str) -> int:
        with self._lock:
            seq = self._next_seq.get(key, 0)
            self._next_seq[key] = seq + 1
            return seq

    def skip(self, key: str, seq: int, apply) -> None:
        self.finish(key, seq, _SKIP, apply)

    def finish(self, key: str, seq: int, result, apply) -> None:
        """Record `result` for ticket `seq`, then apply every result that is now next in line.
        Only one thread applies per key at a time, so `apply` never runs concurrently for a key."""
        with self._lock:
            self._done.setdefault(key, {})[seq] = result
            if key in self._applying:
                return  # the thread already applying will pick this one up
            self._applying.add(key)
        try:
            while True:
                with self._lock:
                    n = self._next_apply.get(key, 0)
                    done = self._done[key]
                    if n not in done:
                        self._applying.discard(key)
                        return
                    item = done.pop(n)
                    self._next_apply[key] = n + 1
                if item is not _SKIP:
                    apply(item)
        except BaseException:
            with self._lock:
                self._applying.discard(key)
            raise


_SKIP = object()
