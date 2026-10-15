"""Periodic mode (no YOLO): merges per-window VLM answers into events. With no trigger to agree
with, confidence comes from agreement across consecutive windows (docs/DESIGN.md §3.5)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Literal

from aequitas.types import Status
from aequitas.understand.prompts import VlmAnswer


@dataclass
class OpenEvent:
    event_uid: str
    category: str
    start_ms: int
    end_ms: int
    votes: int


@dataclass
class Decision:
    action: Literal["none", "new", "update"]
    event: OpenEvent | None = None
    status: Status | None = None
    confidence: float = 0.0
    summary: str | None = None


class WindowAggregator:
    def __init__(self, confirm_votes: int = 2, max_gap_ms: int = 10_000):
        self.confirm_votes = max(1, confirm_votes)
        self.max_gap_ms = max_gap_ms
        self._open: dict[str, OpenEvent] = {}
        self._last: dict[str, tuple[set[str], int, int]] = {}  # camera -> (categories voted, start, end)

    def score(self, votes: int) -> tuple[Status, float]:
        if votes >= self.confirm_votes:
            return Status.CONFIRMED, min(0.9, 0.7 + 0.05 * (votes - self.confirm_votes))
        return Status.UNVERIFIED, 0.4

    def add(self, camera_id: str, start_ms: int, end_ms: int, ans: VlmAnswer | None,
            also: frozenset[str] = frozenset()) -> Decision:
        """`also`: categories a second signal votes for in this window besides the VLM's label (the violence
        clip model at its stand-alone threshold votes 'fighting'). Short clips have only two windows; when the
        VLM calls the second one person_down while the clip model scores both 0.98+, the fight still confirms."""
        if ans is None:  # VLM failure: neither open nor close anything
            return Decision("none")
        votes_now = ({ans.category} - {"none"}) | set(also)
        last = self._last.get(camera_id)
        self._last[camera_id] = (votes_now, start_ms, end_ms)
        cur = self._open.get(camera_id)
        if cur is not None and cur.category in votes_now and start_ms - cur.end_ms <= self.max_gap_ms:
            cur.votes += 1
            cur.end_ms = max(cur.end_ms, end_ms)
            status, conf = self.score(cur.votes)
            return Decision("update", cur, status, conf)
        if ans.category == "none":
            self._open.pop(camera_id, None)
            return Decision("none")
        ev = OpenEvent(uuid.uuid4().hex, ans.category, start_ms, end_ms, 1)
        if last is not None and ans.category in last[0] and start_ms - last[2] <= self.max_gap_ms:
            ev.votes, ev.start_ms = 2, last[1]  # the previous window already voted for this category
        self._open[camera_id] = ev
        status, conf = self.score(ev.votes)
        return Decision("new", ev, status, conf, ans.summary.strip())
