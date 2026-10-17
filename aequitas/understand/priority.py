"""People first: a question an officer is waiting on goes ahead of background video analysis.

The laptop has one GPU and one model server. Live CCTV keeps sending windows to it, so a chat question used
to queue behind them (measured: 1.3 s idle, 5.0 s with a locality running). While any interactive request
(chat, per-video questions, translation) is in progress, the pipeline's workers wait before starting their
next window, so a question only ever waits for the window already being analysed. The wait is capped, so a
stuck request can't stop the cameras for long."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

MAX_HOLD_S = 20.0


class PriorityGate:
    def __init__(self):
        self._n = 0
        self._cv = threading.Condition()

    @contextmanager
    def interactive(self):
        """Wrap work a person is waiting on."""
        with self._cv:
            self._n += 1
        try:
            yield
        finally:
            with self._cv:
                self._n -= 1
                self._cv.notify_all()

    @property
    def busy(self) -> bool:
        return self._n > 0

    def wait_turn(self, max_wait_s: float = MAX_HOLD_S) -> float:
        """Background work calls this before each job. Returns the seconds it waited."""
        t0 = time.monotonic()
        with self._cv:
            while self._n > 0:
                left = max_wait_s - (time.monotonic() - t0)
                if left <= 0:
                    break
                self._cv.wait(min(left, 0.25))
        return time.monotonic() - t0


GATE = PriorityGate()
