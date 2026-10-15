"""Trained-model integration (kaggle/ notebooks): clip scorer timing and the clip path in the fight rule."""

import numpy as np

from minerva.triggers.engine import TriggerEngine
from minerva.config import CameraCfg, TriggersCfg
from minerva.types import Category
from minerva.vision.violence import ViolenceScorer

from helpers import person, run, scene

T0 = 1_700_000_000_000


class FakeModel:
    clip_len, clip_fps, threshold = 8, 4.0, 0.6

    def __init__(self):
        self.calls = []

    def prepare(self, frame):
        return frame  # frames carry their own timestamp value for the test

    def score(self, frames):
        self.calls.append([int(f[0, 0, 0]) for f in frames])
        return 0.9


def test_scorer_waits_for_history_then_runs_on_stride_with_time_matched_frames():
    m = FakeModel()
    s = ViolenceScorer(m, stride_ms=500)
    results = []
    for i in range(17):  # 8 fps input for ~2 s; the model wants 4 fps x 8 frames
        ts = T0 + i * 125
        results.append(s.update(ts, np.full((2, 2, 3), i, np.uint8)))
    assert results[0] is None  # not enough history at first
    assert results[11] is None and results[12] == 0.9  # first score once 1.5 s of history exists
    assert len(m.calls) == 2  # at frames 12 and 16: every 500 ms, not every frame
    assert m.calls[-1] == [2, 4, 6, 8, 10, 12, 14, 16]  # 250 ms apart, oldest to newest, ends now


def _engine():
    return TriggerEngine(TriggersCfg(), CameraCfg(camera_id="cam1", name="t", source="x"))


def test_clip_model_triggers_fighting_without_close_tracks():
    # Two people far apart (the tracker-based path cannot fire), but the clip model is confident.
    seq = []
    for i in range(10):
        s = scene(T0 + i * 250, [person(1, 10, 10, 60, 130), person(2, 800, 10, 850, 130)])
        s.violence, s.violence_threshold = 0.9, 0.6
        seq.append(s)
    cands, _ = run(_engine(), seq)
    assert len(cands) == 1
    c = cands[0]
    assert c.category == Category.FIGHTING and c.detail["source"] == "clip_model"
    assert c.track_ids == (1, 2) and c.score == 0.9


def test_clip_model_below_threshold_or_nobody_in_view_does_not_fire():
    low = []
    for i in range(10):
        s = scene(T0 + i * 250, [person(1, 10, 10, 60, 130)])
        s.violence, s.violence_threshold = 0.5, 0.6
        low.append(s)
    assert run(_engine(), low)[0] == []
    empty = []
    for i in range(10):
        s = scene(T0 + i * 250, [])
        s.violence, s.violence_threshold = 0.95, 0.6
        empty.append(s)
    assert run(_engine(), empty)[0] == []
