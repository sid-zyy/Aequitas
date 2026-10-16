"""CameraTracker on synthetic detections (Ultralytics trackers, no model weights needed)."""

import numpy as np
import pytest
import supervision as sv

from aequitas.config import DetectorCfg, TrackerCfg
from aequitas.vision.detector import CameraTracker

DET = DetectorCfg(min_person_conf=0.0, min_person_height_px=0)


def dets(*rows):
    """rows of (x1, y1, x2, y2, conf, class_id)"""
    if not rows:
        return sv.Detections.empty()
    a = np.array(rows, dtype=float)
    return sv.Detections(xyxy=a[:, :4], confidence=a[:, 4], class_id=a[:, 5].astype(int))


def walk(tracker, frames):
    return [tracker.update(d) for d in frames]


@pytest.mark.parametrize("algorithm", ["bytetrack", "ocsort"])
def test_person_keeps_one_id_while_walking(algorithm):
    tr = CameraTracker(TrackerCfg(algorithm=algorithm), DET, fps=4)
    out = walk(tr, [dets((100 + 3 * i, 100, 140 + 3 * i, 200, 0.9, 0)) for i in range(10)])
    ids = {p.track_id for persons, _ in out for p in persons}
    assert len(ids) == 1
    assert all(len(persons) == 1 for persons, _ in out[1:])


def test_activation_threshold_is_the_real_start_threshold():
    cfg = TrackerCfg(activation_threshold=0.3)
    at = walk(CameraTracker(cfg, DET, fps=4), [dets((100, 100, 140, 200, 0.3, 0))] * 4)
    below = walk(CameraTracker(cfg, DET, fps=4), [dets((100, 100, 140, 200, 0.29, 0))] * 4)
    assert all(persons for persons, _ in at), "a person at exactly the threshold must be tracked"
    assert not any(persons for persons, _ in below)


@pytest.mark.parametrize("fuse_score,kept", [(True, False), (False, True)])
def test_fuse_score_drops_detections_at_or_below_one_minus_matching_threshold(fuse_score, kept):
    # cost = 1 - IoU * 0.2 >= 0.8 = matching_threshold: the track starts on frame 1 and never matches again
    cfg = TrackerCfg(activation_threshold=0.15, matching_threshold=0.8, fuse_score=fuse_score)
    out = walk(CameraTracker(cfg, DET, fps=4), [dets((100, 100, 140, 200, 0.2, 0))] * 4)
    assert bool(out[-1][0]) is kept


def test_reports_the_detection_box_and_confidence():
    tr = CameraTracker(TrackerCfg(), DET, fps=4)
    out = walk(tr, [dets((100 + 5 * i, 100, 140 + 5 * i, 200, 0.8, 0)) for i in range(5)])
    p = out[-1][0][0]
    assert p.box == (120.0, 100.0, 160.0, 200.0)
    assert p.conf == pytest.approx(0.8)
    assert p.label == "person"


def test_people_and_objects_are_tracked_separately_with_offset_ids():
    tr = CameraTracker(TrackerCfg(), DET, fps=4)
    frame = dets((100, 100, 140, 200, 0.9, 0), (102, 150, 130, 190, 0.9, 24))  # overlapping person and backpack
    persons, objects = walk(tr, [frame] * 3)[-1]
    assert [p.label for p in persons] == ["person"]
    assert [o.label for o in objects] == ["backpack"]
    assert persons[0].track_id < 1_000_000 <= objects[0].track_id


@pytest.mark.parametrize("gap,same", [(4, True), (12, False)])
def test_lost_track_buffer_is_in_frames(gap, same):
    tr = CameraTracker(TrackerCfg(lost_track_buffer=8), DET, fps=4)
    box = dets((100, 100, 140, 200, 0.9, 0))
    first = walk(tr, [box] * 3)[-1][0][0].track_id
    walk(tr, [dets()] * gap)
    again = walk(tr, [box] * 3)[-1][0][0].track_id
    assert (again == first) is same


def test_empty_frames_and_min_person_filters():
    tr = CameraTracker(TrackerCfg(), DetectorCfg(min_person_conf=0.5, min_person_height_px=50), fps=4)
    assert tr.update(dets()) == ([], [])
    out = walk(tr, [dets((100, 100, 140, 130, 0.9, 0), (300, 100, 340, 200, 0.4, 0))] * 3)
    assert out[-1] == ([], [])  # one too short, one below min_person_conf
