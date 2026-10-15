import math

import pytest

from minerva.config import CameraCfg, CrowdZone, EntryLine, TriggersCfg, Zones
from minerva.triggers.base import Episodes, line_side, point_in_polygon
from minerva.triggers.engine import TriggerEngine
from minerva.types import Category

from helpers import bag, person, run, scene

T0 = 1_700_000_000_000
STEP = 250  # 4 fps


def engine(zones=Zones(), **overrides):
    cfg = TriggersCfg(**overrides)
    return TriggerEngine(cfg, CameraCfg(camera_id="cam1", name="Test", source="x.mp4", zones=zones))


def frames(seconds, start=0, step=STEP):
    return [T0 + start + i * step for i in range(int(seconds * 1000 / step))]


# ---------- primitives ----------

def test_episodes_fire_once_after_sustain_and_survive_short_dropout():
    ep = Episodes(sustain_ms=1000, release_ms=500)
    fired_at = []
    for ts in range(0, 3000, 250):
        holding = [] if ts == 500 else ["k"]  # one-frame dropout must not reset the onset
        fired, _ = ep.step(ts, holding)
        if fired:
            fired_at.append(ts)
    assert fired_at == [1000]


def test_episodes_reset_after_long_dropout():
    ep = Episodes(sustain_ms=1000, release_ms=300)
    ep.step(0, ["k"])
    ep.step(600, [])  # gone for 600 ms > release
    assert not ep.active("k")


def test_geometry():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((5, 5), square)
    assert not point_in_polygon((15, 5), square)
    assert line_side((0, 0), (10, 0), (5, 5)) == 1  # image y grows downward
    assert line_side((0, 0), (10, 0), (5, -5)) == -1


# ---------- person down ----------

def test_person_down_fires_once_after_fall():
    e = engine()
    upright = [scene(ts, [person(1, 100, 100, 140, 220)]) for ts in frames(2)]
    lying = [scene(ts, [person(1, 80, 190, 200, 230)]) for ts in frames(5, start=2000)]
    cands, ongoing = run(e, upright + lying)
    assert len(cands) == 1
    c = cands[0]
    assert c.category == Category.PERSON_DOWN
    assert c.onset_ms == T0 + 2000
    assert c.trigger_ms - c.onset_ms >= 3000
    assert c.detail["method"] == "aspect"
    assert ongoing  # stays down -> episode extends instead of re-firing


def test_person_down_ignores_people_already_lying_and_crouching():
    e = engine()
    lying_from_start = [scene(ts, [person(1, 80, 190, 200, 230)]) for ts in frames(6)]
    crouching = [scene(ts, [person(2, 100, 100, 200, 210)]) for ts in frames(6, start=6000)]
    cands, _ = run(e, lying_from_start + crouching)
    assert cands == []


def test_person_down_uses_pose_when_confident():
    from minerva.types import PoseResult

    e = engine()
    seq = []
    for ts in frames(2):
        seq.append(scene(ts, [person(1, 100, 100, 140, 220)]))
    for ts in frames(4, start=2000):
        s = scene(ts, [person(1, 100, 150, 180, 230)])  # w/h = 1.0: box alone is ambiguous
        s.pose = lambda t: PoseResult(80.0, 0.9)
        seq.append(s)
    cands, _ = run(e, seq)
    assert len(cands) == 1 and cands[0].detail["method"] == "pose"


# ---------- fighting ----------

def _fight_scenes(motion, gap_x=0, seconds=3):
    out = []
    for i, ts in enumerate(frames(seconds)):
        j = 10 if i % 2 else -10  # back-and-forth jitter
        a = person(1, 100 + j, 100, 160 + j, 220)
        b = person(2, 150 - j + gap_x, 100, 210 - j + gap_x, 220)
        out.append(scene(ts, [a, b], motion=motion))
    return out


def test_fighting_fires_on_close_erratic_high_motion():
    cands, _ = run(engine(), _fight_scenes(motion=5.0))
    assert len(cands) == 1
    assert cands[0].category == Category.FIGHTING
    assert cands[0].track_ids == (1, 2)
    assert cands[0].score == pytest.approx(5 / 6, abs=1e-3)


def test_fighting_needs_motion_and_proximity():
    assert run(engine(), _fight_scenes(motion=1.0))[0] == []
    assert run(engine(), _fight_scenes(motion=5.0, gap_x=500))[0] == []


# ---------- tailgating ----------

LINE_ZONES = Zones(entry_line=EntryLine(p1=(0, 0.5), p2=(1, 0.5), entering="positive"))


def _walk(tid, start_ms, y_from, y_to, x=300, seconds=2, step=STEP):
    n = int(seconds * 1000 / step)
    return {T0 + start_ms + i * step: person(tid, x, y_from + (y_to - y_from) * i / (n - 1) - 120,
                                             x + 50, y_from + (y_to - y_from) * i / (n - 1))
            for i in range(n)}


def _merge(*walks):
    by_ts = {}
    for w in walks:
        for ts, p in w.items():
            by_ts.setdefault(ts, []).append(p)
    return [scene(ts, by_ts[ts]) for ts in sorted(by_ts)]


def test_tailgating_two_people_close_together():
    seq = _merge(_walk(1, 0, 400, 600), _walk(2, 1000, 400, 600, x=500))
    cands, _ = run(engine(LINE_ZONES), seq)
    assert len(cands) == 1
    assert cands[0].category == Category.TAILGATING
    assert set(cands[0].track_ids) == {1, 2}
    assert 0 < cands[0].score <= 1


def test_tailgating_ignores_spaced_entries_and_exits():
    spaced = _merge(_walk(1, 0, 400, 600), _walk(2, 6000, 400, 600, x=500))
    assert run(engine(LINE_ZONES), spaced)[0] == []
    exits = _merge(_walk(1, 0, 600, 400), _walk(2, 500, 600, 400, x=500))
    assert run(engine(LINE_ZONES), exits)[0] == []


def test_third_person_extends_group_instead_of_new_event():
    seq = _merge(_walk(1, 0, 400, 600), _walk(2, 750, 400, 600, x=500), _walk(3, 1500, 400, 600, x=700))
    cands, ongoing = run(engine(LINE_ZONES), seq)
    assert len(cands) == 1
    assert ongoing


# ---------- crowd ----------

def test_overcrowding_after_sustain():
    zones = Zones(crowd_zone=CrowdZone(max_people=3))
    crowd = [person(i, 100 * i, 100, 100 * i + 50, 220) for i in range(1, 5)]
    seq = [scene(ts, crowd) for ts in frames(12, step=500)]
    cands, _ = run(engine(zones), seq)
    assert len(cands) == 1
    c = cands[0]
    assert (c.category, c.subtype) == (Category.CROWD_INCIDENT, "overcrowding")
    assert c.trigger_ms - c.onset_ms >= 10_000
    assert c.score == pytest.approx(min(1, (4 - 3) / 3 + 0.5), abs=1e-3)


def test_no_overcrowding_below_limit():
    zones = Zones(crowd_zone=CrowdZone(max_people=5))
    crowd = [person(i, 100 * i, 100, 100 * i + 50, 220) for i in range(1, 5)]
    assert run(engine(zones), [scene(ts, crowd) for ts in frames(12, step=500)])[0] == []


# ---------- suspicious ----------

WATCH = Zones(watch_zones={"lobby": [(0, 0), (1, 0), (1, 1), (0, 1)]})


def test_loitering_fires_after_dwell():
    seq = []
    for i, ts in enumerate(frames(70, step=1000)):
        j = 5 * math.sin(i)
        seq.append(scene(ts, [person(1, 400 + j, 300, 450 + j, 420)]))
    cands, _ = run(engine(WATCH), seq)
    assert len(cands) == 1
    c = cands[0]
    assert (c.subtype, c.zone) == ("loitering", "lobby")
    assert c.trigger_ms - c.onset_ms >= 60_000


def test_walking_through_is_not_loitering():
    seq = [scene(ts, [person(1, 10 * i, 300, 10 * i + 50, 420)]) for i, ts in enumerate(frames(70, step=1000))]
    assert run(engine(WATCH), seq)[0] == []


def test_abandoned_bag_after_owner_leaves():
    seq = []
    for i, ts in enumerate(frames(60, step=500)):
        t = i * 0.5
        persons, objects = [], []
        if t >= 12:
            objects.append(bag(1_000_001, 500, 400, 540, 440))
        if 12 <= t < 15:
            persons.append(person(7, 480, 300, 530, 440))  # owner standing next to the bag
        elif t >= 15:
            persons.append(person(7, 900, 300, 950, 440))  # owner walked away
        seq.append(scene(ts, persons, objects))
    cands, _ = run(engine(), seq)
    assert len(cands) == 1
    c = cands[0]
    assert c.subtype == "abandoned_object"
    assert c.detail["owner_track"] == 7
    assert c.trigger_ms - c.onset_ms >= 30_000


def test_bag_present_at_start_is_background():
    seq = [scene(ts, [], [bag(1_000_001, 500, 400, 540, 440)]) for ts in frames(60, step=500)]
    assert run(engine(), seq)[0] == []
