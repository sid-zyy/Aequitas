"""Chat typo fixing (aequitas/chat/spelling.py) and the "last hour" range ending now, to the second."""

from datetime import datetime, timedelta

from aequitas.chat.answer import _STOP, ChatService
from aequitas.chat.parser import QuestionParser, finalize, keyword_filter
from aequitas.chat.spelling import correct, distance, targets
from aequitas.config import CameraCfg
from aequitas.store.db import EventRecord, EventStore
from aequitas.taxonomy import DEFAULT_CATEGORIES

VOCAB = targets(DEFAULT_CATEGORIES)
KNOWN = set(_STOP) | {"forum", "mall", "ladder"}


def fix(q):
    return correct(q, VOCAB, KNOWN)[0]


def test_common_typos_are_fixed():
    assert fix("Any fire incdent") == "Any fire incident"
    assert fix("acident on the road") == "accident on the road"
    assert fix("figths today") == "fights today"  # swapped letters count as one edit
    assert fix("any harasment yesterday") == "any harassment yesterday"
    assert fix("robery near forum mall") == "robbery near forum mall"
    assert fix("Stampeed?") == "Stampede?"


def test_real_words_are_left_alone():
    for q in ["Anyone near the forest?", "bright lights at night", "chasing a thief", "a man with a ladder",
              "Any incidents in the last hour?", "people running past the mall"]:
        assert fix(q) == q, q
    assert correct("a flight of stairs", VOCAB, KNOWN)[1] == []  # one letter from "fight", but a real word
    assert correct("the tyres were smoking", VOCAB, KNOWN | {"smoking"})[1] == []  # seen in a summary: real


def test_ambiguous_typos_are_not_guessed():
    assert fix("flams") == "flams"  # flame or flames: equally close


def test_distance_counts_a_swap_as_one():
    assert distance("figths", "fights") == 1 and distance("incdent", "incident") == 1
    assert distance("forest", "arrest") == 2


def test_chat_reads_the_fixed_question_and_says_so(tmp_path):
    s = EventStore(tmp_path / "t.db")
    s.upsert_camera(CameraCfg(camera_id="cam1", name="Gate", source="a.mp4"))
    t = int(datetime.now().timestamp() * 1000) - 3_600_000 * 3
    eid = s.insert_event(EventRecord(event_uid="e1", camera_id="cam1", start_ts=t, end_ts=t + 4000, category="fire",
                                     subtype=None, status="confirmed", confidence=0.9, trigger_score=0,
                                     vlm_category="fire", summary="Smoke rises from a parked scooter.", zone=None,
                                     track_ids=[], keyframe_path="x.jpg", model_versions={}))
    ans = ChatService(s, QuestionParser(None), summarize=False).ask("Any fire incdent")
    assert ans.text.startswith("Read as: “Any fire incident”")
    assert [r["event_id"] for r in ans.rows] == [eid]


def test_last_hour_ends_now_to_the_second():
    now = datetime(2026, 10, 17, 12, 30, 55).astimezone()
    f = finalize(keyword_filter("any incidents in the last hour", now, []), now, []).filter
    assert f.end == now  # not 12:30:00, which missed an event at 12:30:41
    assert f.start == (now - timedelta(hours=1)).replace(second=0)


def test_a_camera_answers_to_the_start_of_its_name_when_that_is_unique():
    from aequitas.chat.parser import short_names

    cams = [{"camera_id": "a", "name": "Jyoti Nivas College Road", "aliases": [], "area": "Koramangala"},
            {"camera_id": "b", "name": "MG Road Metro Station", "aliases": [], "area": None},
            {"camera_id": "c", "name": "MG Road Boulevard Gate", "aliases": [], "area": None}]
    assert short_names(cams) == {"a": "Jyoti Nivas"}  # "MG Road" is shared, so it names neither
    now = datetime(2026, 10, 17, 12, 0).astimezone()
    assert finalize(keyword_filter("accident at jyoti nivas", now, cams), now, cams).filter.camera_ids == ["a"]
    assert finalize(keyword_filter("accident at jyothi nivas", now, cams), now, cams).filter.camera_ids == ["a"]
