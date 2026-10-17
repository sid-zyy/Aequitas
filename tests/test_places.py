"""The chat knows where places are: an area (locality) means all its cameras, misspellings are forgiven,
naming one camera narrows to it, answers say which area each camera is in, and "Where is ...?" is answered."""

from datetime import datetime

import pytest

from aequitas.chat.answer import ChatService
from aequitas.chat.parser import QuestionParser, keyword_filter, resolve_cameras
from aequitas.config import CameraCfg, Settings
from aequitas.store.db import EventRecord, EventStore
from aequitas.web.live import LiveManager, load_localities

NOW = datetime(2026, 10, 17, 15, 0).astimezone()


def _cam(cid, name, area, aliases=()):
    return CameraCfg(camera_id=cid, name=name, aliases=[f"{area} {name}"], area=area, area_aliases=list(aliases),
                     source="x.mp4")


CAMS = [
    _cam("kor-forum", "Forum Mall, Hosur Road", "Koramangala", ["Kormangala"]),
    _cam("kor-80ft", "80 Feet Road", "Koramangala", ["Kormangala"]),
    _cam("ind-100ft", "100 Feet Road", "Indiranagar", ["Indira Nagar"]),
    _cam("ind-double-road", "Double Road Restaurant", "Indiranagar", ["Indira Nagar"]),
    CameraCfg(camera_id="cam2", name="Corridor", aliases=["camera 2"], source="y.mp4"),  # no area
]


@pytest.fixture
def store(tmp_path):
    s = EventStore(tmp_path / "t.db")
    for c in CAMS:
        s.upsert_camera(c)
    return s


def _ids(store, question):
    cams = store.cameras()
    return sorted(resolve_cameras(keyword_filter(question, NOW, cams).cameras, cams)[0])


@pytest.mark.parametrize("question,want", [
    ("Any fights in Indiranagar today?", ["ind-100ft", "ind-double-road"]),
    ("what happened in indira nagar", ["ind-100ft", "ind-double-road"]),
    ("fights in kormangala", ["kor-80ft", "kor-forum"]),  # alias
    ("show incidents at koramangla", ["kor-80ft", "kor-forum"]),  # a misspelling no alias covers
    ("Any fights at Forum Mall in Koramangala?", ["kor-forum"]),  # the camera, not the whole area
    ("accidents on 80 feet road", ["kor-80ft"]),
    ("anything at double road restaurant", ["ind-double-road"]),
    ("Any fires today?", []),  # no place named
])
def test_places_resolve_to_cameras(store, question, want):
    assert _ids(store, question) == want


def test_llm_area_mention_means_every_camera_there(store):
    ids, unknown = resolve_cameras(["Indiranagar"], store.cameras())
    assert sorted(ids) == ["ind-100ft", "ind-double-road"] and not unknown
    ids, unknown = resolve_cameras(["Kormangala"], store.cameras())
    assert sorted(ids) == ["kor-80ft", "kor-forum"]
    assert resolve_cameras(["Atlantis"], store.cameras()) == ([], ["Atlantis"])


def _event(store, cam, cat, summary):
    now = int(datetime.now().timestamp() * 1000)
    store.insert_event(EventRecord(f"{cam}-{cat}", cam, now - 60_000, now - 55_000, cat, None, "confirmed", 0.9,
                                   0.9, cat, summary, None, [], "k.jpg", {}))


def test_answers_name_the_area_and_where_questions_are_answered(store):
    _event(store, "ind-double-road", "fighting", "Two people fight near tables.")
    _event(store, "kor-80ft", "accident", "A car hits a scooter.")
    chat = ChatService(store, QuestionParser(None))

    a = chat.ask("Any fights in Indiranagar?")
    assert a.total == 1 and "in Indiranagar (all 2 cameras)" in a.text
    assert "Double Road Restaurant, Indiranagar" in a.text

    w = chat.ask("Where is Double Road Restaurant?")
    assert w.text.startswith("Double Road Restaurant is in Indiranagar (camera ind-double-road).")
    w = chat.ask("where is kormangala")
    assert w.text.startswith("Koramangala is an area with 2 camera points: 80 Feet Road, Forum Mall, Hosur Road.")
    assert chat.ask("Where is the corridor?").text.startswith("Corridor is camera cam2; it isn't assigned to an area.")


def test_live_localities_register_their_area_and_keep_it(tmp_path):
    yml = tmp_path / "loc.yaml"
    yml.write_text("localities:\n  - id: indiranagar\n    name: Indiranagar\n    aliases: [Indira Nagar]\n"
                   "    cameras:\n      - {id: ind-12main, name: 12th Main, file: none.mp4}\n", encoding="utf-8")
    store = EventStore(tmp_path / "t.db")
    LiveManager(Settings(data_dir=tmp_path), store, load_localities(yml))
    cam = store.cameras()[0]
    assert cam["area"] == "Indiranagar" and cam["area_aliases"] == ["Indira Nagar"]
    assert cam["aliases"] == ["Indiranagar 12th Main"]
