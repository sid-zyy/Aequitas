"""Configurable categories: the police list by default, replaceable from settings (faculty dataset)."""

import sqlite3

import pytest

from minerva.chat.parser import QuestionParser, keyword_filter
from minerva.config import Settings
from minerva.store.db import EventRecord, EventStore
from minerva.taxonomy import DEFAULT_CATEGORIES, choices, label
from minerva.understand.prompts import classify_text_prompt, label_model
from minerva.understand.vlm_client import LlmClient
from minerva.config import LlmEndpoint

from test_store_chat import NOW


def test_default_list_has_the_police_categories_and_none():
    for c in ["arrest", "fire", "riot", "chase", "weapon", "fighting", "person_down"]:
        assert c in DEFAULT_CATEGORIES
    assert choices(DEFAULT_CATEGORIES)[-1] == "none" and label("person_down") == "person down"


def test_guardrail_peaceful_gatherings_are_none():
    p = classify_text_prompt("A large group walks in a procession holding flags.")
    assert "peaceful gathering" in p and "procession" in p


def test_settings_replace_the_category_list_and_reject_bad_names():
    s = Settings.model_validate({"categories": {"shoplifting": "a person hides goods", "loitering": "stays"}})
    assert list(s.categories) == ["shoplifting", "loitering"]
    for bad in ({"none": "x"}, {"Bad Name": "x"}, {}):
        with pytest.raises(ValueError):
            Settings.model_validate({"categories": bad})


class Echo(LlmClient):
    def __init__(self, reply):
        super().__init__(LlmEndpoint())
        self.reply, self.prompts, self.schemas = reply, [], []

    def _chat(self, messages, schema, name):
        self.prompts.append(messages[0]["content"])
        self.schemas.append(schema)
        return self.reply


def test_classifier_uses_the_configured_list():
    custom = {"shoplifting": "a person hides goods and leaves without paying"}
    c = Echo('{"category": "shoplifting"}')
    label_, _, _ = c.classify_text("A man hides a bottle in his coat.", categories=custom)
    assert label_ == "shoplifting"
    assert "shoplifting" in c.prompts[0] and "riot" not in c.prompts[0]
    assert c.schemas[0]["properties"]["category"]["enum"] == ["shoplifting", "none"]
    # a default label is invalid under the custom list
    assert Echo('{"category": "riot"}').classify_text("x", categories=custom)[0] is None
    assert label_model(("a", "none")) is label_model(("a", "none"))  # cached


@pytest.mark.parametrize("question,cat", [
    ("show all arrests today", "arrest"), ("was there any fire near the market?", "fire"),
    ("any stone pelting at the junction", "riot"), ("list chases this week", "chase"),
    ("did anyone carry a knife", "weapon"), ("chain snatching incidents", "theft"),
])
def test_keyword_parser_knows_the_police_categories(question, cat):
    assert cat in keyword_filter(question, NOW, []).categories


def test_parser_drops_categories_outside_the_list(tmp_path):
    from minerva.chat.parser import LlmFilter, finalize

    f = finalize(LlmFilter(categories=["fire", "made_up"]), NOW, []).filter
    assert f.categories == ["fire"]
    assert QuestionParser(None, categories={"fire": "x"}).categories == {"fire": "x"}


def _event(uid, cat):
    return EventRecord(uid, "cam1", 0, 1000, cat, None, "confirmed", 0.8, 0.0, cat, "s", None, [], "k.jpg", {})


def test_new_categories_are_stored_and_old_databases_lose_the_check(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE cameras (camera_id TEXT PRIMARY KEY, name TEXT NOT NULL, aliases TEXT NOT NULL DEFAULT '[]',"
        " source_uri TEXT NOT NULL, zones TEXT NOT NULL DEFAULT '{}');"
        "INSERT INTO cameras VALUES ('cam1','Gate','[]','x','{}');"
        "CREATE TABLE events (event_id INTEGER PRIMARY KEY, event_uid TEXT NOT NULL UNIQUE,"
        " camera_id TEXT NOT NULL REFERENCES cameras(camera_id), start_ts INTEGER NOT NULL, end_ts INTEGER NOT NULL,"
        " category TEXT NOT NULL CHECK (category IN\n ('person_down','fighting','tailgating','crowd_incident',"
        "'suspicious_activity')), subtype TEXT, status TEXT NOT NULL, confidence REAL NOT NULL, trigger_score REAL"
        " NOT NULL, vlm_category TEXT, summary TEXT, zone TEXT, track_ids TEXT NOT NULL DEFAULT '[]',"
        " keyframe_path TEXT NOT NULL, clip_path TEXT, model_versions TEXT NOT NULL, created_at INTEGER NOT NULL);"
        "INSERT INTO events VALUES (7,'old','cam1',0,1,'fighting',NULL,'confirmed',0.9,0.5,'fighting',"
        "'Two men shove each other.',NULL,'[]','k.jpg',NULL,'{}',0);"
    )
    con.commit()
    con.close()
    store = EventStore(path)
    store.insert_event(_event("new", "fire"))
    from minerva.store.query import QueryFilter

    assert [r["category"] for r in store.query(QueryFilter(categories=["fire"]))] == ["fire"]
    old = store.query(QueryFilter(text="shove"))  # the FTS index still finds the old row
    assert [r["event_id"] for r in old] == [7]
    assert "CHECK (category" not in store._conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='events'").fetchone()[0]


def test_categories_endpoint(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from minerva.chat.answer import ChatService
    from minerva.web import server as web
    from minerva.web.jobs import JobManager

    monkeypatch.setattr(JobManager, "submit", lambda self, upload_id: None)
    s = Settings(data_dir=tmp_path)
    app = web.create_app(s, ChatService(EventStore(s.db_path), QuestionParser(None)), [])
    cats = TestClient(app).get("/api/categories").json()
    assert {"name": "person_down", "label": "person down",
            "definition": DEFAULT_CATEGORIES["person_down"]} in cats
