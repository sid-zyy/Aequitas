"""eval-chat: scoring of parsed filters against expected ones (keyword parser, no model)."""

from aequitas.chat.parser import QuestionParser
from aequitas.eval import chat_eval
from aequitas.store.query import QueryFilter


def test_question_set_loads_and_keyword_baseline_runs():
    items, cams, now = chat_eval.load("config/chat_eval.yaml")
    assert len(items) >= 25 and now.tzinfo is not None
    r = chat_eval.run(QuestionParser(None), items, cams, now)
    assert r["n"] == len(items) and r["llm_answered"] == 0
    assert 0 < r["all_correct"] <= r["n"]
    by_q = {row["q"]: row for row in r["rows"]}
    assert by_q["Show all fights on MG Road today"]["all"]  # the common phrasing must work without a model
    assert "| keywords only |" in chat_eval.report_table({"keywords only": r})


def test_score_fields_and_time_tolerance():
    items, cams, now = chat_eval.load("config/chat_eval.yaml")
    item = {"q": "x", "categories": ["fire"], "cameras": ["ksr"], "start": "2026-10-16T06:00",
            "end": "2026-10-16T12:00", "intent": "count"}
    tz = now.tzinfo
    from datetime import datetime
    f = QueryFilter(categories=["fire"], camera_ids=["ksr"], intent="count",
                    start=datetime(2026, 10, 16, 6, 0, tzinfo=tz), end=datetime(2026, 10, 16, 12, 1, tzinfo=tz))
    assert all(chat_eval.score(item, f, now).values())  # 1 minute off is fine
    wrong = f.model_copy(update={"camera_ids": [], "end": datetime(2026, 10, 16, 13, 0, tzinfo=tz)})
    s = chat_eval.score(item, wrong, now)
    assert not s["cameras"] and not s["time"] and s["categories"] and s["intent"]
