"""Citizen portal in Hindi/Kannada, officer-side translation (fake LLM), and the rule-based emergency assistant."""

import re
import secrets

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from aequitas import emergency
from aequitas.auth import Auth
from aequitas.chat.answer import ChatService
from aequitas.chat.parser import QuestionParser
from aequitas.config import LlmEndpoint, Settings
from aequitas.store.db import EventStore
from aequitas.translate import Translation, detect_language, translate_to_english
from aequitas.web import server as web
from aequitas.web.jobs import JobManager

PW = "test-" + secrets.token_hex(8)  # generated per run; test-only
KANNADA = "ಬಸ್ ನಿಲ್ದಾಣದಲ್ಲಿ ಒಬ್ಬ ವ್ಯಕ್ತಿ ಮಹಿಳೆಯನ್ನು ಹಿಂಬಾಲಿಸಿದನು"
HINDI = "बस स्टॉप पर एक आदमी ने महिला का पीछा किया और हाथ पकड़ा"


class FakeLlm:
    """Stands in for the local LLM client: records prompts and returns a fixed translation."""

    def __init__(self, english="A man followed a woman at the bus stop."):
        self.cfg = LlmEndpoint()
        self.model = self.cfg.model
        self.english, self.prompts = english, []

    def structured(self, messages, model_cls, retries=1, max_tokens=None):
        self.prompts.append(messages[0]["content"])
        return None if self.english is None else model_cls(english=self.english)


@pytest.fixture
def c(tmp_path, monkeypatch):
    monkeypatch.setattr(JobManager, "submit", lambda self, upload_id: None)
    s = Settings(data_dir=tmp_path / "data", authenticity={"frame_model": None})
    store = EventStore(s.db_path)
    Auth(store).create_user("commissioner", PW, "admin")
    client = TestClient(web.create_app(s, ChatService(store, QuestionParser(None)), []))
    client.store, client.tmp = store, tmp_path
    client.app.state.translator = FakeLlm()
    return client


def _report(c, description, name="a.mp4", **form):
    path = c.tmp / name
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(10):
        vw.write(np.full((48, 64, 3), (i * 9 + len(name) * 31) % 255, np.uint8))
    vw.release()
    with open(path, "rb") as f:
        r = c.post("/api/citizen/reports", files={"file": (name, f, "video/mp4")},
                   data={"description": description, "consent": "true", **form})
    assert r.status_code == 200, r.text
    return r.json()["report_id"]


def _login(c):
    assert c.post("/api/auth/login", json={"username": "commissioner", "password": PW}).status_code == 200


# ---------- language ----------

def test_script_decides_the_language():
    assert detect_language(KANNADA) == "kn"
    assert detect_language(HINDI, "en") == "hi"
    assert detect_language("Mujhe madad chahiye, ek aadmi peecha kar raha hai", "hi") == "hi"  # Latin Hindi
    assert detect_language("A man followed me", "fr") == "en"


def test_kannada_report_is_stored_as_written_with_its_language(c):
    rid = _report(c, KANNADA, language="en")  # UI left on English, but the text is Kannada
    _login(c)
    r = c.get(f"/api/reports/{rid}").json()
    assert r["description"] == KANNADA and r["language"] == "kn" and r["urgent"] is False
    assert r["translation"] is None


def test_translate_shows_english_next_to_the_original_and_is_cached(c):
    rid = _report(c, HINDI, language="hi")
    assert c.post(f"/api/reports/{rid}/translate").status_code == 401  # officers only
    _login(c)
    t = c.post(f"/api/reports/{rid}/translate").json()
    assert t["text"] == "A man followed a woman at the bus stop." and t["from"] == "hi" and t["cached"] is False
    assert isinstance(t["ms"], int) and t["model"] == "ollama:gemma4:e2b"
    fake = c.app.state.translator
    assert HINDI in fake.prompts[0] and "Hindi" in fake.prompts[0]
    again = c.post(f"/api/reports/{rid}/translate").json()
    assert again["cached"] is True and len(fake.prompts) == 1  # no second model call
    r = c.get(f"/api/reports/{rid}").json()
    assert r["description"] == HINDI and r["translation"]["text"] == t["text"]  # original kept
    assert any(a["action"] == "report_translated" for a in r["audit"])


def test_translation_failure_is_reported_not_stored(c):
    rid = _report(c, KANNADA)
    _login(c)
    c.app.state.translator = FakeLlm(english=None)
    r = c.post(f"/api/reports/{rid}/translate")
    assert r.status_code == 502 and "did not answer" in r.json()["detail"]
    assert c.get(f"/api/reports/{rid}").json()["translation"] is None
    assert c.post("/api/reports/RNOPE/translate").status_code == 404


def test_translate_to_english_times_the_call():
    english, ms = translate_to_english(FakeLlm(english="  Hello.  "), KANNADA, "kn")
    assert english == "Hello." and ms >= 0
    assert Translation(english="x").english == "x"


# ---------- urgent flag ----------

def test_urgent_reports_go_to_the_top_of_the_queue(c):
    first = _report(c, "Someone is selling stolen phones near the market.", name="a.mp4")
    urgent = _report(c, "A man is following me with a knife right now.", name="b.mp4", urgent="true")
    _login(c)
    q = c.get("/api/reports/active").json()["reports"]
    assert [r["report_id"] for r in q] == [urgent, first] and q[0]["urgent"] is True and q[1]["urgent"] is False


# ---------- emergency assistant ----------

ALLOWED = {"112", "100", "101", "108", "1091"}


@pytest.mark.parametrize("text,want", [
    ("yes", "yes"), ("Haan", "yes"), ("हाँ", "yes"), ("ಹೌದು", "yes"),
    ("no", "no"), ("नहीं", "no"), ("ಇಲ್ಲ", "no"), ("no, I just want to report something", "no"),
    ("bachao! someone is attacking", "yes"), ("आग लगी है", "yes"), ("ಬೆಂಕಿ ಬಿದ್ದಿದೆ", "yes"),
    ("what is this site", "unclear"), ("", "unclear"),
])
def test_intent(text, want):
    assert emergency.intent(text) == want


@pytest.mark.parametrize("lang", ["en", "hi", "kn"])
def test_assistant_only_ever_shows_the_official_numbers(lang):
    for msg, choice in [(None, None), ("help", None), (None, "yes"), (None, "no"), ("??", None)]:
        r = emergency.reply(lang, msg, choice)
        numbers = set(re.findall(r"\d{3,}", r["reply"])) | {h["number"] for h in r["helplines"]}
        assert numbers <= ALLOWED
        assert not re.search(r"dispatched|on (the|their) way|sent help|भेज दी|ಕಳುಹಿಸಲಾಗಿದೆ", r["reply"], re.I)
    em = emergency.reply(lang, choice="yes")
    assert em["state"] == "emergency" and em["urgent"] is True
    assert [h["number"] for h in em["helplines"]] == ["112", "100", "101", "108", "1091"]


def test_assistant_speaks_the_selected_language(c):
    hi = c.post("/api/citizen/assist", json={"lang": "hi"}).json()
    assert hi["state"] == "ask" and re.search(r"[ऀ-ॿ]", hi["reply"])
    assert [b["label"] for b in hi["choices"]] == ["हाँ", "नहीं"]
    kn = c.post("/api/citizen/assist", json={"lang": "kn", "choice": "yes"}).json()  # public: no sign-in
    assert kn["state"] == "emergency" and re.search(r"[ಀ-೿]", kn["reply"]) and kn["urgent"]
    assert c.post("/api/citizen/assist", json={"lang": "en", "message": "no"}).json()["state"] == "report"
    assert c.post("/api/citizen/assist", json={"lang": "xx"}).status_code == 422
