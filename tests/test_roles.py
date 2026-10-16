"""Two roles: police (sign-in) and citizens (public report portal), plus the AI-likelihood check."""

import os
import secrets
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from aequitas import authenticity as auth_mod
from aequitas.auth import Auth, LockedOut, hash_password, verify_password
from aequitas.chat.answer import ChatService
from aequitas.chat.parser import QuestionParser
from aequitas.config import Settings
from aequitas.store.db import EventStore
from aequitas.web import server as web
from aequitas.web.jobs import JobManager

PW = "test-" + secrets.token_hex(8)  # generated per run; test-only


def _video(path, seconds=1, fps=10, extra: bytes = b""):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(seconds * fps):
        vw.write(np.full((48, 64, 3), i * 9 % 255, np.uint8))
    vw.release()
    if extra:
        with open(path, "ab") as f:
            f.write(extra)
    return path


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    monkeypatch.setattr(JobManager, "submit", lambda self, upload_id: None)
    s = Settings(data_dir=tmp_path / "data", authenticity={"frame_model": None})
    store = EventStore(s.db_path)
    Auth(store).create_user("commissioner", PW, "admin")
    c = TestClient(web.create_app(s, ChatService(store, QuestionParser(None)), []))
    c.store, c.tmp = store, tmp_path
    return c


def _login(c):
    r = c.post("/api/auth/login", json={"username": "commissioner", "password": PW})
    assert r.status_code == 200
    return r


def _report(c, tmp_path, name="a.mp4", extra=b"", **form):
    data = {"description": "A man kept following a woman and grabbed her arm.", "consent": "true", **form}
    with open(_video(tmp_path / name, extra=extra), "rb") as f:
        return c.post("/api/citizen/reports", files={"file": (name, f, "video/mp4")}, data=data)


# ---------- passwords and sessions ----------

def test_passwords_are_salted_and_verified():
    h1, h2 = hash_password(PW), hash_password(PW)
    assert h1 != h2 and h1.startswith("scrypt$") and PW not in h1
    assert verify_password(PW, h1) and not verify_password(PW + "x", h1) and not verify_password(PW, "garbage")


def test_accounts_need_strong_passwords_and_known_roles(tmp_path):
    a = Auth(EventStore(tmp_path / "t.db"))
    with pytest.raises(ValueError):
        a.create_user("x", "short", "admin")
    with pytest.raises(ValueError):
        a.create_user("x", PW, "superuser")
    a.create_user("x", PW, "officer")
    with pytest.raises(Exception):
        a.create_user("X", PW, "officer")  # usernames are case-insensitive and unique


def test_lockout_after_repeated_failures(tmp_path):
    a = Auth(EventStore(tmp_path / "t.db"), max_failures=3, lockout_s=60)
    a.create_user("officer1", PW)
    for _ in range(3):
        assert a.login("officer1", "wrong-password", "1.2.3.4") is None
    with pytest.raises(LockedOut):
        a.login("officer1", PW, "1.2.3.4")
    assert a.login("officer1", PW, "5.6.7.8") is not None  # other addresses are not blocked


def test_police_api_requires_sign_in_and_citizen_api_does_not(app_client):
    c = app_client
    assert c.get("/api/events").status_code == 401
    assert c.get("/api/reports").status_code == 401
    assert c.post("/api/ask", json={"question": "fights today?"}).status_code == 401
    assert c.get("/api/citizen/reports/RXXXX?code=abcd").status_code == 404  # reachable, just unknown
    r = c.post("/api/auth/login", json={"username": "commissioner", "password": "wrong-password"})
    assert r.status_code == 401
    r = _login(c)
    assert r.json() == {"user_id": 1, "username": "commissioner", "role": "admin"}
    assert "httponly" in r.headers["set-cookie"].lower() and "samesite=strict" in r.headers["set-cookie"].lower()
    assert c.get("/api/events").status_code == 200 and c.get("/api/auth/me").json()["role"] == "admin"
    c.post("/api/auth/logout")
    assert c.get("/api/events").status_code == 401


def test_session_tokens_are_not_stored_in_plain_text(app_client):
    _login(app_client)
    token = app_client.cookies.get("aeq_session")
    rows = app_client.store._conn.execute("SELECT token_hash FROM sessions").fetchall()
    assert token and all(token not in r[0] for r in rows)


# ---------- citizen reports ----------

def test_citizen_report_status_by_report_id(app_client):
    c = app_client
    r = _report(c, c.tmp, location="MG Road bus stop", incident_at="2026-10-15T18:30")
    assert r.status_code == 200
    rid = r.json()["report_id"]
    assert "tracking_code" not in r.json()
    st = c.get(f"/api/citizen/reports/{rid}").json()
    assert st["status"] == "Received" and "description" not in st and "notes" not in st and "contact" not in st
    assert c.get(f"/api/citizen/reports/{rid.lower()}").json()["report_id"] == rid  # typed in lower case
    assert c.get("/api/citizen/reports/RNOTREAL").status_code == 404
    row = c.store._conn.execute("SELECT * FROM reports").fetchone()
    assert len(row["sha256"]) == 64 and row["status"] == "received"


def test_citizen_input_is_validated(app_client, tmp_path):
    c = app_client
    assert _report(c, tmp_path, consent="false").status_code == 400
    assert _report(c, tmp_path, description="short").status_code == 422
    bad = tmp_path / "notes.txt"
    bad.write_text("not a video")
    with open(bad, "rb") as f:
        r = c.post("/api/citizen/reports", files={"file": ("notes.txt", f)},
                   data={"description": "Something happened near the gate.", "consent": "true"})
    assert r.status_code == 400
    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"not really a video")
    with open(fake, "rb") as f:
        r = c.post("/api/citizen/reports", files={"file": ("fake.mp4", f)},
                   data={"description": "Something happened near the gate.", "consent": "true"})
    assert r.status_code == 400


def test_citizen_reports_are_rate_limited(app_client, tmp_path):
    codes = [_report(app_client, tmp_path, name=f"v{i}.mp4").status_code for i in range(6)]
    assert codes[:5] == [200] * 5 and codes[5] == 429


def test_identical_file_is_linked_as_duplicate(app_client, tmp_path):
    first = _report(app_client, tmp_path).json()["report_id"]
    _report(app_client, tmp_path)  # same bytes again
    rows = app_client.store._conn.execute("SELECT report_id, duplicate_of FROM reports ORDER BY created_at").fetchall()
    assert rows[0]["duplicate_of"] is None and rows[1]["duplicate_of"] == first


def test_police_review_status_notes_and_audit(app_client):
    c = app_client
    rid = _report(c, c.tmp).json()["report_id"]
    _login(c)
    listed = c.get("/api/reports").json()
    assert listed[0]["report_id"] == rid and listed[0]["video_url"].startswith("/api/uploads/")
    assert "tracking_hash" not in listed[0]
    r = c.patch(f"/api/reports/{rid}", json={"status": "under_review", "notes": "Contacted the complainant."})
    assert r.status_code == 200 and r.json()["status"] == "under_review"
    assert c.patch(f"/api/reports/{rid}", json={"status": "deleted"}).status_code == 422
    detail = c.get(f"/api/reports/{rid}").json()
    actions = [a["action"] for a in detail["audit"]]
    assert actions[:2] == ["report_submitted", "report_updated"] and "report_viewed" in actions
    assert detail["audit"][1]["username"] == "commissioner"
    # the citizen sees the new status but never the police notes
    assert "notes" not in c.get(f"/api/citizen/reports/{rid}", params={"code": "x" * 8}).text


# ---------- AI-likelihood ----------

def test_metadata_markers_c2pa_and_camera(tmp_path):
    m = auth_mod.metadata_signals(b"\x00\x01moov....udta\x00\x00\x00encoder: Google Veo 3 by DeepMind\x00\x00")
    assert m["generators"] == ["Google Veo", "Veo 3"]
    m = auth_mod.metadata_signals(b"c2pa....digitalSourceType http://cv.iptc.org/trainedAlgorithmicMedia..")
    assert m["c2pa_present"] and m["ai_source_type"] == ["trainedAlgorithmicMedia"]
    m = auth_mod.metadata_signals(b"....com.apple.quicktime.make\x00Apple\x00com.apple.quicktime.model iPhone")
    assert m["camera_metadata"] and not m["generators"]


def test_random_video_bytes_do_not_trigger_generator_markers():
    rng = np.random.default_rng(0)
    for _ in range(3):
        m = auth_mod.metadata_signals(rng.integers(0, 256, 8 * 1024**2, dtype=np.uint8).tobytes())
        assert m["generators"] == [] and m["ai_source_type"] == []


def test_combining_signals_into_a_likelihood():
    cfg = auth_mod.AuthenticityCfg()
    none = {"ai_source_type": [], "generators": [], "camera_metadata": []}
    assert auth_mod.combine({**none, "ai_source_type": ["trainedAlgorithmicMedia"]}, None, 0.1, cfg)[::2] == (0.99, "provenance")
    assert auth_mod.combine({**none, "generators": ["HeyGen"]}, None, 0.2, cfg)[::2] == (0.9, "metadata")
    assert auth_mod.combine({**none, "camera_metadata": ["(c)mak"]}, None, 0.99, cfg)[::2] == (0.1, "camera")
    p, reasons, basis = auth_mod.combine(none, None, None, cfg)
    assert p is None and basis == "none" and "not enough signal" in reasons[0]


def test_frame_detector_alone_never_gives_a_percentage():
    # real stock video scores 100 % on the image detector: a frame-only result is a check, not a likelihood
    none = {"ai_source_type": [], "generators": [], "camera_metadata": []}
    p, reasons, basis = auth_mod.combine(none, None, 1.0, auth_mod.AuthenticityCfg(frame_flag_at=None))
    assert p is None and basis == "frames" and reasons[0] == "Frame check: inconclusive"
    assert not any("%" in r for r in reasons)  # no raw score in what the officer reads
    cfg = auth_mod.AuthenticityCfg(frame_flag_at=0.9, frame_flag_note="flags 5 % of real test clips")
    assert auth_mod.combine(none, None, 0.95, cfg)[1][0] == ("Frame check: AI-like frames, needs a human look "
                                                              "(flags 5 % of real test clips)")
    assert auth_mod.combine(none, None, 0.5, cfg)[1][0].startswith("Frame check: inconclusive")


def test_frame_detector_leaves_the_gpu_to_the_vlm_when_memory_is_short(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (int(0.8 * 1024**3), 6 * 1024**3))
    assert auth_mod._device(1.5) == -1
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (2 * 1024**3, 6 * 1024**3))
    assert auth_mod._device(1.5) == 0


def test_classifier_labels_are_mapped_whatever_the_model_calls_them():
    assert auth_mod.ai_probability([{"label": "artificial", "score": 0.8}, {"label": "human", "score": 0.2}]) == 0.8
    assert auth_mod.ai_probability([{"label": "FAKE", "score": 0.3}, {"label": "REAL", "score": 0.7}]) == 0.3
    assert auth_mod.ai_probability([{"label": "real", "score": 0.9}]) == pytest.approx(0.1)


def test_analyze_video_end_to_end_with_a_fake_frame_detector(tmp_path, monkeypatch):
    monkeypatch.setattr(auth_mod, "frame_classifier",
                        lambda model, *_: (lambda img: [{"label": "artificial", "score": 0.92},
                                                    {"label": "human", "score": 0.08}]))
    r = auth_mod.analyze_video(_video(tmp_path / "ai.mp4"),
                               auth_mod.AuthenticityCfg(frame_model="fake/model", frame_flag_at=None))
    # the frame detector alone gives no percentage (image detectors score real compressed video 100 % too)
    assert r["ai_likelihood_pct"] is None and r["verdict"] == "inconclusive" and r["basis"] == "frames"
    assert r["frame_score"] == 0.92 and len(r["frames"]["per_frame"]) == 8
    assert r["headline"] == "Frame check: inconclusive"
    flagged = auth_mod.analyze_video(_video(tmp_path / "ai2.mp4"), auth_mod.AuthenticityCfg(frame_model="fake/model"))
    assert flagged["verdict"] == "uncertain" and flagged["ai_likelihood_pct"] is None  # default flag at 0.85
    assert flagged["headline"].startswith("Frame check: AI-like frames")
    tagged = _video(tmp_path / "tagged.mp4", extra=b"\x00\x00\x00 udta: made with HeyGen avatar\x00")
    r = auth_mod.analyze_video(tagged, auth_mod.AuthenticityCfg(frame_model=None))
    assert r["ai_likelihood_pct"] == 90 and "HeyGen" in r["reasons"][0]
    assert "not proof" in r["disclaimer"]


def test_report_gets_its_authenticity_result(app_client, tmp_path):
    rid = _report(app_client, tmp_path, extra=b"\x00 encoder RunwayML Gen-3 \x00").json()["report_id"]
    for _ in range(50):  # runs in a background thread
        row = app_client.store._conn.execute("SELECT authenticity FROM reports WHERE report_id=?", (rid,)).fetchone()
        if row[0]:
            break
        time.sleep(0.05)
    _login(app_client)
    a = app_client.get(f"/api/reports/{rid}").json()["authenticity"]
    assert a["verdict"] == "likely_ai" and a["ai_likelihood_pct"] == 90


def test_weak_demo_password_only_with_explicit_opt_in(tmp_path):
    a = Auth(EventStore(tmp_path / "t.db"))
    with pytest.raises(ValueError):
        a.create_user("admin", "admin", "admin")
    a.create_user("admin", "admin", "admin", allow_weak=True)
    assert a.login("admin", "admin") is not None
    detail = a.store._conn.execute("SELECT detail FROM audit_log WHERE action='user_created'").fetchone()[0]
    assert '"weak_password": true' in detail  # flagged in the audit trail


def test_active_requests_queue_is_open_reports_oldest_first(app_client, tmp_path):
    c = app_client
    first = _report(c, tmp_path, name="a.mp4").json()["report_id"]
    second = _report(c, tmp_path, name="b.mp4", extra=b"\x00different bytes\x00").json()["report_id"]
    third = _report(c, tmp_path, name="c.mp4", extra=b"\x00other bytes\x00").json()["report_id"]
    assert c.get("/api/reports/active").status_code == 401  # police only
    _login(c)
    c.patch(f"/api/reports/{second}", json={"status": "under_review"})
    c.patch(f"/api/reports/{third}", json={"status": "closed"})
    q = c.get("/api/reports/active").json()
    assert [r["report_id"] for r in q["reports"]] == [first, second]  # closed ones leave the queue
    assert q["counts"] == {"received": 1, "under_review": 1, "action_taken": 0}
    assert q["reports"][0]["categories"] == [] and "tracking_hash" not in q["reports"][0]
    from aequitas.store.db import EventRecord  # the AI found something in the first video
    cam = c.store._conn.execute("SELECT camera_id FROM reports WHERE report_id=?", (first,)).fetchone()[0]
    c.store.insert_event(EventRecord("e1", cam, 0, 4000, "fighting", None, "unverified", 0.8, 0.0,
                                     "fighting", "two people trade punches", None, [], "k.jpg", {}))
    top = c.get("/api/reports/active").json()["reports"][0]
    assert top["categories"] == ["fighting"] and top["findings"][0]["summary"] == "two people trade punches"


def test_unfinished_authenticity_checks_resume_after_a_restart(app_client, tmp_path, monkeypatch):
    rid = _report(app_client, tmp_path).json()["report_id"]
    for _ in range(50):  # let the first check finish, then pretend the server died mid-check
        if app_client.store._conn.execute("SELECT authenticity FROM reports").fetchone()[0]:
            break
        time.sleep(0.05)
    app_client.store._conn.execute("UPDATE reports SET authenticity=NULL")
    app_client.store._conn.commit()
    s = Settings(data_dir=app_client.tmp / "data", authenticity={"frame_model": None})
    web.create_app(s, ChatService(app_client.store, QuestionParser(None)), [])  # "restart"
    for _ in range(100):
        row = app_client.store._conn.execute("SELECT authenticity FROM reports WHERE report_id=?", (rid,)).fetchone()
        if row[0]:
            break
        time.sleep(0.05)
    assert row[0] and "verdict" in row[0]


def test_fresh_install_gets_a_demo_admin_only_when_there_are_no_accounts(tmp_path):
    s = Settings(data_dir=tmp_path / "fresh", authenticity={"frame_model": None})
    c = TestClient(web.create_app(s, ChatService(EventStore(s.db_path), QuestionParser(None)), []))
    assert c.post("/api/auth/login", json={"username": "admin", "password": "admin"}).status_code == 200
    s2 = Settings(data_dir=tmp_path / "existing", authenticity={"frame_model": None})
    store = EventStore(s2.db_path)
    Auth(store).create_user("commissioner", PW, "admin")
    c2 = TestClient(web.create_app(s2, ChatService(store, QuestionParser(None)), []))
    assert c2.post("/api/auth/login", json={"username": "admin", "password": "admin"}).status_code == 401
