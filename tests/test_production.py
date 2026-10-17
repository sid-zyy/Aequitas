"""Production features: the tamper-evident audit chain, evidence packages, and the health page."""

import hashlib
import io
import json
import secrets
import time
import zipfile

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from aequitas.auth import Auth
from aequitas.chat.answer import ChatService
from aequitas.chat.parser import QuestionParser
from aequitas.config import CameraCfg, Settings
from aequitas.evidence import build_package
from aequitas.reports import ReportStore, file_sha256
from aequitas.store.db import EventRecord, EventStore
from aequitas.web import health as health_mod
from aequitas.web import server as web
from aequitas.web.jobs import JobManager

PW = "test-" + secrets.token_hex(8)  # generated per run; test-only


def _store(tmp_path):
    return EventStore(tmp_path / "data" / "a.db")


def _sql(store, sql, params=()):
    with store._lock, store._conn:
        store._conn.execute(sql, params)


# ---------- audit chain ----------

def test_audit_chain_verifies_and_key_is_kept_outside_the_database(tmp_path):
    store = _store(tmp_path)
    auth = Auth(store)
    for i in range(4):
        auth.audit(None, "report_viewed", f"R{i}", {"i": i}, "10.0.0.1")
    res = auth.verify_chain()
    assert res["ok"] and res["checked"] == 4 and len(res["head"]) == 64
    assert (tmp_path / "data" / "audit.key").is_file()
    assert Auth(store).verify_chain()["ok"]  # a restart reuses the same key


def test_audit_chain_catches_an_edited_entry(tmp_path):
    store = _store(tmp_path)
    auth = Auth(store)
    for i in range(4):
        auth.audit(None, "report_updated", "R1", {"status": "under_review" if i else "received"})
    _sql(store, "UPDATE audit_log SET detail=? WHERE id=2", (json.dumps({"status": "closed"}),))
    res = auth.verify_chain()
    assert not res["ok"] and res["broken_at"] == 2 and "changed" in res["reason"]


def test_audit_chain_catches_a_deleted_entry(tmp_path):
    store = _store(tmp_path)
    auth = Auth(store)
    for i in range(4):
        auth.audit(None, "login", f"user{i}")
    _sql(store, "DELETE FROM audit_log WHERE id=2")
    res = auth.verify_chain()
    assert not res["ok"] and res["broken_at"] == 3


def test_audit_chain_needs_the_key(tmp_path):
    store = _store(tmp_path)
    Auth(store).audit(None, "login", "a")
    (tmp_path / "data" / "audit.key").write_bytes(secrets.token_bytes(32))  # someone swaps the key
    assert not Auth(store).verify_chain()["ok"]


def test_entries_from_before_the_chain_are_sealed_on_first_start(tmp_path):
    store = _store(tmp_path)
    for i in range(3):
        _sql(store, "INSERT INTO audit_log(user_id, action, target, detail, ip, created_at) VALUES (?,?,?,?,?,?)",
             (None, "legacy", f"t{i}", None, "", 1000 + i))
    auth = Auth(store)
    res = auth.verify_chain()
    assert res["ok"] and res["checked"] == 3
    auth.audit(None, "login", "x")
    assert auth.verify_chain()["checked"] == 4


# ---------- web: audit page is admin-only ----------

@pytest.fixture
def signed_in(tmp_path, monkeypatch):
    monkeypatch.setattr(JobManager, "submit", lambda self, upload_id: None)
    s = Settings(data_dir=tmp_path / "data", authenticity={"frame_model": None},
                 live={"localities_file": tmp_path / "none.yaml"})
    store = EventStore(s.db_path)
    a = Auth(store)
    a.create_user("commissioner", PW, "admin")
    a.create_user("constable", PW, "officer")
    c = TestClient(web.create_app(s, ChatService(store, QuestionParser(None)), []))
    c.store = store
    return c


def test_audit_log_api_is_admin_only_and_reports_the_chain(signed_in):
    c = signed_in
    assert c.post("/api/auth/login", json={"username": "constable", "password": PW}).status_code == 200
    assert c.get("/api/audit").status_code == 403
    assert c.get("/api/audit/verify").status_code == 403
    c.post("/api/auth/logout")
    assert c.post("/api/auth/login", json={"username": "commissioner", "password": PW}).status_code == 200
    log = c.get("/api/audit").json()
    actions = [e["action"] for e in log["entries"]]
    assert actions[0] == "login" and "user_created" in actions and "login" in log["actions"]
    assert log["entries"][0]["username"] == "commissioner"
    v = c.get("/api/audit/verify").json()
    assert v["ok"] and v["checked"] == len(actions)
    only = c.get("/api/audit", params={"action": "user_created"}).json()["entries"]
    assert {e["action"] for e in only} == {"user_created"} and len(only) == 2


# ---------- evidence package ----------

def _video(path):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(20):
        vw.write(np.full((48, 64, 3), i * 9 % 255, np.uint8))
    vw.release()
    return path


def _citizen_incident(tmp_path, store):
    vid = _video(tmp_path / "u1.mp4")
    kf = tmp_path / "k.jpg"
    cv2.imwrite(str(kf), np.zeros((20, 20, 3), np.uint8))
    now = int(time.time() * 1000)
    store.upsert_camera(CameraCfg(camera_id="report-u1", name="Citizen report", source="citizen"))
    store.insert_upload({"upload_id": "u1", "camera_id": "report-u1", "filename": "phone.mp4", "path": vid.as_posix(),
                         "start_ts": now - 10_000, "duration_ms": 2000, "mode": "periodic", "vlm_model": "m",
                         "status": "done", "frames_total": 8, "created_at": now})
    rid = ReportStore(store).create("u1", "report-u1", "A man pushed another man to the ground.", file_sha256(vid))
    eid = store.insert_event(EventRecord("ev1", "report-u1", now - 9_000, now - 8_000, "fighting", None, "confirmed",
                                         0.9, 0.97, "fighting", "Two people grapple.", None, [], kf.as_posix(),
                                         {"vlm": "ollama:gemma4:e2b", "violence": "r3d18"}))
    store.review_event(eid, "confirmed", "constable")
    return eid, rid, vid


def test_evidence_package_is_self_checking(tmp_path):
    store = _store(tmp_path)
    auth = Auth(store)
    eid, rid, vid = _citizen_incident(tmp_path, store)
    auth.audit(None, "report_event_reviewed", f"{rid}/{eid}", {"decision": "confirmed"})
    data, m = build_package(store, auth, eid, "commissioner")
    z = zipfile.ZipFile(io.BytesIO(data))
    names = set(z.namelist())
    assert {"keyframe.jpg", "video/u1.mp4", "manifest.json", "README.txt", "SHA256SUMS"} <= names
    for line in z.read("SHA256SUMS").decode().splitlines():  # every listed hash matches its file
        digest, name = line.split("  ", 1)
        assert hashlib.sha256(z.read(name)).hexdigest() == digest
    assert z.read("video/u1.mp4") == vid.read_bytes()  # byte for byte, not re-encoded
    man = json.loads(z.read("manifest.json"))
    assert man["source"]["unchanged_since_arrival"] is True
    assert man["human_review"]["decision"] == "confirmed" and man["incident"]["ai_status"] == "confirmed"
    assert man["citizen_report"]["report_id"] == rid
    assert [e["action"] for e in man["audit"]["entries"]] == ["report_event_reviewed"]
    assert man["audit"]["chain_at_export"]["ok"]
    readme = z.read("README.txt").decode()
    assert "Matches the hash taken when the citizen uploaded it" in readme and "CONFIRMED by constable" in readme


def test_evidence_package_flags_a_changed_video_and_skips_huge_ones(tmp_path):
    store = _store(tmp_path)
    auth = Auth(store)
    eid, _rid, vid = _citizen_incident(tmp_path, store)
    with open(vid, "ab") as f:
        f.write(b"tampered")
    _data, m = build_package(store, auth, eid, None)
    assert m["source"]["unchanged_since_arrival"] is False
    _data, m = build_package(store, auth, eid, None, max_video_mb=0)
    assert not m["source"]["included"] and "video/u1.mp4" not in m["files"]


def test_evidence_endpoint_audits_the_export(signed_in):
    c = signed_in
    eid, _rid, _vid = _citizen_incident(c.store.path.parent.parent, c.store)
    assert c.get(f"/api/events/{eid}/evidence").status_code == 401
    c.post("/api/auth/login", json={"username": "constable", "password": PW})
    r = c.get(f"/api/events/{eid}/evidence")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert r.headers["x-evidence-sha256"] == hashlib.sha256(r.content).hexdigest()
    rows = c.app.state.auth.audit_trail(f"event/{eid}")
    assert rows[-1]["action"] == "evidence_exported"
    assert json.loads(rows[-1]["detail"])["package_sha256"] == r.headers["x-evidence-sha256"]
    assert c.get("/api/events/999/evidence").status_code == 404
    second = zipfile.ZipFile(io.BytesIO(c.get(f"/api/events/{eid}/evidence").content))
    assert "evidence_exported" in [e["action"] for e in json.loads(second.read("manifest.json"))["audit"]["entries"]]


# ---------- health ----------

class _DeadClient:
    def ping(self):
        return False


def test_health_reports_each_part(signed_in, monkeypatch):
    c = signed_in
    monkeypatch.setattr("aequitas.understand.vlm_client.make_client", lambda cfg: _DeadClient())
    monkeypatch.setattr(health_mod, "gpu_info", lambda: None)
    c.post("/api/auth/login", json={"username": "constable", "password": PW})
    h = c.get("/api/health").json()
    by = {x["name"]: x for x in h["checks"]}
    assert by["AI model server"]["status"] == "down" and h["status"] == "down"
    assert by["GPU"]["status"] == "warn"
    assert by["Audit log"]["status"] == "ok" and h["audit"]["ok"]
    assert {"Live analysis", "Video queue", "Storage"} <= set(by)
    now = int(time.time() * 1000)
    c.store.record_window("cam1", now - 4000, now, "People walk.", "none", 0.9, 600, 300, 2500)
    c.store.record_window("cam1", now - 4000, now, None, None, None, None, None, None)
    p = c.get("/api/health").json()["pipeline"]
    assert p["windows"] == 2 and p["failed"] == 1 and p["summarize_ms_p50"] == 600
