"""Citizen reports: a video plus a description, analysed by the normal pipeline and reviewed by the police.

Citizens have no account: they get a report ID and a one-time tracking code (only its hash is kept).
Every file is fingerprinted (sha256) on arrival so later tampering is detectable, and an identical file
uploaded again is linked to the first report instead of creating unrelated work."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time

STATUSES = ("received", "under_review", "action_taken", "closed", "rejected")
PUBLIC_STATUS = {  # what the citizen sees
    "received": "Received", "under_review": "Under review by police", "action_taken": "Action taken",
    "closed": "Closed", "rejected": "Closed without action",
}


def _hash(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


class ReportStore:
    def __init__(self, store):
        self.store = store

    def _q(self, sql: str, params: tuple = (), commit: bool = False):
        with self.store._lock:
            if commit:
                with self.store._conn:
                    return self.store._conn.execute(sql, params)
            return self.store._conn.execute(sql, params).fetchall()

    def create(self, upload_id: str, camera_id: str, description: str, sha256: str, location: str | None = None,
               incident_at: int | None = None, contact: str | None = None, language: str = "en",
               urgent: bool = False) -> tuple[str, str]:
        """Returns (report_id, tracking_code). The code is shown to the citizen once and not stored."""
        report_id = "R" + secrets.token_hex(4).upper()
        code = secrets.token_urlsafe(9)
        dup = self._q("SELECT report_id FROM reports WHERE sha256=? ORDER BY created_at LIMIT 1", (sha256,))
        now = int(time.time() * 1000)
        self._q("INSERT INTO reports(report_id, tracking_hash, upload_id, camera_id, description, location, "
                "incident_at, contact, sha256, duplicate_of, status, created_at, updated_at, language, urgent) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (report_id, _hash(code), upload_id, camera_id, description, location, incident_at, contact, sha256,
                 dup[0]["report_id"] if dup else None, "received", now, now, language, int(urgent)), commit=True)
        return report_id, code

    def get(self, report_id: str) -> dict | None:
        rows = self._q("SELECT * FROM reports WHERE report_id=?", (report_id,))
        return self._row(rows[0]) if rows else None

    def list(self, status: str | None = None, limit: int = 100) -> list[dict]:
        if status:
            rows = self._q("SELECT * FROM reports WHERE status=? ORDER BY created_at DESC LIMIT ?", (status, limit))
        else:
            rows = self._q("SELECT * FROM reports ORDER BY created_at DESC LIMIT ?", (limit,))
        return [self._row(r) for r in rows]

    def check_code(self, report_id: str, code: str) -> dict | None:
        rows = self._q("SELECT * FROM reports WHERE report_id=?", (report_id,))
        if not rows or not hmac.compare_digest(rows[0]["tracking_hash"], _hash(code or "")):
            return None
        return self._row(rows[0])

    def update(self, report_id: str, status: str | None = None, notes: str | None = None) -> None:
        if status is not None and status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        sets, params = ["updated_at=?"], [int(time.time() * 1000)]
        if status is not None:
            sets.append("status=?")
            params.append(status)
        if notes is not None:
            sets.append("notes=?")
            params.append(notes)
        self._q(f"UPDATE reports SET {', '.join(sets)} WHERE report_id=?", (*params, report_id), commit=True)

    def set_authenticity(self, report_id: str, result: dict) -> None:
        self._q("UPDATE reports SET authenticity=? WHERE report_id=?", (json.dumps(result), report_id), commit=True)

    def set_translation(self, report_id: str, result: dict) -> None:
        self._q("UPDATE reports SET translation_json=? WHERE report_id=?", (json.dumps(result, ensure_ascii=False),
                                                                            report_id), commit=True)

    @staticmethod
    def _row(r) -> dict:
        d = dict(r)
        d.pop("tracking_hash", None)
        d["authenticity"] = json.loads(d["authenticity"]) if d.get("authenticity") else None
        d["language"] = d.get("language") or "en"
        d["urgent"] = bool(d.get("urgent"))
        t = d.pop("translation_json", None)
        d["translation"] = json.loads(t) if t else None
        return d


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()
