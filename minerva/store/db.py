"""SQLite access. One connection shared across threads behind a lock; WAL lets the UI read concurrently."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from minerva.config import CameraCfg
from minerva.store.query import QueryFilter, build_sql


@dataclass
class EventRecord:
    event_uid: str
    camera_id: str
    start_ts: int
    end_ts: int
    category: str
    subtype: str | None
    status: str
    confidence: float
    trigger_score: float
    vlm_category: str | None
    summary: str | None
    zone: str | None
    track_ids: list[int]
    keyframe_path: str
    model_versions: dict
    clip_path: str | None = None
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class Timings:
    detect_ms: float | None = None
    trigger_ms: float | None = None
    queue_ms: float | None = None
    vlm_ms: float | None = None
    db_ms: float | None = None
    proc_ms: float | None = None
    e2e_ms: float | None = None


class EventStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        schema = resources.files("minerva.store").joinpath("schema.sql").read_text(encoding="utf-8")
        with self._lock:
            self._conn.executescript(schema)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def upsert_camera(self, cam: CameraCfg) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO cameras(camera_id, name, aliases, source_uri, zones) VALUES (?,?,?,?,?) "
                "ON CONFLICT(camera_id) DO UPDATE SET name=excluded.name, aliases=excluded.aliases, "
                "source_uri=excluded.source_uri, zones=excluded.zones",
                (cam.camera_id, cam.name, json.dumps(cam.aliases), cam.source, cam.zones.model_dump_json()),
            )

    def cameras(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT camera_id, name, aliases FROM cameras ORDER BY camera_id").fetchall()
        return [{"camera_id": r["camera_id"], "name": r["name"], "aliases": json.loads(r["aliases"])} for r in rows]

    def insert_event(self, ev: EventRecord, timings: Timings | None = None) -> int:
        t0 = time.perf_counter()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO events(event_uid, camera_id, start_ts, end_ts, category, subtype, status, confidence, "
                "trigger_score, vlm_category, summary, zone, track_ids, keyframe_path, clip_path, model_versions, "
                "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    ev.event_uid, ev.camera_id, ev.start_ts, ev.end_ts, ev.category, ev.subtype, ev.status,
                    round(min(1.0, max(0.0, ev.confidence)), 4), ev.trigger_score, ev.vlm_category, ev.summary,
                    ev.zone, json.dumps(ev.track_ids), ev.keyframe_path, ev.clip_path,
                    json.dumps(ev.model_versions), ev.created_at,
                ),
            )
            event_id = cur.lastrowid
            if timings is not None:
                timings.db_ms = (time.perf_counter() - t0) * 1000
                self._conn.execute(
                    "INSERT INTO event_timings(event_id, detect_ms, trigger_ms, queue_ms, vlm_ms, db_ms, proc_ms, "
                    "e2e_ms) VALUES (?,?,?,?,?,?,?,?)",
                    (event_id, timings.detect_ms, timings.trigger_ms, timings.queue_ms, timings.vlm_ms,
                     timings.db_ms, timings.proc_ms, timings.e2e_ms),
                )
        return event_id

    def finish_timings(self, event_id: int, proc_ms: float, e2e_ms: float) -> None:
        """proc/e2e are only known after the commit, so they are written in a second step."""
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE event_timings SET proc_ms=?, e2e_ms=? WHERE event_id=?", (proc_ms, e2e_ms, event_id)
            )

    def extend_end(self, event_uid: str, end_ts: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE events SET end_ts=MAX(end_ts, ?) WHERE event_uid=?", (end_ts, event_uid)
            )

    def update_event(self, event_uid: str, end_ts: int, status: str, confidence: float) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE events SET end_ts=MAX(end_ts, ?), status=?, confidence=? WHERE event_uid=?",
                (end_ts, status, round(min(1.0, max(0.0, confidence)), 4), event_uid),
            )

    def record_dropped(self, camera_id: str, ts: int, category: str, score: float, reason: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO dropped_candidates(camera_id, ts, category, trigger_score, reason) VALUES (?,?,?,?,?)",
                (camera_id, ts, category, score, reason),
            )

    def query(self, f: QueryFilter) -> list[dict]:
        sql, params = build_sql(f)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["track_ids"] = json.loads(d["track_ids"])
            d["model_versions"] = json.loads(d["model_versions"])
            out.append(d)
        return out

    def count(self, f: QueryFilter) -> int:
        sql, params = build_sql(f, count=True)
        with self._lock:
            return self._conn.execute(sql, params).fetchone()[0]

    # ---------- web app helpers ----------

    def camera(self, camera_id: str) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM cameras WHERE camera_id=?", (camera_id,)).fetchone()
        if r is None:
            return None
        d = dict(r)
        d["aliases"] = json.loads(d["aliases"])
        d["zones"] = json.loads(d["zones"])
        return d

    def event(self, event_id: int) -> dict | None:
        with self._lock:
            r = self._conn.execute(
                "SELECT e.*, c.name AS camera_name FROM events e LEFT JOIN cameras c USING(camera_id) "
                "WHERE e.event_id=?", (event_id,),
            ).fetchone()
        if r is None:
            return None
        d = dict(r)
        d["track_ids"] = json.loads(d["track_ids"])
        d["model_versions"] = json.loads(d["model_versions"])
        return d

    def stats(self, since_ms: int) -> dict:
        with self._lock:
            q = self._conn.execute
            by_status = dict(q("SELECT status, COUNT(*) FROM events GROUP BY status").fetchall())
            by_category = dict(q("SELECT category, COUNT(*) FROM events WHERE status='confirmed' "
                                 "GROUP BY category").fetchall())
            today = q("SELECT COUNT(*) FROM events WHERE status='confirmed' AND start_ts>=?", (since_ms,)).fetchone()[0]
            cameras = q("SELECT COUNT(*) FROM cameras").fetchone()[0]
            videos = q("SELECT COUNT(*) FROM uploads WHERE status='done'").fetchone()[0]
        return {"by_status": by_status, "by_category": by_category, "confirmed_today": today,
                "cameras": cameras, "videos_analyzed": videos}

    def count_events_since(self, camera_id: str, created_after_ms: int) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE camera_id=? AND created_at>=?", (camera_id, created_after_ms)
            ).fetchone()[0]

    def insert_upload(self, u: dict) -> None:
        cols = ",".join(u)
        with self._lock, self._conn:
            self._conn.execute(f"INSERT INTO uploads({cols}) VALUES ({','.join('?' * len(u))})", tuple(u.values()))

    def update_upload(self, upload_id: str, **fields) -> None:
        if not fields:
            return
        sets = ",".join(f"{k}=?" for k in fields)
        with self._lock, self._conn:
            self._conn.execute(f"UPDATE uploads SET {sets} WHERE upload_id=?", (*fields.values(), upload_id))

    def uploads(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT u.*, c.name AS camera_name FROM uploads u LEFT JOIN cameras c USING(camera_id) "
                "ORDER BY u.created_at DESC LIMIT ?", (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def upload(self, upload_id: str) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM uploads WHERE upload_id=?", (upload_id,)).fetchone()
        return dict(r) if r else None

    def upload_for_event(self, camera_id: str, ts_ms: int) -> dict | None:
        """The uploaded video that covers this moment on this camera, if any."""
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM uploads WHERE camera_id=? AND start_ts<=? AND start_ts+COALESCE(duration_ms,0)>=? "
                "ORDER BY created_at DESC LIMIT 1", (camera_id, ts_ms, ts_ms),
            ).fetchone()
        return dict(r) if r else None

    def fail_interrupted_uploads(self) -> int:
        with self._lock, self._conn:
            return self._conn.execute(
                "UPDATE uploads SET status='failed', error='interrupted by restart' "
                "WHERE status IN ('queued','processing','verifying')"
            ).rowcount

    def timings(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT t.*, e.category, e.camera_id FROM event_timings t JOIN events e USING(event_id)"
            ).fetchall()
        return [dict(r) for r in rows]
