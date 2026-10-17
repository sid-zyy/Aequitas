"""SQLite access. One connection shared across threads behind a lock; WAL lets the UI read concurrently."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from aequitas.config import CameraCfg
from aequitas.store.query import QueryFilter, build_sql


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
    summarize_ms: float | None = None
    classify_ms: float | None = None


# Columns added after the first release: (table, column, type). Older databases get them on open.
MIGRATIONS = [
    ("event_timings", "summarize_ms", "REAL"),
    ("event_timings", "classify_ms", "REAL"),
    ("uploads", "started_at", "INTEGER"),
    ("windows", "violence", "REAL"),
    ("windows", "source", "TEXT"),
    ("uploads", "kind", "TEXT"),        # NULL: dashboard upload or citizen report; "bodycam": officer recording
    ("uploads", "view", "TEXT"),        # cctv | close (stage-1 prompt variant)
    ("uploads", "officer", "TEXT"),     # bodycam: officer or unit, as typed by the uploader
    ("events", "review", "TEXT"),       # officer decision: confirmed | dismissed (NULL = not reviewed yet)
    ("events", "reviewed_by", "TEXT"),
    ("events", "reviewed_at", "INTEGER"),
    ("uploads", "summary_json", "TEXT"),  # bodycam: whole-recording overview + key points
    ("uploads", "summary_ms", "REAL"),
    ("reports", "language", "TEXT"),        # en | hi | kn: the language the citizen wrote in (text kept as written)
    ("reports", "urgent", "INTEGER"),       # 1: the citizen told the portal's assistant they need help now
    ("reports", "translation_json", "TEXT"),  # officer-requested English translation {text, ms, model}
    ("cameras", "area", "TEXT"),            # locality the camera is in (chat: "fights in Indiranagar")
    ("cameras", "area_aliases", "TEXT"),    # JSON list of other names for the locality
    ("audit_log", "prev_hash", "TEXT"),     # hash chain (aequitas/auth.py): the previous entry's hash
    ("audit_log", "hash", "TEXT"),          # HMAC-SHA256 over this entry and prev_hash
]


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
        schema = resources.files("aequitas.store").joinpath("schema.sql").read_text(encoding="utf-8")
        with self._lock:
            self._conn.executescript(schema)
            self._migrate()

    def _migrate(self) -> None:
        self._drop_category_check()
        for table, column, kind in MIGRATIONS:
            have = {r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})")}
            if column not in have:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
        self._conn.commit()

    def _drop_category_check(self) -> None:
        """Databases from before configurable categories have a CHECK limiting `category` to the first
        five. SQLite cannot drop a constraint, so the table is rebuilt (rowids, and so the FTS index
        and timings, stay valid)."""
        row = self._conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='events'").fetchone()
        if row is None or "CHECK (category IN" not in row["sql"]:
            return
        new_sql = re.sub(r"CHECK \(category IN\s*\([^)]*\)\)", "", row["sql"]).replace(
            "CREATE TABLE events", "CREATE TABLE events_new", 1)
        self._conn.execute("PRAGMA foreign_keys=OFF")
        try:
            with self._conn:
                self._conn.execute(new_sql)
                self._conn.execute("INSERT INTO events_new SELECT * FROM events")
                self._conn.execute("DROP TABLE events")  # also drops its indexes and triggers
                self._conn.execute("ALTER TABLE events_new RENAME TO events")
        finally:
            self._conn.execute("PRAGMA foreign_keys=ON")
        schema = resources.files("aequitas.store").joinpath("schema.sql").read_text(encoding="utf-8")
        self._conn.executescript(schema)  # recreate the indexes and FTS triggers
        with self._conn:
            self._conn.execute("INSERT INTO events_fts(events_fts) VALUES ('rebuild')")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def upsert_camera(self, cam: CameraCfg) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO cameras(camera_id, name, aliases, source_uri, zones, area, area_aliases) VALUES (?,?,?,?,?,?,?) "
                "ON CONFLICT(camera_id) DO UPDATE SET name=excluded.name, aliases=excluded.aliases, "
                "source_uri=excluded.source_uri, zones=excluded.zones, area=excluded.area, "
                "area_aliases=excluded.area_aliases",
                (cam.camera_id, cam.name, json.dumps(cam.aliases), cam.source, cam.zones.model_dump_json(),
                 cam.area, json.dumps(cam.area_aliases)),
            )

    def cameras(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT camera_id, name, aliases, area, area_aliases FROM cameras ORDER BY camera_id").fetchall()
        return [{"camera_id": r["camera_id"], "name": r["name"], "aliases": json.loads(r["aliases"]),
                 "area": r["area"], "area_aliases": json.loads(r["area_aliases"] or "[]")} for r in rows]

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
                    "e2e_ms, summarize_ms, classify_ms) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (event_id, timings.detect_ms, timings.trigger_ms, timings.queue_ms, timings.vlm_ms,
                     timings.db_ms, timings.proc_ms, timings.e2e_ms, timings.summarize_ms, timings.classify_ms),
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

    def record_window(self, camera_id: str, start_ts: int, end_ts: int, summary: str | None, label: str | None,
                      confidence: float | None, summarize_ms: float | None, classify_ms: float | None,
                      lag_ms: float | None, keep: int = 200, violence: float | None = None,
                      source: str | None = None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO windows(camera_id, start_ts, end_ts, summary, label, confidence, summarize_ms, "
                "classify_ms, lag_ms, created_at, violence, source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (camera_id, start_ts, end_ts, summary, label, confidence, summarize_ms, classify_ms, lag_ms,
                 int(time.time() * 1000), violence, source),
            )
            self._conn.execute(  # keep only the latest `keep` per camera
                "DELETE FROM windows WHERE camera_id=? AND id <= "
                "(SELECT id FROM windows WHERE camera_id=? ORDER BY id DESC LIMIT 1 OFFSET ?)",
                (camera_id, camera_id, keep),
            )

    def recent_windows(self, camera_id: str, n: int = 5) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM windows WHERE camera_id=? ORDER BY id DESC LIMIT ?", (camera_id, n)
            ).fetchall()
        return [dict(r) for r in rows]

    def record_stage(self, run_id: str | None, camera_id: str, kind: str, queue_ms: float | None,
                     summarize_ms: float | None, classify_ms: float | None, ok: bool) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO stage_timings(run_id, camera_id, kind, queue_ms, summarize_ms, classify_ms, ok, "
                "created_at) VALUES (?,?,?,?,?,?,?,?)",
                (run_id, camera_id, kind, queue_ms, summarize_ms, classify_ms, int(ok), int(time.time() * 1000)),
            )

    def stage_summary(self, run_id: str) -> dict:
        """Per-stage totals for one run: job count, and sum/mean ms of each stage."""
        with self._lock:
            r = self._conn.execute(
                "SELECT SUM(kind NOT IN ('skipped','gated','blank')) AS jobs, "
                "SUM(ok AND kind NOT IN ('skipped','gated','blank')) AS ok, "
                "SUM(kind='skipped') AS skipped, SUM(kind='gated') AS gated, SUM(kind='blank') AS blank, SUM(summarize_ms) AS sum_s, AVG(summarize_ms) AS avg_s, "
                "SUM(classify_ms) AS sum_c, AVG(classify_ms) AS avg_c, AVG(queue_ms) AS avg_q "
                "FROM stage_timings WHERE run_id=?", (run_id,),
            ).fetchone()

        def rnd(v):
            return round(v, 1) if v is not None else None

        return {"jobs": r["jobs"] or 0, "ok": r["ok"] or 0, "skipped_static": r["skipped"] or 0,
                "skipped_gate": r["gated"] or 0,
                "skipped_blank": r["blank"] or 0,
                "summarize_ms": {"total": rnd(r["sum_s"]), "mean": rnd(r["avg_s"])},
                "classify_ms": {"total": rnd(r["sum_c"]), "mean": rnd(r["avg_c"])},
                "queue_ms": {"mean": rnd(r["avg_q"])}}

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

    # ---------- the chatbot's vector search (aequitas/chat/vectors.py) ----------

    def event_ids(self, f: QueryFilter, cap: int = 5000) -> list[int]:
        """IDs of every event the structured filter allows (no text match): the candidates vector search ranks."""
        sql, params = build_sql(f.model_copy(update={"text": None, "limit": 1}))
        sql = sql.replace("SELECT e.*, c.name AS camera_name, c.area AS camera_area", "SELECT e.event_id", 1)
        with self._lock:
            return [r[0] for r in self._conn.execute(sql, params[:-1] + [cap]).fetchall()]

    def events_by_ids(self, ids: list[int]) -> list[dict]:
        """Events in the order of `ids` (missing ones skipped)."""
        rows = {r["event_id"]: r for r in (self.event(i) for i in dict.fromkeys(ids)) if r}
        return [rows[i] for i in ids if i in rows]

    def fts_ids(self, text: str, ids: list[int] | None = None, limit: int = 50) -> list[int]:
        """Events whose summary contains the words, best BM25 rank first (the keyword half of hybrid search)."""
        from aequitas.store.query import fts_expression

        expr = fts_expression(text)
        if not expr:
            return []
        sql = "SELECT rowid FROM events_fts WHERE events_fts MATCH ?"
        params: list = [expr]
        if ids is not None:
            if not ids:
                return []
            sql += f" AND rowid IN ({','.join('?' * len(ids))})"
            params += ids
        with self._lock:
            return [r[0] for r in self._conn.execute(sql + " ORDER BY rank LIMIT ?", params + [limit]).fetchall()]

    def summaries_after(self, kind: str, after_id: int, limit: int = 256) -> list[dict]:
        """Events or windows with a summary and an ID above `after_id`, oldest first (for indexing)."""
        table, key = ("events", "event_id") if kind == "event" else ("windows", "id")
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {key} AS id, camera_id, start_ts, end_ts, summary FROM {table} "
                f"WHERE {key} > ? AND summary IS NOT NULL AND summary != '' ORDER BY {key} LIMIT ?",
                (after_id, limit)).fetchall()
        return [dict(r) for r in rows]

    def summary_words(self) -> set[str]:
        """Every word (3+ letters) in a stored event or window summary: the chat's typo fixer leaves these alone."""
        with self._lock:
            rows = self._conn.execute("SELECT summary FROM events WHERE summary IS NOT NULL UNION ALL "
                                      "SELECT summary FROM windows WHERE summary IS NOT NULL").fetchall()
        return {w for (s,) in rows for w in re.findall(r"[a-z]{3,}", s.lower())}

    def window_ids(self, camera_ids: list[str] | None = None, start_ms: int | None = None,
                   end_ms: int | None = None) -> list[int]:
        where, params = ["summary IS NOT NULL"], []
        if camera_ids:
            where.append(f"camera_id IN ({','.join('?' * len(camera_ids))})")
            params += camera_ids
        if start_ms is not None:
            where.append("end_ts >= ?")
            params.append(start_ms)
        if end_ms is not None:
            where.append("start_ts <= ?")
            params.append(end_ms)
        with self._lock:
            return [r[0] for r in self._conn.execute(
                f"SELECT id FROM windows WHERE {' AND '.join(where)}", params).fetchall()]

    def windows_by_ids(self, ids: list[int]) -> list[dict]:
        if not ids:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT w.*, c.name AS camera_name, c.area AS camera_area FROM windows w "
                f"LEFT JOIN cameras c USING(camera_id) WHERE w.id IN ({','.join('?' * len(ids))})", ids).fetchall()
        by_id = {r["id"]: dict(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def windows_near(self, camera_id: str, start_ms: int, end_ms: int, pad_ms: int = 20000) -> list[dict]:
        """The analysed windows around an event: what the camera saw just before, during and after it."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT w.*, c.name AS camera_name, c.area AS camera_area FROM windows w "
                "LEFT JOIN cameras c USING(camera_id) WHERE w.camera_id=? AND w.end_ts >= ? AND w.start_ts <= ? "
                "AND w.summary IS NOT NULL ORDER BY w.start_ts LIMIT 12",
                (camera_id, start_ms - pad_ms, end_ms + pad_ms)).fetchall()
        return [dict(r) for r in rows]

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
                "SELECT e.*, c.name AS camera_name, c.area AS camera_area FROM events e LEFT JOIN cameras c USING(camera_id) "
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

    def uploads_of_kind(self, kind: str, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM uploads WHERE kind=? ORDER BY created_at DESC LIMIT ?", (kind, limit)).fetchall()
        return [dict(r) for r in rows]

    def windows_for(self, camera_id: str) -> list[dict]:
        """Every stored window of a camera, oldest first (a bodycam recording has its own camera)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM windows WHERE camera_id=? ORDER BY start_ts, id", (camera_id,)).fetchall()
        return [dict(r) for r in rows]

    def events_for(self, camera_id: str) -> list[dict]:
        """Every event of a camera in time order, any status (the review page shows unverified ones too)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE camera_id=? ORDER BY start_ts, event_id", (camera_id,)).fetchall()
        return [dict(r) for r in rows]

    def review_event(self, event_id: int, decision: str, by: str) -> bool:
        """An officer's decision on an AI-detected event. The AI's own status stays as it was."""
        with self._lock, self._conn:
            cur = self._conn.execute("UPDATE events SET review=?, reviewed_by=?, reviewed_at=? WHERE event_id=?",
                                     (decision, by, int(time.time() * 1000), event_id))
        return cur.rowcount == 1

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

    def event_timings(self, event_id: int) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM event_timings WHERE event_id=?", (event_id,)).fetchone()
        return dict(r) if r else None

    def window_stats(self, since_ms: int) -> dict:
        """Live analysis since a moment: windows analysed, failed, and p50 stage times and lag (health page)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT label, summarize_ms, classify_ms, lag_ms FROM windows WHERE created_at>=?", (since_ms,)
            ).fetchall()

        def p50(key):
            v = sorted(r[key] for r in rows if r[key] is not None)
            return round(v[len(v) // 2], 1) if v else None

        return {"windows": len(rows), "failed": sum(r["label"] is None for r in rows),
                "summarize_ms_p50": p50("summarize_ms"), "classify_ms_p50": p50("classify_ms"),
                "lag_ms_p50": p50("lag_ms")}

    def upload_counts(self) -> dict:
        with self._lock:
            return dict(self._conn.execute("SELECT status, COUNT(*) FROM uploads GROUP BY status").fetchall())

    def dropped_since(self, since_ms: int) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM dropped_candidates WHERE ts>=?", (since_ms,)).fetchone()[0]

    def timings(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT t.*, e.category, e.camera_id FROM event_timings t JOIN events e USING(event_id)"
            ).fetchall()
        return [dict(r) for r in rows]
