-- Minerva event store. This schema is the contract between the vision side and the chat side.
-- All times are UTC epoch milliseconds.

CREATE TABLE IF NOT EXISTS cameras (
  camera_id     TEXT PRIMARY KEY,
  name          TEXT NOT NULL,
  aliases       TEXT NOT NULL DEFAULT '[]',
  source_uri    TEXT NOT NULL,
  zones         TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS events (
  event_id        INTEGER PRIMARY KEY,
  event_uid       TEXT NOT NULL UNIQUE,
  camera_id       TEXT NOT NULL REFERENCES cameras(camera_id),
  start_ts        INTEGER NOT NULL,
  end_ts          INTEGER NOT NULL,
  category        TEXT NOT NULL CHECK (category IN
                    ('person_down','fighting','tailgating','crowd_incident','suspicious_activity')),
  subtype         TEXT,
  status          TEXT NOT NULL CHECK (status IN ('confirmed','disputed','rejected','unverified')),
  confidence      REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
  trigger_score   REAL NOT NULL,
  vlm_category    TEXT,
  summary         TEXT,
  zone            TEXT,
  track_ids       TEXT NOT NULL DEFAULT '[]',
  keyframe_path   TEXT NOT NULL,
  clip_path       TEXT,
  model_versions  TEXT NOT NULL,
  created_at      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_events_cam_time ON events(camera_id, start_ts);
CREATE INDEX IF NOT EXISTS ix_events_cat_time ON events(category, start_ts);

CREATE VIRTUAL TABLE IF NOT EXISTS events_fts
  USING fts5(summary, content='events', content_rowid='event_id');

CREATE TRIGGER IF NOT EXISTS events_ai AFTER INSERT ON events BEGIN
  INSERT INTO events_fts(rowid, summary) VALUES (new.event_id, coalesce(new.summary, ''));
END;
CREATE TRIGGER IF NOT EXISTS events_ad AFTER DELETE ON events BEGIN
  INSERT INTO events_fts(events_fts, rowid, summary) VALUES ('delete', old.event_id, coalesce(old.summary, ''));
END;
CREATE TRIGGER IF NOT EXISTS events_au AFTER UPDATE OF summary ON events BEGIN
  INSERT INTO events_fts(events_fts, rowid, summary) VALUES ('delete', old.event_id, coalesce(old.summary, ''));
  INSERT INTO events_fts(rowid, summary) VALUES (new.event_id, coalesce(new.summary, ''));
END;

CREATE TABLE IF NOT EXISTS event_timings (
  event_id   INTEGER PRIMARY KEY REFERENCES events(event_id),
  detect_ms  REAL,
  trigger_ms REAL,
  queue_ms   REAL,
  vlm_ms     REAL,
  db_ms      REAL,
  proc_ms    REAL,  -- trigger frame captured -> row committed
  e2e_ms     REAL   -- condition onset -> row committed (includes the rule's sustain time)
);

-- Videos uploaded through the web app, and the analysis job for each.
CREATE TABLE IF NOT EXISTS uploads (
  upload_id     TEXT PRIMARY KEY,
  camera_id     TEXT NOT NULL REFERENCES cameras(camera_id),
  filename      TEXT NOT NULL,
  path          TEXT NOT NULL,
  start_ts      INTEGER NOT NULL,   -- wall time the first frame represents
  duration_ms   INTEGER,
  mode          TEXT NOT NULL,
  vlm_model     TEXT,
  status        TEXT NOT NULL,      -- queued | processing | verifying | done | failed | cancelled
  frames_done   INTEGER NOT NULL DEFAULT 0,
  frames_total  INTEGER,
  events_found  INTEGER NOT NULL DEFAULT 0,
  error         TEXT,
  created_at    INTEGER NOT NULL,
  finished_at   INTEGER
);
CREATE INDEX IF NOT EXISTS ix_uploads_cam_time ON uploads(camera_id, start_ts);

CREATE TABLE IF NOT EXISTS dropped_candidates (
  camera_id     TEXT NOT NULL,
  ts            INTEGER NOT NULL,
  category      TEXT NOT NULL,
  trigger_score REAL NOT NULL,
  reason        TEXT NOT NULL
);
