-- Aequitas event store. This schema is the contract between the vision side and the chat side.
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
  category        TEXT NOT NULL,  -- a configured category name (aequitas/taxonomy.py)
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
  vlm_ms     REAL,  -- summarize_ms + classify_ms
  db_ms      REAL,
  summarize_ms REAL,  -- stage 1: VLM summary of the key frames
  classify_ms  REAL,  -- stage 2: text classification of the summary
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
  started_at    INTEGER,            -- analysis began; finished_at - started_at = total execution time
  finished_at   INTEGER
);
CREATE INDEX IF NOT EXISTS ix_uploads_cam_time ON uploads(camera_id, start_ts);

-- One row per understanding job (summarize + classify), including windows that produced no event,
-- so per-video stage totals are complete. run_id is the upload ID for web uploads.
CREATE TABLE IF NOT EXISTS stage_timings (
  run_id        TEXT,
  camera_id     TEXT NOT NULL,
  kind          TEXT NOT NULL,      -- window | candidate
  queue_ms      REAL,
  summarize_ms  REAL,
  classify_ms   REAL,
  ok            INTEGER NOT NULL,   -- 1 if both stages returned valid output
  created_at    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_stage_timings_run ON stage_timings(run_id);

CREATE TABLE IF NOT EXISTS dropped_candidates (
  camera_id     TEXT NOT NULL,
  ts            INTEGER NOT NULL,
  category      TEXT NOT NULL,
  trigger_score REAL NOT NULL,
  reason        TEXT NOT NULL
);

-- Police accounts (created with `aequitas create-user`), sign-in sessions and an audit trail.
CREATE TABLE IF NOT EXISTS users (
  user_id       INTEGER PRIMARY KEY,
  username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
  password_hash TEXT NOT NULL,              -- scrypt, salted
  role          TEXT NOT NULL CHECK (role IN ('admin','officer')),
  disabled      INTEGER NOT NULL DEFAULT 0,
  created_at    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
  token_hash    TEXT PRIMARY KEY,           -- sha256 of the cookie token; the token itself is never stored
  user_id       INTEGER NOT NULL REFERENCES users(user_id),
  created_at    INTEGER NOT NULL,
  expires_at    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
  id            INTEGER PRIMARY KEY,
  user_id       INTEGER,                    -- NULL for citizen / system actions
  action        TEXT NOT NULL,
  target        TEXT,
  detail        TEXT,
  ip            TEXT,
  created_at    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_audit_target ON audit_log(target);

-- Citizen reports (public portal) and their review by the police.
CREATE TABLE IF NOT EXISTS reports (
  report_id      TEXT PRIMARY KEY,
  tracking_hash  TEXT NOT NULL,             -- unused (was a tracking-code hash); kept so old databases load
  upload_id      TEXT NOT NULL REFERENCES uploads(upload_id),
  camera_id      TEXT NOT NULL,
  description    TEXT NOT NULL,
  location       TEXT,
  incident_at    INTEGER,                   -- when the citizen says it happened (epoch ms)
  contact        TEXT,                      -- optional; visible to police only
  sha256         TEXT NOT NULL,             -- of the uploaded file, computed on arrival (integrity)
  duplicate_of   TEXT,                      -- an earlier report with the identical file
  status         TEXT NOT NULL CHECK (status IN ('received','under_review','action_taken','closed','rejected')),
  authenticity   TEXT,                      -- JSON from aequitas/authenticity.py
  notes          TEXT,                      -- police notes, never shown to the citizen
  created_at     INTEGER NOT NULL,
  updated_at     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_reports_sha ON reports(sha256);
CREATE INDEX IF NOT EXISTS ix_reports_created ON reports(created_at);

-- Every analysed window (incident or not): what the live dashboard shows per camera.
CREATE TABLE IF NOT EXISTS windows (
  id            INTEGER PRIMARY KEY,
  camera_id     TEXT NOT NULL,
  start_ts      INTEGER NOT NULL,
  end_ts        INTEGER NOT NULL,
  summary       TEXT,
  label         TEXT,                       -- stage-2 category, or none; NULL if a stage failed
  confidence    REAL,
  summarize_ms  REAL,
  classify_ms   REAL,
  lag_ms        REAL,                       -- last frame captured -> result ready ("behind live")
  created_at    INTEGER NOT NULL,
  violence      REAL,                       -- violence clip model score (second signal), if loaded
  source        TEXT                        -- vlm | clip_model | both: which signal set the label
);
CREATE INDEX IF NOT EXISTS ix_windows_cam ON windows(camera_id, id);
