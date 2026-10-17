"""Aequitas web app: upload videos for analysis, browse events, and ask questions.
Serves a JSON API under /api and the single-page frontend from ./static. Binds to localhost."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response as RawResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from aequitas.auth import Auth, LockedOut
from aequitas.chat.answer import ChatService
from aequitas.chat.video_qa import answer_video
from aequitas import emergency
from aequitas.config import CameraCfg, Settings, resolve_device
from aequitas.store.db import EventStore
from aequitas.store.query import QueryFilter
from aequitas.reports import PUBLIC_STATUS, ReportStore
from aequitas.taxonomy import label
from aequitas.translate import LANGUAGES, detect_language, translate_to_english
from aequitas.types import Status
from aequitas.understand.priority import GATE
from aequitas.web.bodycam import ReviewBody, make_summary, reviewed, timeline
from aequitas.web.jobs import JobManager, frames_expected, probe_video
from aequitas.web.live import LiveManager, load_localities

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
DIST = Path(__file__).parent / "frontend_dist"  # built React app
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MAX_UPLOAD_BYTES = 4 * 1024**3


class ChatContext(BaseModel):
    """What the previous answer was about, sent back by the client so follow-ups work ("and yesterday?")."""

    filter: dict | None = None
    event_ids: list[int] = Field(default_factory=list, max_length=100)
    mode: str | None = Field(default=None, max_length=20)


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    context: ChatContext | None = None

    def ctx(self) -> dict | None:
        return self.context.model_dump() if self.context and self.context.filter else None


class AiSwitch(BaseModel):
    paused: bool


class LoginBody(BaseModel):
    username: str = Field(min_length=1, max_length=40)
    password: str = Field(min_length=1, max_length=200)


class AssistBody(BaseModel):
    lang: Literal["en", "hi", "kn"] = "en"
    message: str | None = Field(default=None, max_length=500)
    choice: Literal["yes", "no"] | None = None


class VideoTurn(BaseModel):
    q: str = Field(max_length=500)
    a: str = Field(max_length=2000)


class VideoAskBody(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    history: list[VideoTurn] = Field(default_factory=list, max_length=8)


class ReportUpdate(BaseModel):
    status: Literal["received", "under_review", "action_taken", "closed", "rejected"] | None = None
    notes: str | None = Field(default=None, max_length=4000)


SESSION_COOKIE = "aeq_session"
# Reachable without signing in: the login itself and the citizen portal.
PUBLIC_API = ("/api/auth/login", "/api/citizen/")


def _session_token(request) -> str | None:
    token = request.cookies.get(SESSION_COOKIE)
    header = request.headers.get("authorization", "")
    if not token and header.lower().startswith("bearer "):
        token = header[7:].strip()
    return token


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "camera"


def _event_json(r: dict, store: EventStore) -> dict:
    upload = store.upload_for_event(r["camera_id"], r["start_ts"])
    video = None
    if upload:
        video = {"url": f"/api/uploads/{upload['upload_id']}/video",
                 "offset_s": max(0.0, (r["start_ts"] - upload["start_ts"]) / 1000)}
    return {
        "id": r["event_id"],
        "camera_id": r["camera_id"],
        "camera_name": r.get("camera_name") or r["camera_id"],
        "area": r.get("camera_area"),
        "start_ts": r["start_ts"],
        "end_ts": r["end_ts"],
        "category": r["category"],
        "subtype": r.get("subtype"),
        "status": r["status"],
        "confidence": r["confidence"],
        "trigger_score": r.get("trigger_score"),
        "vlm_category": r.get("vlm_category"),
        "summary": r.get("summary"),
        "zone": r.get("zone"),
        "track_ids": r.get("track_ids", []),
        "models": r.get("model_versions", {}),
        "keyframe_url": f"/api/events/{r['event_id']}/keyframe",
        "video": video,
        "relevance": r.get("relevance"),  # cosine similarity to the question (vector search), if ranked by it
        "matched": r.get("matched"),      # meaning | words | meaning+words
    }


def _moment_json(w: dict, store: EventStore) -> dict:
    """An analysed window the chat found by meaning: what a camera saw, incident or not."""
    upload = store.upload_for_event(w["camera_id"], w["start_ts"])
    video = None
    if upload:
        video = {"url": f"/api/uploads/{upload['upload_id']}/video",
                 "offset_s": max(0.0, (w["start_ts"] - upload["start_ts"]) / 1000)}
    return {"id": w["id"], "camera_id": w["camera_id"], "camera_name": w.get("camera_name") or w["camera_id"],
            "area": w.get("camera_area"), "start_ts": w["start_ts"], "end_ts": w["end_ts"],
            "summary": w.get("summary"), "label": w.get("label"), "relevance": w.get("relevance"), "video": video}


def create_app(settings: Settings, service: ChatService, camera_cfgs: list[CameraCfg]) -> FastAPI:
    store = service.store
    for cam in camera_cfgs:
        store.upsert_camera(cam)
    n = store.fail_interrupted_uploads()
    if n:
        log.warning("%d upload job(s) were interrupted by the last shutdown", n)
    jobs = JobManager(settings, store)
    upload_dir = settings.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(settings.device)

    app = FastAPI(title="Aequitas", docs_url="/api/docs", redoc_url=None)

    if settings.chat.vector.enabled and service.index is None and not getattr(service, "_index_loading", False):
        service._index_loading = True

        def _open_index():  # loading the embedding model takes seconds: don't hold up the server start
            from aequitas.chat.vectors import open_index

            idx = open_index(settings.chat.vector, store, settings.data_dir, settings.vlm.host)
            if idx is not None:
                service.index = idx
                idx.start_background_sync(settings.chat.vector.sync_s)
                log.info("vector index ready: %s", idx.stats())

        threading.Thread(target=_open_index, name="vector-open", daemon=True).start()

    auth = Auth(store, settings.auth.session_hours, settings.auth.max_failures, settings.auth.lockout_s)
    if settings.auth.demo_admin and not auth.users():
        auth.create_user("admin", "admin", "admin", allow_weak=True)
        log.warning("no accounts yet: created demo account admin/admin (change it: aequitas create-user)")
    app.state.auth = auth
    started_at = time.time()

    def _require_admin(request: Request) -> None:
        """Admin-only pages (the audit log). With sign-in switched off (development), anyone may see them."""
        user = request.state.user
        if user is None and not settings.auth.required:
            return
        if user is None or user["role"] != "admin":
            raise HTTPException(403, "Only an admin can see this")

    @app.middleware("http")
    async def guard_and_revalidate(request: Request, call_next):
        path = request.url.path
        request.state.user = None
        if path.startswith("/api/") and not path.startswith(PUBLIC_API):
            user = auth.session_user(_session_token(request))
            if user is None and settings.auth.required:
                return JSONResponse({"detail": "sign in required"}, status_code=401)
            request.state.user = user
        response = await call_next(request)
        if not path.startswith("/api/"):
            # The frontend changes with the code; make browsers revalidate it (ETags keep this cheap).
            response.headers["Cache-Control"] = "no-cache"
        return response

    # ---------- police sign-in ----------

    @app.post("/api/auth/login")
    def login(body: LoginBody, request: Request, response: Response):
        ip = request.client.host if request.client else ""
        try:
            res = auth.login(body.username, body.password, ip)
        except LockedOut:
            raise HTTPException(429, "Too many failed sign-ins. Try again later.")
        if res is None:
            raise HTTPException(401, "Wrong username or password")
        token, user = res
        response.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="strict",
                            secure=settings.auth.cookie_secure, max_age=int(settings.auth.session_hours * 3600))
        return user

    @app.post("/api/auth/logout")
    def logout(request: Request, response: Response):
        auth.logout(_session_token(request))
        response.delete_cookie(SESSION_COOKIE)
        return {"ok": True}

    @app.get("/api/auth/me")
    def me(request: Request):
        if request.state.user is None:
            raise HTTPException(401, "sign in required")
        return request.state.user

    # ---------- status ----------

    @app.get("/api/status")
    def status():
        vlm_online, models = False, []
        try:
            from aequitas.understand.vlm_client import make_client

            client = make_client(settings.vlm)
            vlm_online = client.ping()
            if settings.vlm.backend == "ollama" and vlm_online:
                for m in client.client.list()["models"]:
                    name = m["model"]
                    try:
                        caps = client.client.show(name).get("capabilities") or []
                    except Exception:
                        caps = []
                    if not caps or "vision" in caps:
                        models.append(name)
        except Exception as e:
            log.debug("status check failed: %s", e)
        llm = service.parser.llm
        return {
            "device": device,
            "mode": settings.analysis.mode,
            "vlm": {"backend": settings.vlm.backend, "model": settings.vlm.model, "online": vlm_online},
            "chat": {"model": llm.model if llm else None, "online": llm is not None},
            "vision_models": models or [settings.vlm.model],
            "sample_fps": settings.sample_fps,
        }

    @app.get("/api/health")
    def health():
        """Control-room view of the system: model server, GPU, live lag, queue, audit chain, storage."""
        from aequitas.web.health import collect

        return collect(settings, store, auth, getattr(app.state, "live", None), started_at)

    # ---------- audit log (admin) ----------

    @app.get("/api/audit")
    def audit_log(request: Request, limit: int = Query(200, ge=1, le=1000), action: str | None = None,
                  user: str | None = None, target: str | None = Query(None, max_length=100),
                  before_id: int | None = None):
        _require_admin(request)
        return {"entries": auth.audit_log(limit, action, user, target, before_id), "actions": auth.audit_actions()}

    @app.get("/api/audit/verify")
    def audit_verify(request: Request):
        _require_admin(request)
        t0 = time.perf_counter()
        res = auth.verify_chain()
        return {**res, "ms": round((time.perf_counter() - t0) * 1000, 1)}

    @app.get("/api/stats")
    def stats():
        day0 = datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
        return store.stats(int(day0.timestamp() * 1000))

    @app.get("/api/categories")
    def categories():
        """The configured stage-2 label set, for filters and badges in the UI."""
        return [{"name": n, "label": label(n), "definition": d} for n, d in settings.categories.items()]

    @app.get("/api/cameras")
    def cameras():
        return [{"camera_id": c["camera_id"], "name": c["name"], "aliases": c["aliases"]} for c in store.cameras()]

    # ---------- events ----------

    @app.get("/api/events")
    def events(camera: str | None = None, category: str | None = None,
               status: Literal["confirmed", "all", "disputed", "rejected", "unverified"] = "confirmed",
               q: str | None = Query(None, max_length=200), limit: int = Query(60, ge=1, le=100)):
        statuses = list(Status) if status == "all" else [Status(status)]
        f = QueryFilter(categories=[category] if category else [], camera_ids=[camera] if camera else [],
                        statuses=statuses, text=q or None, limit=limit)
        return {"total": store.count(f), "events": [_event_json(r, store) for r in store.query(f)]}

    @app.get("/api/events/{event_id}")
    def event(event_id: int):
        r = store.event(event_id)
        if r is None:
            raise HTTPException(404, "event not found")
        return _event_json(r, store)

    @app.get("/api/events/{event_id}/keyframe")
    def keyframe(event_id: int):
        r = store.event(event_id)  # look up by ID only: clients never pass file paths
        if r is None or not Path(r["keyframe_path"]).is_file():
            raise HTTPException(404, "keyframe not found")
        return FileResponse(r["keyframe_path"], media_type="image/jpeg")

    @app.get("/api/events/{event_id}/evidence")
    def evidence(event_id: int, request: Request):
        """The incident as a self-checking ZIP: key frame, original video, manifest, hashes, audit trail."""
        from aequitas.evidence import build_package

        user = request.state.user
        try:
            data, manifest = build_package(store, auth, event_id, user["username"] if user else None)
        except LookupError:
            raise HTTPException(404, "event not found")
        digest = hashlib.sha256(data).hexdigest()
        auth.audit(user["user_id"] if user else None, "evidence_exported", f"event/{event_id}",
                   {"package_sha256": digest, "bytes": len(data), "files": sorted(manifest["files"]),
                    "audit_head": manifest["audit"]["chain_at_export"].get("head")},
                   request.client.host if request.client else "")
        name = f"aequitas-evidence-{event_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip"
        return RawResponse(data, media_type="application/zip",
                           headers={"Content-Disposition": f'attachment; filename="{name}"',
                                    "X-Evidence-SHA256": digest})

    # ---------- chat ----------

    @app.post("/api/ask")
    def ask(body: AskBody):
        t0 = time.perf_counter()
        with GATE.interactive():  # live analysis waits while an officer's question is answered
            ans = service.ask(body.question.strip(), body.ctx())
        f = ans.parsed.filter if ans.parsed else None
        return {
            "mode": ans.mode,
            "weak": ans.weak,
            "earlier": ans.earlier,
            "moments": [_moment_json(w, store) for w in ans.moments],
            "suggestions": ans.suggestions,
            "context": ans.context,
            "text": ans.text,
            "summary": ans.summary,
            "total": ans.total,
            "intent": f.intent if f else "list",
            "warnings": ans.parsed.warnings if ans.parsed else [],
            "parsed_with": ans.parsed.method if ans.parsed else None,
            "filter": f.model_dump(mode="json") if f else None,
            "events": [_event_json(r, store) for r in ans.rows],
            "elapsed_ms": round((time.perf_counter() - t0) * 1000),
            "timings": ans.timings,
        }

    @app.get("/api/chat/index")
    def chat_index():
        """The chatbot's vector database: engine, embedding model, how many summaries it holds."""
        idx = service.index
        if idx is None:
            return {"enabled": settings.chat.vector.enabled, "ready": False}
        return {"enabled": True, "ready": True, **idx.stats()}

    @app.post("/api/ask/stream")
    def ask_stream(body: AskBody):
        """Server-sent events: `retrieved` (the event list, from the database), `token` (summary text
        as it is generated), `summary` (final, grounded summary or null: drop the streamed text),
        `done` (timings)."""

        def events():
            with GATE.interactive():
                yield from _stream()

        def _stream():
            for ev in service.ask_stream(body.question.strip(), body.ctx()):
                if ev["type"] == "retrieved":
                    p = ev["parsed"]
                    f = p.filter
                    ev = {"type": "retrieved", "total": ev["total"], "head": ev["head"], "tail": ev["tail"],
                          "intent": f.intent, "warnings": p.warnings, "parsed_with": p.method,
                          "filter": f.model_dump(mode="json"), "retrieve_ms": ev["retrieve_ms"],
                          "parse_ms": ev["parse_ms"], "query_ms": ev.get("query_ms"),
                          "embed_ms": ev.get("embed_ms"), "vector_ms": ev.get("vector_ms"),
                          "mode": ev["mode"], "kept": ev["kept"], "weak": ev["weak"], "earlier": ev["earlier"], "suggestions": ev["suggestions"],
                          "context": ev["context"],
                          "moments": [_moment_json(w, store) for w in ev["moments"]],
                          "events": [_event_json(r, store) for r in ev["rows"]]}
                yield f"event: {ev['type']}\ndata: {json.dumps(ev, default=str)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ---------- uploads ----------

    def _upload_json(u: dict) -> dict:
        live = jobs.progress(u["upload_id"])
        done = live.get("frames_done", u["frames_done"])
        total = u["frames_total"] or 0
        status = u["status"]
        if status == "processing" and live.get("phase") in ("verifying", "summarising"):
            status = live["phase"]
        return {
            "id": u["upload_id"], "camera_id": u["camera_id"], "camera_name": u.get("camera_name"),
            "filename": u["filename"], "status": status, "mode": u["mode"], "vlm_model": u["vlm_model"],
            "frames_done": done, "frames_total": total,
            "progress": min(1.0, done / total) if total else 0.0,
            "vlm_backlog": live.get("vlm_backlog", 0), "events_found": u["events_found"],
            "duration_s": (u["duration_ms"] or 0) / 1000, "start_ts": u["start_ts"],
            "error": u["error"], "created_at": u["created_at"], "finished_at": u["finished_at"],
            "started_at": u.get("started_at"), "total_s": _total_s(u), "stages": store.stage_summary(u["upload_id"]),
        }

    def _total_s(u: dict) -> float | None:
        """Total execution time: analysis start -> done. Still running: elapsed so far."""
        start = u.get("started_at")
        if not start:
            return None
        end = u["finished_at"] or (int(time.time() * 1000) if u["status"] in ("processing", "verifying") else None)
        return round((end - start) / 1000, 2) if end else None

    @app.get("/api/uploads")
    def uploads():
        return [_upload_json(u) for u in store.uploads()]

    @app.post("/api/uploads")
    def create_upload(
        file: UploadFile = File(...),
        camera_id: str = Form(""),
        new_camera_name: str = Form(""),
        recorded_at: str = Form(""),
        mode: Literal["triggered", "periodic"] = Form("triggered"),
        vlm_model: str = Form(""),
    ):
        ext = Path(file.filename or "").suffix.lower()
        if ext not in VIDEO_EXTS:
            raise HTTPException(400, f"Unsupported file type '{ext}'. Use one of: {', '.join(sorted(VIDEO_EXTS))}")

        if new_camera_name.strip():
            name = new_camera_name.strip()[:60]
            existing = {c["camera_id"] for c in store.cameras()}
            cid, i = f"cam-{_slug(name)}", 2
            while cid in existing:
                cid, i = f"cam-{_slug(name)}-{i}", i + 1
            store.upsert_camera(CameraCfg(camera_id=cid, name=name, source="upload"))
        elif camera_id and store.camera(camera_id):
            cid = camera_id
        else:
            raise HTTPException(400, "Choose an existing camera or name a new one")

        if recorded_at.strip():
            try:
                start = datetime.fromisoformat(recorded_at.strip())
            except ValueError:
                raise HTTPException(400, "recorded_at must be an ISO date-time")
            start = start.astimezone() if start.tzinfo is None else start
        else:
            start = datetime.now().astimezone()

        upload_id = uuid.uuid4().hex[:12]
        dest = upload_dir / f"{upload_id}{ext}"
        written = 0
        with dest.open("wb") as out:
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(413, "File is larger than 4 GB")
                out.write(chunk)

        duration_ms, _fps = probe_video(str(dest))
        if duration_ms <= 0:
            dest.unlink(missing_ok=True)
            raise HTTPException(400, "That file could not be read as a video")

        store.insert_upload({
            "upload_id": upload_id, "camera_id": cid, "filename": Path(file.filename).name[:200],
            "path": dest.as_posix(), "start_ts": int(start.timestamp() * 1000), "duration_ms": duration_ms,
            "mode": mode, "vlm_model": vlm_model.strip() or settings.vlm.model, "status": "queued",
            "frames_total": frames_expected(duration_ms, settings.sample_fps),
            "created_at": int(time.time() * 1000),
        })
        jobs.submit(upload_id)
        return _upload_json(next(u for u in store.uploads() if u["upload_id"] == upload_id))

    @app.post("/api/uploads/{upload_id}/cancel")
    def cancel_upload(upload_id: str):
        if not store.upload(upload_id):
            raise HTTPException(404, "upload not found")
        return {"cancelled": jobs.cancel(upload_id)}

    @app.get("/api/uploads/{upload_id}/video")
    def upload_video(upload_id: str):
        u = store.upload(upload_id)
        if u is None or not Path(u["path"]).is_file():
            raise HTTPException(404, "video not found")
        media = {".mp4": "video/mp4", ".m4v": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime"}
        return FileResponse(u["path"], media_type=media.get(Path(u["path"]).suffix.lower(), "application/octet-stream"))

    from aequitas.web import bodycam
    bodycam.register(app, settings, store, jobs, auth, upload_dir, _upload_json, _total_s, VIDEO_EXTS,
                     MAX_UPLOAD_BYTES)

    # ---------- live CCTV dashboard ----------

    live = LiveManager(settings, store, load_localities(settings.live.localities_file))
    app.state.live = live

    @app.get("/api/live/localities")
    def live_localities():
        return [{"id": loc.id, "name": loc.name,
                 "cameras": [{"id": c.id, "name": c.name, "has_footage": live.has_footage(c.id)} for c in loc.cameras]}
                for loc in live.localities.values()]

    @app.post("/api/live/{locality_id}/start")
    def live_start(locality_id: str, request: Request):
        try:
            live.start(locality_id)
        except KeyError:
            raise HTTPException(404, "unknown locality")
        user = request.state.user
        auth.audit(user["user_id"] if user else None, "live_started", locality_id,
                   ip=request.client.host if request.client else "")
        return live.state(locality_id)

    @app.post("/api/live/stop")
    def live_stop():
        live.stop()
        return {"ok": True}

    @app.get("/api/live/ai")
    def live_ai():
        return {"paused": live.paused}

    @app.post("/api/live/ai")
    def live_ai_set(body: AiSwitch, request: Request):
        paused = body.paused
        live.set_paused(paused)
        user = request.state.user
        auth.audit(user["user_id"] if user else None, "live_ai_paused" if paused else "live_ai_resumed", None,
                   ip=request.client.host if request.client else "")
        return {"paused": live.paused}

    @app.get("/api/live/{locality_id}")
    def live_state(locality_id: str):
        try:
            st = live.state(locality_id)
        except KeyError:
            raise HTTPException(404, "unknown locality")
        st["concerns"] = [_event_json(e, store) for e in st["concerns"]]
        return st

    @app.get("/api/live/cameras/{camera_id}/poster.jpg")
    def live_poster(camera_id: str):
        """A still from the camera's clip, shown in the tile at once so it is never black while the browser's video
        decoder catches up (the GPU is busy with the model). Made once per clip version and cached on disk."""
        cam = live.cameras.get(camera_id)
        if cam is None or cam.is_live or not live.has_footage(camera_id):
            raise HTTPException(404, "no footage for this camera")
        src = Path(cam.file)
        out = settings.data_dir / "posters" / f"{camera_id}-{int(src.stat().st_mtime)}.jpg"
        if not out.is_file():
            import cv2

            cap = cv2.VideoCapture(str(src))
            cap.set(cv2.CAP_PROP_POS_MSEC, 500)  # just past any fade-in
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise HTTPException(404, "could not read the clip")
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (640, round(640 * h / w)))
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return FileResponse(out, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

    @app.get("/api/live/cameras/{camera_id}/video")
    def live_video(camera_id: str):
        cam = live.cameras.get(camera_id)
        if cam is None or cam.is_live or not live.has_footage(camera_id):  # only files named in the config
            raise HTTPException(404, "no footage for this camera")
        return FileResponse(cam.file, media_type="video/mp4")

    @app.post("/api/live/cameras/{camera_id}/frame")
    async def live_frame(camera_id: str, request: Request):
        """One JPEG frame from the browser's camera (source: browser). The page sends them at sample_fps."""
        cam = live.cameras.get(camera_id)
        if cam is None or cam.source != "browser":
            raise HTTPException(404, "not a browser camera")
        body = await request.body()
        if not body or len(body) > 4 * 1024 * 1024:
            raise HTTPException(400, "expected one JPEG frame under 4 MB")
        import cv2
        import numpy as np

        img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "not an image")
        from aequitas.ingest.frame_source import push_frame

        return {"accepted": push_frame(camera_id, img)}

    @app.get("/api/live/cameras/{camera_id}/mjpeg")
    def live_mjpeg(camera_id: str):
        """The frames the pipeline is reading from a live source (webcam/RTSP), as an MJPEG stream. The
        browser cannot open the webcam itself while the pipeline holds it."""
        cam = live.cameras.get(camera_id)
        if cam is None or not cam.is_live:
            raise HTTPException(404, "not a live camera")

        def frames():
            import cv2

            idle = 0
            while idle < 300:  # stop after ~30 s without frames (analysis stopped)
                img = live.latest_frame(camera_id)
                if img is None:
                    idle += 1
                    time.sleep(0.1)
                    continue
                idle = 0
                ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
                time.sleep(0.12)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    # ---------- citizen portal (public) ----------

    reports = ReportStore(store)
    citizen_hits: dict[str, list[float]] = {}

    def _rate_limited(ip: str) -> bool:
        now = time.time()
        recent = [t for t in citizen_hits.get(ip, []) if now - t < 3600]
        citizen_hits[ip] = recent
        if len(recent) >= settings.citizen.reports_per_hour_per_ip:
            return True
        recent.append(now)
        return False

    def _check_authenticity(report_id: str, path: str) -> None:
        try:
            from aequitas.authenticity import analyze_video

            reports.set_authenticity(report_id, analyze_video(path, settings.authenticity))
        except Exception as e:
            log.exception("authenticity check failed for %s", report_id)
            reports.set_authenticity(report_id, {"error": str(e)[:300]})

    def _resume_authenticity() -> None:
        """A check runs in a thread, so a restart mid-check left the report 'checking…' forever.
        Re-run any report that has no result yet, or whose frame check crashed, one at a time."""
        for r in reports.list(limit=500):
            a = r.get("authenticity")
            if a and not ((a.get("frames") or {}).get("error")):
                continue
            up = store.upload(r["upload_id"]) or {}
            if up.get("path") and Path(up["path"]).exists():
                _check_authenticity(r["report_id"], up["path"])
            else:
                reports.set_authenticity(r["report_id"], {"error": "video file missing"})

    threading.Thread(target=_resume_authenticity, daemon=True, name="authenticity-resume").start()

    @app.post("/api/citizen/reports")
    def citizen_report(
        request: Request,
        file: UploadFile = File(...),
        description: str = Form(..., min_length=10, max_length=2000),
        location: str = Form("", max_length=200),
        incident_at: str = Form(""),
        contact: str = Form("", max_length=200),
        consent: bool = Form(False),
        language: str = Form("en", max_length=5),
        urgent: bool = Form(False),
    ):
        """Anyone can report: a video and what happened. Returns the report ID used to check the status."""
        if not consent:
            raise HTTPException(400, "Please confirm the video may be shared with the police")
        ip = request.client.host if request.client else ""
        if _rate_limited(ip):
            raise HTTPException(429, "Too many reports from this connection. Please try again later.")
        ext = Path(file.filename or "").suffix.lower()
        if ext not in VIDEO_EXTS:
            raise HTTPException(400, f"Unsupported file type '{ext}'. Use one of: {', '.join(sorted(VIDEO_EXTS))}")
        when = None
        if incident_at.strip():
            try:
                dt = datetime.fromisoformat(incident_at.strip())
                when = int((dt.astimezone() if dt.tzinfo is None else dt).timestamp() * 1000)
            except ValueError:
                raise HTTPException(400, "incident_at must be an ISO date-time")

        upload_id = uuid.uuid4().hex[:12]
        dest = upload_dir / f"{upload_id}{ext}"
        limit, written, sha = settings.citizen.max_upload_mb * 1024**2, 0, hashlib.sha256()
        with dest.open("wb") as out:
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > limit:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(413, f"The video is larger than {settings.citizen.max_upload_mb} MB")
                sha.update(chunk)
                out.write(chunk)
        duration_ms, _fps = probe_video(str(dest))
        if duration_ms <= 0:
            dest.unlink(missing_ok=True)
            raise HTTPException(400, "That file could not be read as a video")

        cam_id = f"report-{upload_id}"
        store.upsert_camera(CameraCfg(camera_id=cam_id, name="Citizen report", source="citizen"))
        start = when or int(time.time() * 1000)
        store.insert_upload({
            "upload_id": upload_id, "camera_id": cam_id, "filename": "citizen-report" + ext,
            "path": dest.as_posix(), "start_ts": start, "duration_ms": duration_ms,
            "mode": "periodic", "vlm_model": settings.vlm.model, "status": "queued",
            "frames_total": frames_expected(duration_ms, settings.sample_fps),
            "created_at": int(time.time() * 1000),
        })
        lang = detect_language(description, language)
        report_id = reports.create(upload_id, cam_id, description.strip(), sha.hexdigest(),
                                         location.strip() or None, when, contact.strip() or None, lang, urgent)
        auth.audit(None, "report_submitted", report_id, {"sha256": sha.hexdigest(), "bytes": written}, ip)
        jobs.submit(upload_id)
        threading.Thread(target=_check_authenticity, args=(report_id, dest.as_posix()), daemon=True).start()
        return {"report_id": report_id, "status": PUBLIC_STATUS["received"],
                "message": "Keep the report ID to check the status."}

    @app.get("/api/citizen/reports/{report_id}")
    def citizen_status(report_id: str):
        """Public: the status and dates only, never the description, video or notes."""
        r = reports.get(report_id.strip().upper())
        if r is None:
            raise HTTPException(404, "No report with that ID")
        return {"report_id": r["report_id"], "status": PUBLIC_STATUS[r["status"]],
                "submitted_at": r["created_at"], "updated_at": r["updated_at"]}

    @app.post("/api/citizen/assist")
    def citizen_assist(body: AssistBody):
        """The portal's emergency assistant (rule-based: fixed helpline numbers, never promises help is sent)."""
        return emergency.reply(body.lang, body.message, body.choice)

    # ---------- police review of citizen reports ----------

    def _report_json(r: dict, detail: bool = False) -> dict:
        up = store.upload(r["upload_id"]) or {}
        out = {**r, "video_url": f"/api/uploads/{r['upload_id']}/video",
               "analysis": {"status": up.get("status"), "events_found": up.get("events_found"),
                            "total_s": _total_s(up) if up else None}}
        if detail:
            f = QueryFilter(camera_ids=[r["camera_id"]], statuses=list(Status), limit=100, sort="oldest")
            out["events"] = [_event_json(e, store) for e in store.query(f)]
            out["stages"] = store.stage_summary(r["upload_id"])
            out["audit"] = auth.audit_trail(r["report_id"])
        return out

    @app.get("/api/reports")
    def list_reports(status: str | None = None):
        return [_report_json(r) for r in reports.list(status)]

    ACTIVE = ("received", "under_review", "action_taken")

    @app.get("/api/reports/active")
    def active_reports():
        """The work queue: reports still open, urgent ones first, then oldest first (longest waiting at the top),
        with the incident labels the AI found so the officer can triage without opening each one."""
        rows = [r for r in reports.list(limit=500) if r["status"] in ACTIVE]
        out = []
        for r in sorted(rows, key=lambda r: (not r["urgent"], r["created_at"])):  # urgent first
            j = _report_json(r)
            f = QueryFilter(camera_ids=[r["camera_id"]], statuses=list(Status), limit=50, sort="oldest")
            evs = [e for e in store.query(f) if e["category"] != "none"]
            t0 = (store.upload(r["upload_id"]) or {}).get("start_ts") or 0
            j["categories"] = sorted({e["category"] for e in evs})
            j["findings"] = [{"category": e["category"], "status": e["status"],
                              "at_s": round((e["start_ts"] - t0) / 1000, 1),
                              "summary": e.get("summary")} for e in evs[:6]]
            out.append(j)
        return {"reports": out, "counts": {s: sum(r["status"] == s for r in rows) for s in ACTIVE}}

    @app.get("/api/reports/{report_id}")
    def get_report(report_id: str, request: Request):
        r = reports.get(report_id)
        if r is None:
            raise HTTPException(404, "report not found")
        user = request.state.user
        auth.audit(user["user_id"] if user else None, "report_viewed", report_id,
                   ip=request.client.host if request.client else "")
        return _report_json(r, detail=True)

    def _translator():
        if getattr(app.state, "translator", None) is None:
            llm = getattr(service.parser, "llm", None)
            if llm is None:
                from aequitas.understand.vlm_client import make_client

                llm = make_client(settings.chat.endpoint(settings.vlm))
            app.state.translator = llm
        return app.state.translator

    @app.post("/api/reports/{report_id}/translate")
    def translate_report(report_id: str, request: Request, refresh: bool = False):
        """English translation of a Hindi/Kannada report by the local LLM. The original stays as written;
        the translation is cached on the report and timed."""
        r = reports.get(report_id)
        if r is None:
            raise HTTPException(404, "report not found")
        if r["translation"] and not refresh:
            return {**r["translation"], "cached": True}
        llm = _translator()
        with GATE.interactive():
            english, ms = translate_to_english(llm, r["description"], r["language"])
        if english is None:
            raise HTTPException(502, f"The translation model ({llm.model}) did not answer. Is it running?")
        result = {"text": english, "ms": round(ms), "model": f"{llm.cfg.backend}:{llm.model}",
                  "from": r["language"], "from_name": LANGUAGES.get(r["language"], r["language"])}
        reports.set_translation(report_id, result)
        user = request.state.user
        auth.audit(user["user_id"] if user else None, "report_translated", report_id,
                   {"ms": result["ms"], "model": result["model"]}, request.client.host if request.client else "")
        return {**result, "cached": False}

    # ---------- one video in depth: citizen reports reviewed like bodycam recordings, plus a per-video chat ----------


    def _report_upload(report_id: str) -> tuple[dict, dict]:
        r = reports.get(report_id)
        u = store.upload(r["upload_id"]) if r else None
        if r is None or u is None:
            raise HTTPException(404, "report not found")
        return r, u

    def _video_detail(u: dict) -> dict:
        tl = timeline(u, store.windows_for(u["camera_id"]), store.events_for(u["camera_id"]))
        return {**_upload_json(u), **tl, "video_url": f"/api/uploads/{u['upload_id']}/video",
                "officer": u.get("officer"), "view": u.get("view"),
                "summary": json.loads(u["summary_json"]) if u.get("summary_json") else None,
                "summary_s": round(u["summary_ms"] / 1000, 2) if u.get("summary_ms") else None}

    def _ask_video(u: dict, body: VideoAskBody, request: Request, target: str, citizen_text: str | None) -> dict:
        d = _video_detail(u)
        llm = _translator()
        relevant: list[int] = []
        if service.index is not None:  # this video's windows nearest in meaning to the question (Qdrant)
            try:
                hits = service.index.search(service.index.embed_query(body.question), "window",
                                            store.window_ids([u["camera_id"]]), limit=5)
                relevant = [h.ref_id for h in hits if h.score >= settings.chat.vector.min_score]
            except Exception as e:
                log.warning("vector search for %s failed: %s", target, e)
        try:
            with GATE.interactive():
                res = answer_video(llm, body.question, d["windows"], d["events"], d["summary"], d["duration_s"],
                                   citizen_text, [h.model_dump() for h in body.history], relevant)
        except Exception as e:
            log.warning("video question failed for %s: %s", target, e)
            raise HTTPException(503, f"The chat model ({llm.model}) did not answer. Is it running?")
        user = request.state.user
        auth.audit(user["user_id"] if user else None, "video_question", target,
                   {"question": body.question[:200], "ms": res["ms"]}, request.client.host if request.client else "")
        return {**res, "model": llm.model}

    @app.get("/api/reports/{report_id}/analysis")
    def report_analysis(report_id: str):
        """The report's video as a review timeline: every window's summary, key moments with keyframes, overview."""
        _r, u = _report_upload(report_id)
        return _video_detail(u)

    @app.post("/api/reports/{report_id}/summary")
    def report_summary(report_id: str, request: Request):
        _r, u = _report_upload(report_id)
        if u["status"] != "done":
            raise HTTPException(409, "The video is still being analysed")
        if make_summary(settings, store, u) is None:
            raise HTTPException(503, "The AI could not write a summary (no window summaries, or the model is offline)")
        user = request.state.user
        auth.audit(user["user_id"] if user else None, "report_summary_written", report_id,
                   ip=request.client.host if request.client else "")
        return _video_detail(store.upload(u["upload_id"]))

    @app.post("/api/reports/{report_id}/events/{event_id}/review")
    def report_review(report_id: str, event_id: int, body: ReviewBody, request: Request):
        _r, u = _report_upload(report_id)
        ev = store.event(event_id)
        if ev is None or ev["camera_id"] != u["camera_id"]:
            raise HTTPException(404, "event not found in this report")
        user = request.state.user
        store.review_event(event_id, body.decision, (user["username"] if user else "officer") if body.decision else None)
        auth.audit(user["user_id"] if user else None, "report_event_reviewed", f"{report_id}/{event_id}",
                   {"decision": body.decision}, request.client.host if request.client else "")
        return reviewed(u, event_id, store)

    @app.post("/api/reports/{report_id}/ask")
    def report_ask(report_id: str, body: VideoAskBody, request: Request):
        """Chat about this report's video only, answered from its analysis (and the citizen's words, unverified)."""
        r, u = _report_upload(report_id)
        text = (r.get("translation") or {}).get("text") or r["description"]
        return _ask_video(u, body, request, report_id, text)

    @app.post("/api/bodycam/{upload_id}/ask")
    def bodycam_ask(upload_id: str, body: VideoAskBody, request: Request):
        u = store.upload(upload_id)
        if u is None or u.get("kind") != "bodycam":
            raise HTTPException(404, "recording not found")
        return _ask_video(u, body, request, upload_id, None)

    @app.patch("/api/reports/{report_id}")
    def update_report(report_id: str, body: ReportUpdate, request: Request):
        if reports.get(report_id) is None:
            raise HTTPException(404, "report not found")
        try:
            reports.update(report_id, body.status, body.notes)
        except ValueError as e:
            raise HTTPException(400, str(e))
        user = request.state.user
        auth.audit(user["user_id"] if user else None, "report_updated", report_id,
                   body.model_dump(exclude_none=True), request.client.host if request.client else "")
        return _report_json(reports.get(report_id), detail=True)

    if (DIST / "index.html").is_file():
        # The React app (frontend/, `npm run build`). Unknown non-API paths return the app, which routes them.
        app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404, "not found")
            f = (DIST / path).resolve()
            if path and f.is_file() and DIST.resolve() in f.parents:
                return FileResponse(f)
            return FileResponse(DIST / "index.html")
    else:  # React app not built: the older plain-JS pages
        app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


def serve(settings: Settings, service: ChatService, camera_cfgs: list[CameraCfg], port: int = 8000) -> None:
    import uvicorn

    app = create_app(settings, service, camera_cfgs)
    print(f"Aequitas is running at http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
