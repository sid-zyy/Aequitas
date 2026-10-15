"""Minerva web app: upload videos for analysis, browse events, and ask questions.
Serves a JSON API under /api and the single-page frontend from ./static. Binds to localhost."""

from __future__ import annotations

import logging
import re
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from minerva.chat.answer import ChatService
from minerva.config import CameraCfg, Settings, resolve_device
from minerva.store.db import EventStore
from minerva.store.query import QueryFilter
from minerva.types import Category, Status
from minerva.web.jobs import JobManager, frames_expected, probe_video

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MAX_UPLOAD_BYTES = 4 * 1024**3


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=500)


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
    }


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

    app = FastAPI(title="Minerva", docs_url="/api/docs", redoc_url=None)

    @app.middleware("http")
    async def revalidate_frontend(request, call_next):
        # The frontend changes with the code; make browsers revalidate it (ETags keep this cheap).
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    # ---------- status ----------

    @app.get("/api/status")
    def status():
        vlm_online, models = False, []
        try:
            from minerva.understand.vlm_client import make_client

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

    @app.get("/api/stats")
    def stats():
        day0 = datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
        return store.stats(int(day0.timestamp() * 1000))

    @app.get("/api/cameras")
    def cameras():
        return [{"camera_id": c["camera_id"], "name": c["name"], "aliases": c["aliases"]} for c in store.cameras()]

    # ---------- events ----------

    @app.get("/api/events")
    def events(camera: str | None = None, category: Category | None = None,
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

    # ---------- chat ----------

    @app.post("/api/ask")
    def ask(body: AskBody):
        t0 = time.perf_counter()
        ans = service.ask(body.question.strip())
        f = ans.parsed.filter if ans.parsed else None
        return {
            "text": ans.text,
            "summary": ans.summary,
            "total": ans.total,
            "intent": f.intent if f else "list",
            "warnings": ans.parsed.warnings if ans.parsed else [],
            "parsed_with": ans.parsed.method if ans.parsed else None,
            "filter": f.model_dump(mode="json") if f else None,
            "events": [_event_json(r, store) for r in ans.rows],
            "elapsed_ms": round((time.perf_counter() - t0) * 1000),
        }

    # ---------- uploads ----------

    def _upload_json(u: dict) -> dict:
        live = jobs.progress(u["upload_id"])
        done = live.get("frames_done", u["frames_done"])
        total = u["frames_total"] or 0
        status = u["status"]
        if status == "processing" and live.get("phase") == "verifying":
            status = "verifying"
        return {
            "id": u["upload_id"], "camera_id": u["camera_id"], "camera_name": u.get("camera_name"),
            "filename": u["filename"], "status": status, "mode": u["mode"], "vlm_model": u["vlm_model"],
            "frames_done": done, "frames_total": total,
            "progress": min(1.0, done / total) if total else 0.0,
            "vlm_backlog": live.get("vlm_backlog", 0), "events_found": u["events_found"],
            "duration_s": (u["duration_ms"] or 0) / 1000, "start_ts": u["start_ts"],
            "error": u["error"], "created_at": u["created_at"], "finished_at": u["finished_at"],
        }

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

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


def serve(settings: Settings, service: ChatService, camera_cfgs: list[CameraCfg], port: int = 8000) -> None:
    import uvicorn

    app = create_app(settings, service, camera_cfgs)
    print(f"Minerva is running at http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
