"""Bodycam review: an officer's recording through the normal pipeline, reviewed as a timeline.

Each recording gets its own camera, so its windows and events are exactly the recording's. Windows are
summarised with the close-view prompt (people near the camera: hands, arms, posture; never faces). Every
AI-detected moment waits for an officer's decision (confirm / dismiss); the report export says which is which."""

from __future__ import annotations

import html
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from aequitas.config import CameraCfg, Settings
from aequitas.store.db import EventStore
from aequitas.taxonomy import label as cat_label
from aequitas.web.jobs import frames_expected, probe_video


class ReviewBody(BaseModel):
    decision: Literal["confirmed", "dismissed"] | None  # None clears the decision


def mmss(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def timeline(upload: dict, windows: list[dict], events: list[dict]) -> dict:
    """Window and event times relative to the start of the recording, in seconds."""
    t0 = upload["start_ts"]
    return {
        "windows": [{"t": round((w["start_ts"] - t0) / 1000, 2), "end": round((w["end_ts"] - t0) / 1000, 2),
                     "label": w["label"], "summary": w["summary"], "source": w["source"],
                     "violence": w["violence"]} for w in windows],
        "events": [{"id": e["event_id"], "t": round((e["start_ts"] - t0) / 1000, 2),
                    "end": round((e["end_ts"] - t0) / 1000, 2), "category": e["category"], "status": e["status"],
                    "confidence": e["confidence"], "summary": e["summary"], "review": e.get("review"),
                    "reviewed_by": e.get("reviewed_by"), "keyframe_url": f"/api/events/{e['event_id']}/keyframe"}
                   for e in events],
    }


def report_html(upload: dict, tl: dict, total_s: float | None) -> str:
    """A printable report. Officer decisions are shown next to the AI's; unreviewed moments are marked."""
    e = html.escape
    when = datetime.fromtimestamp(upload["start_ts"] / 1000).astimezone().strftime("%d %b %Y, %H:%M")
    moments = [ev for ev in tl["events"] if ev["review"] != "dismissed"]
    dismissed = [ev for ev in tl["events"] if ev["review"] == "dismissed"]
    unreviewed = sum(ev["review"] is None for ev in tl["events"])

    def decision(ev):
        return {"confirmed": f"Confirmed by {e(ev['reviewed_by'] or 'officer')}",
                "dismissed": f"Dismissed by {e(ev['reviewed_by'] or 'officer')}"}.get(ev["review"], "Not reviewed")

    def rows(evs):
        return "".join(f"<tr><td>{mmss(ev['t'])}</td><td>{e(cat_label(ev['category']))}</td>"
                       f"<td>{e(ev['status'])}</td><td>{decision(ev)}</td><td>{e(ev['summary'] or '')}</td></tr>"
                       for ev in evs) or '<tr><td colspan="5">None</td></tr>'

    log_rows = "".join(f"<tr><td>{mmss(w['t'])}</td><td>{e(cat_label(w['label']) if w['label'] else 'failed')}</td>"
                       f"<td>{e(w['summary'] or '')}</td></tr>" for w in tl["windows"])
    head = "<tr><th>Time</th><th>Category</th><th>AI status</th><th>Officer</th><th>What the AI saw</th></tr>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Bodycam report {e(upload['upload_id'])}</title>
<style>body{{font:14px/1.5 system-ui,sans-serif;margin:32px;color:#111}}h1{{font-size:20px}}h2{{font-size:16px;margin-top:28px}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #bbb;padding:4px 8px;text-align:left;vertical-align:top}}
th{{background:#eee}}.warn{{border:1px solid #c80;background:#fff6e0;padding:8px 12px}}dl{{display:grid;grid-template-columns:max-content 1fr;gap:2px 16px}}</style>
</head><body>
<h1>Bodycam recording review</h1>
<dl><dt>Recording</dt><dd>{e(upload['filename'])}</dd><dt>Officer / unit</dt><dd>{e(upload.get('officer') or '-')}</dd>
<dt>Recorded</dt><dd>{when}</dd><dt>Length</dt><dd>{mmss((upload['duration_ms'] or 0) / 1000)}</dd>
<dt>Analysed in</dt><dd>{f'{total_s:.1f} s' if total_s is not None else '-'} (model {e(upload['vlm_model'] or '')})</dd></dl>
<p class="warn">AI-generated draft. The AI describes what is visible and never identifies people. Every moment must
be checked against the video by an officer; {unreviewed} of {len(tl['events'])} are not reviewed yet.</p>
<h2>Key moments</h2><table>{head}{rows(moments)}</table>
<h2>Dismissed by an officer</h2><table>{head}{rows(dismissed)}</table>
<h2>Full timeline (every analysed {e('4 s')} window)</h2><table><tr><th>Time</th><th>Label</th><th>Summary</th></tr>{log_rows}</table>
<p style="color:#666;margin-top:24px">Generated {datetime.now().astimezone():%d %b %Y, %H:%M} by Aequitas.</p>
</body></html>"""


def register(app: FastAPI, settings: Settings, store: EventStore, jobs, auth, upload_dir: Path,
             upload_json, total_s, video_exts: set[str], max_bytes: int) -> None:
    def _user(request: Request) -> dict | None:
        return request.state.user

    def _audit(request: Request, action: str, target: str, detail=None) -> None:
        u = _user(request)
        auth.audit(u["user_id"] if u else None, action, target, detail,
                   request.client.host if request.client else "")

    def _json(u: dict) -> dict:
        cam = store.camera(u["camera_id"]) or {}
        return {**upload_json({**u, "camera_name": cam.get("name")}), "officer": u.get("officer"),
                "view": u.get("view")}

    def _get(upload_id: str) -> dict:
        u = store.upload(upload_id)
        if u is None or u.get("kind") != "bodycam":
            raise HTTPException(404, "recording not found")
        return u

    @app.get("/api/bodycam")
    def bodycam_list():
        return [_json(u) for u in store.uploads_of_kind("bodycam")]

    @app.post("/api/bodycam")
    def bodycam_upload(request: Request, file: UploadFile = File(...), officer: str = Form("", max_length=80),
                       recorded_at: str = Form(""), view: Literal["close", "cctv"] = Form("close")):
        ext = Path(file.filename or "").suffix.lower()
        if ext not in video_exts:
            raise HTTPException(400, f"Unsupported file type '{ext}'. Use one of: {', '.join(sorted(video_exts))}")
        if recorded_at.strip():
            try:
                dt = datetime.fromisoformat(recorded_at.strip())
            except ValueError:
                raise HTTPException(400, "recorded_at must be an ISO date-time")
            start = int((dt.astimezone() if dt.tzinfo is None else dt).timestamp() * 1000)
        else:
            start = int(time.time() * 1000)
        upload_id = uuid.uuid4().hex[:12]
        dest = upload_dir / f"{upload_id}{ext}"
        written = 0
        with dest.open("wb") as out:
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(413, "File is too large")
                out.write(chunk)
        duration_ms, _fps = probe_video(str(dest))
        if duration_ms <= 0:
            dest.unlink(missing_ok=True)
            raise HTTPException(400, "That file could not be read as a video")
        cam_id = f"bodycam-{upload_id}"
        name = f"Bodycam · {officer.strip()}" if officer.strip() else "Bodycam"
        store.upsert_camera(CameraCfg(camera_id=cam_id, name=name[:60], source="bodycam"))
        store.insert_upload({
            "upload_id": upload_id, "camera_id": cam_id, "filename": Path(file.filename).name[:200],
            "path": dest.as_posix(), "start_ts": start, "duration_ms": duration_ms, "mode": "periodic",
            "vlm_model": settings.vlm.model, "status": "queued",
            "frames_total": frames_expected(duration_ms, settings.sample_fps),
            "created_at": int(time.time() * 1000), "kind": "bodycam", "view": view,
            "officer": officer.strip() or None,
        })
        _audit(request, "bodycam_uploaded", upload_id, {"bytes": written, "officer": officer.strip() or None})
        jobs.submit(upload_id)
        return _json(store.upload(upload_id))

    @app.get("/api/bodycam/{upload_id}")
    def bodycam_detail(upload_id: str, request: Request):
        u = _get(upload_id)
        _audit(request, "bodycam_viewed", upload_id)
        tl = timeline(u, store.windows_for(u["camera_id"]), store.events_for(u["camera_id"]))
        return {**_json(u), **tl, "video_url": f"/api/uploads/{upload_id}/video"}

    @app.post("/api/bodycam/{upload_id}/events/{event_id}/review")
    def bodycam_review(upload_id: str, event_id: int, body: ReviewBody, request: Request):
        u = _get(upload_id)
        ev = store.event(event_id)
        if ev is None or ev["camera_id"] != u["camera_id"]:
            raise HTTPException(404, "event not found in this recording")
        user = _user(request)
        by = (user["username"] if user else "officer") if body.decision else None
        store.review_event(event_id, body.decision, by)
        _audit(request, "bodycam_event_reviewed", f"{upload_id}/{event_id}", {"decision": body.decision})
        return timeline(u, [], [store.event(event_id)])["events"][0]

    @app.get("/api/bodycam/{upload_id}/report", response_class=HTMLResponse)
    def bodycam_report(upload_id: str, request: Request):
        u = _get(upload_id)
        _audit(request, "bodycam_report_exported", upload_id)
        tl = timeline(u, store.windows_for(u["camera_id"]), store.events_for(u["camera_id"]))
        return HTMLResponse(report_html(u, tl, total_s(u)), headers={
            "Content-Disposition": f'attachment; filename="bodycam-{upload_id}.html"'})
