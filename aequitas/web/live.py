"""Live CCTV dashboard backend: localities with camera points, one locality analysed at a time.

Analysing every camera in the city at once would not fit one laptop GPU, so selecting a locality
starts the pipeline (periodic mode) on that locality's cameras and stops the previous one. Each
camera's footage loops like a live feed; every analysed window is stored, so a tile always shows the
latest understanding and how far behind live it is."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import yaml
from pydantic import BaseModel

from aequitas.config import CameraCfg, Settings
from aequitas.store.db import EventStore
from aequitas.store.query import QueryFilter
from aequitas.types import Status

log = logging.getLogger(__name__)


class LiveCamera(BaseModel):
    id: str
    name: str
    file: str | None = None  # demo footage, looped
    source: str | None = None  # a live source instead: "browser" (the page's camera), webcam index "0", rtsp://
    view: str = "cctv"  # "close" for a webcam near the people (gestures described in detail)
    window_s: float | None = None  # shorter windows react faster (live demo)

    @property
    def is_live(self) -> bool:
        return self.source is not None


class Locality(BaseModel):
    id: str
    name: str
    cameras: list[LiveCamera]


def load_localities(path: str | Path) -> list[Locality]:
    p = Path(path)
    if not p.exists():
        return []
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return [Locality.model_validate(x) for x in data.get("localities", [])]


class LiveManager:
    def __init__(self, settings: Settings, store: EventStore, localities: list[Locality]):
        self.settings = settings
        self.store = store
        self.localities = {loc.id: loc for loc in localities}
        self.cameras = {c.id: c for loc in localities for c in loc.cameras}
        self._lock = threading.Lock()
        self.active: str | None = None
        self.pipeline = None
        self.error: str | None = None
        self.started_at: float | None = None
        for loc in localities:  # register the camera points so events and chat know their names
            for c in loc.cameras:
                store.upsert_camera(CameraCfg(camera_id=c.id, name=c.name, aliases=[f"{loc.name} {c.name}"],
                                              source=c.source or c.file or ""))

    def has_footage(self, camera_id: str) -> bool:
        c = self.cameras.get(camera_id)
        if c is None:
            return False
        return c.is_live or (c.file is not None and Path(c.file).is_file())

    def start(self, locality_id: str) -> None:
        if locality_id not in self.localities:
            raise KeyError(locality_id)
        with self._lock:
            if self.active == locality_id and self.pipeline is not None:
                return
            self._stop_locked()
            loc = self.localities[locality_id]
            cams = [CameraCfg(camera_id=c.id, name=c.name, source=c.source or c.file, realtime=True,
                              loop=not c.is_live, view=c.view, window_s=c.window_s)
                    for c in loc.cameras if self.has_footage(c.id)]
            self.active, self.error, self.started_at = locality_id, None, time.time()
            if not cams:
                self.error = f"No footage found for {loc.name} (expected files under data/footage/)"
                return
            s = self.settings.model_copy(deep=True)
            s.analysis.mode = "periodic"
            try:
                from aequitas.pipeline import Pipeline

                self.pipeline = Pipeline(s, cams, self.store, run_id=f"live-{locality_id}")
            except Exception as e:  # VLM not reachable, etc.
                self.error = str(e)[:300]
                self.pipeline = None
                return
            threading.Thread(target=self.pipeline.run, name=f"live-{locality_id}", daemon=True).start()
            log.info("live analysis started for %s (%d cameras)", loc.name, len(cams))

    def _stop_locked(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
        self.pipeline = None
        self.active = None

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def latest_frame(self, camera_id: str):
        p = self.pipeline
        return p.latest_frames.get(camera_id) if p is not None else None

    def state(self, locality_id: str) -> dict:
        loc = self.localities.get(locality_id)
        if loc is None:
            raise KeyError(locality_id)
        cams = []
        for c in loc.cameras:
            recent = self.store.recent_windows(c.id, 5)
            latest = recent[0] if recent else None
            cams.append({
                "id": c.id, "name": c.name, "has_footage": self.has_footage(c.id), "live_source": c.is_live,
                "view": c.view, "browser_camera": c.source == "browser",
                # files play in a <video>; live sources stream the frames the pipeline reads (MJPEG)
                "video_url": f"/api/live/cameras/{c.id}/{'mjpeg' if c.is_live else 'video'}",
                "latest": latest, "recent": recent,
                "behind_live_s": round(latest["lag_ms"] / 1000, 1) if latest and latest["lag_ms"] else None,
            })
        f = QueryFilter(camera_ids=[c.id for c in loc.cameras], statuses=[Status.CONFIRMED, Status.UNVERIFIED],
                        limit=30)
        concerns = self.store.query(f)
        return {
            "locality": {"id": loc.id, "name": loc.name},
            "running": self.active == locality_id and self.pipeline is not None,
            "error": self.error if self.active == locality_id else None,
            "cameras": cams, "concerns": concerns,
            "model": {"backend": self.settings.vlm.backend, "name": self.settings.vlm.model,
                      "concurrency": self.settings.vlm.concurrency},
        }
