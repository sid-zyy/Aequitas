"""Builds the 2x2 keyframe grid the VLM sees and the event's stored keyframe."""

from __future__ import annotations

import cv2
import numpy as np

from aequitas.config import Zones
from aequitas.ingest.ring_buffer import BufferedFrame
from aequitas.types import Candidate, Category

OBJECT_ID_OFFSET = 1_000_000
HIGHLIGHT = (0, 0, 255)  # red, BGR
ZONE = (0, 200, 255)  # amber


def track_label(tid: int) -> str:
    return f"B{tid - OBJECT_ID_OFFSET}" if tid >= OBJECT_ID_OFFSET else f"P{tid}"


def _letterbox(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = size / max(h, w)
    resized = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y, x = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def _draw_overlay(frame: BufferedFrame, cand: Candidate | None, zones: Zones) -> np.ndarray:
    img = frame.image.copy()
    if cand is None:  # periodic mode: the VLM sees the raw scene
        return img
    h, w = img.shape[:2]
    thick = max(2, w // 320)
    if cand.category == Category.TAILGATING and zones.entry_line is not None:
        p1 = tuple(int(v) for v in (zones.entry_line.p1[0] * w, zones.entry_line.p1[1] * h))
        p2 = tuple(int(v) for v in (zones.entry_line.p2[0] * w, zones.entry_line.p2[1] * h))
        cv2.line(img, p1, p2, ZONE, thick)
    polys = []
    if cand.category == Category.CROWD_INCIDENT and zones.crowd_zone is not None:
        polys.append(zones.crowd_zone.polygon)
    if cand.subtype == "loitering" and cand.zone in zones.watch_zones:
        polys.append(zones.watch_zones[cand.zone])
    for poly in polys:
        pts = np.array([(x * w, y * h) for x, y in poly], dtype=np.int32)
        cv2.polylines(img, [pts], True, ZONE, thick)
    for tid in cand.track_ids:
        box = frame.boxes.get(tid)
        if box is None:
            continue
        x1, y1, x2, y2 = (int(v * frame.scale) for v in box)
        cv2.rectangle(img, (x1, y1), (x2, y2), HIGHLIGHT, thick)
        cv2.putText(img, track_label(tid), (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * thick / 2 + 0.3,
                    HIGHLIGHT, thick)
    return img


def make_grid(frames: list[BufferedFrame], cand: Candidate | None, zones: Zones, tile_px: int = 448) -> np.ndarray:
    """Up to 4 frames, oldest first, tiled 2x2. Each tile is labelled with its time relative to the
    trigger (or to the last frame when there is no candidate)."""
    ref_ms = cand.trigger_ms if cand is not None else (frames[-1].ts_ms if frames else 0)
    tiles = []
    for f in frames[:4]:
        tile = _letterbox(_draw_overlay(f, cand, zones), tile_px)
        label = f"{(f.ts_ms - ref_ms) / 1000:+.1f}s"
        cv2.putText(tile, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(tile, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(tile)
    while len(tiles) < 4:
        tiles.append(tiles[-1].copy() if tiles else np.zeros((tile_px, tile_px, 3), np.uint8))
    return np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])


def encode_jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def window_detail(frames: list[BufferedFrame], size: int = 64) -> float:
    """Picture detail of a window: the highest grayscale standard deviation (0-255) of its frames. A covered or
    black camera is ~0-2; the demo clips' frames are 41 or more. Below a few, there is nothing to look at."""
    if not frames:
        return 0.0
    return float(max(cv2.resize(cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY), (size, size),
                                interpolation=cv2.INTER_AREA).std() for f in frames))


def window_motion(frames: list[BufferedFrame], size: int = 64) -> float:
    """How much changes across a window, 0-1: the largest mean absolute difference between consecutive
    picked frames, on small grayscale copies (cheap, and blind to compression noise at this size)."""
    small = [cv2.resize(cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY), (size, size), interpolation=cv2.INTER_AREA)
             .astype(np.float32) for f in frames]
    if len(small) < 2:
        return 1.0  # nothing to compare: never skip
    return float(max(np.abs(a - b).mean() for a, b in zip(small, small[1:])) / 255.0)
