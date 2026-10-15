"""QueryFilter: the structured query the chatbot produces. Code, not the LLM, turns it into SQL."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from minerva.types import Category, Status


class QueryFilter(BaseModel):
    categories: list[Category] = Field(default_factory=list)  # empty = all
    subtypes: list[str] = Field(default_factory=list)
    camera_ids: list[str] = Field(default_factory=list)  # empty = all
    start: datetime | None = None  # timezone-aware
    end: datetime | None = None
    statuses: list[Status] = Field(default_factory=lambda: [Status.CONFIRMED])
    min_confidence: float = 0.0
    text: str | None = None
    sort: Literal["newest", "oldest", "confidence"] = "newest"
    limit: int = Field(20, ge=1, le=100)
    intent: Literal["list", "count"] = "list"


def to_epoch_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.astimezone()  # naive = local time
    return int(dt.timestamp() * 1000)


def fts_expression(text: str) -> str | None:
    """Turn free text into a safe FTS5 OR-query of quoted words."""
    words = [w for w in re.findall(r"\w+", text.lower()) if len(w) > 2]
    return " OR ".join(f'"{w}"' for w in words) or None


def build_sql(f: QueryFilter, count: bool = False) -> tuple[str, list]:
    where: list[str] = []
    params: list = []

    def in_clause(column: str, values: list) -> None:
        where.append(f"{column} IN ({','.join('?' * len(values))})")
        params.extend(values)

    if f.categories:
        in_clause("e.category", [c.value for c in f.categories])
    if f.subtypes:
        in_clause("e.subtype", f.subtypes)
    if f.camera_ids:
        in_clause("e.camera_id", f.camera_ids)
    if f.statuses:
        in_clause("e.status", [s.value for s in f.statuses])
    # An event matches a window if it overlaps it at all.
    if f.start is not None:
        where.append("e.end_ts >= ?")
        params.append(to_epoch_ms(f.start))
    if f.end is not None:
        where.append("e.start_ts <= ?")
        params.append(to_epoch_ms(f.end))
    if f.min_confidence > 0:
        where.append("e.confidence >= ?")
        params.append(f.min_confidence)
    expr = fts_expression(f.text) if f.text else None
    if expr:
        where.append("e.event_id IN (SELECT rowid FROM events_fts WHERE events_fts MATCH ?)")
        params.append(expr)

    clause = (" WHERE " + " AND ".join(where)) if where else ""
    if count:
        return f"SELECT COUNT(*) FROM events e{clause}", params

    order = {
        "newest": "e.start_ts DESC",
        "oldest": "e.start_ts ASC",
        "confidence": "e.confidence DESC, e.start_ts DESC",
    }[f.sort]
    sql = (
        "SELECT e.*, c.name AS camera_name FROM events e "
        f"LEFT JOIN cameras c ON c.camera_id = e.camera_id{clause} ORDER BY {order} LIMIT ?"
    )
    return sql, params + [f.limit]
