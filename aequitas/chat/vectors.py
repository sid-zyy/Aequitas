"""The chatbot's vector database: every stage-1 summary (events and every analysed window), embedded and kept in
Qdrant, so a question is matched by meaning ("a brawl" finds "two men punch each other") and not only by keyword.

SQLite stays the source of truth. Qdrant holds one point per summary: the embedding plus kind, ID, camera and time.
The index catches up from SQLite on its own (`sync`), so it can be deleted at any time and is rebuilt on start.
Embedded, on-disk Qdrant by default (data_dir/vectors); `chat.vector.url` points the same code at a Qdrant server.

Search is always inside the structured filter: SQL first picks the events the filter allows (category, place,
time, status), and Qdrant ranks only those, so a semantic match can never break the officer's filter."""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from aequitas.config import VectorCfg
from aequitas.store.db import EventStore

log = logging.getLogger(__name__)

KINDS = ("event", "window")
_NS = uuid.UUID("6f1d2c4e-5a7b-4c3d-9e8f-a1b2c3d4e5f6")


class Embedder(Protocol):
    name: str
    dim: int

    def embed_docs(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    """An ONNX sentence-embedding model on the CPU (fastembed). bge-base: ~10 ms per question on the laptop."""

    def __init__(self, model: str, cache_dir: Path, threads: int = 2):
        from fastembed import TextEmbedding

        self.m = TextEmbedding(model, cache_dir=str(cache_dir), threads=threads)
        self.name = model
        self.dim = len(self.embed_query("warm up"))

    def embed_docs(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.m.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.m.query_embed([text]))).tolist()


class OllamaEmbedder:
    """An Ollama embedding model, kept on the CPU (num_gpu 0) so it never evicts the VLM from the 6 GB GPU."""

    def __init__(self, host: str, model: str):
        import ollama

        self.c = ollama.Client(host=host)
        self.name = model
        self.dim = len(self.embed_query("warm up"))

    def embed_docs(self, texts: list[str]) -> list[list[float]]:
        return self.c.embed(model=self.name, input=texts, options={"num_gpu": 0}, keep_alive="30m")["embeddings"]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_docs([text])[0]


def make_embedder(cfg: VectorCfg, data_dir: Path, ollama_host: str) -> Embedder:
    if cfg.embedder == "ollama":
        return OllamaEmbedder(ollama_host, cfg.model)
    return FastEmbedder(cfg.model, cfg.cache_dir, cfg.threads)


@dataclass
class Hit:
    kind: str
    ref_id: int
    score: float  # cosine similarity, -1..1


def point_id(kind: str, ref_id: int) -> str:
    return str(uuid.uuid5(_NS, f"{kind}:{ref_id}"))


class VectorIndex:
    def __init__(self, store: EventStore, embedder: Embedder, path: Path | str | None = None,
                 url: str | None = None):
        from qdrant_client import QdrantClient, models

        self.m = models
        self.store = store
        self.embedder = embedder
        self.where = url or (str(path) if path else ":memory:")
        if url:
            self.client = QdrantClient(url=url)
        elif path:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))
        else:
            self.client = QdrantClient(location=":memory:")
        # One collection per embedding model: switching models never mixes vectors of different spaces.
        self.collection = "summaries_" + re.sub(r"[^a-z0-9]+", "_", embedder.name.lower()).strip("_")
        self._lock = threading.RLock()
        with self._lock:
            if not self.client.collection_exists(self.collection):
                self.client.create_collection(self.collection, vectors_config=models.VectorParams(
                    size=embedder.dim, distance=models.Distance.COSINE))
        self.high = {k: 0 for k in KINDS}  # highest SQLite ID indexed, per kind
        self.counts = {k: 0 for k in KINDS}
        self.last_sync_ms: float | None = None
        self.last_query_ms: dict = {}
        self._load_state()

    def _load_state(self) -> None:
        offset = None
        with self._lock:
            while True:
                points, offset = self.client.scroll(self.collection, limit=1024, offset=offset,
                                                    with_payload=["kind", "ref_id"], with_vectors=False)
                for p in points:
                    k = p.payload["kind"]
                    self.counts[k] += 1
                    self.high[k] = max(self.high[k], p.payload["ref_id"])
                if offset is None:
                    break

    # ---------- indexing ----------

    def sync(self, batch: int = 64) -> int:
        """Embed every summary SQLite has that the index doesn't yet, and forget windows SQLite has pruned.
        Returns the number of summaries added."""
        t0 = time.perf_counter()
        added = 0
        with self._lock:
            for kind in KINDS:
                while True:
                    rows = self.store.summaries_after(kind, self.high[kind], batch)
                    if not rows:
                        break
                    vecs = self.embedder.embed_docs([r["summary"] for r in rows])
                    self.client.upsert(self.collection, points=[
                        self.m.PointStruct(id=point_id(kind, r["id"]), vector=v, payload={
                            "kind": kind, "ref_id": r["id"], "camera_id": r["camera_id"],
                            "start_ts": r["start_ts"], "end_ts": r["end_ts"], "text": r["summary"]})
                        for r, v in zip(rows, vecs)])
                    self.high[kind] = rows[-1]["id"]
                    self.counts[kind] += len(rows)
                    added += len(rows)
            self._prune_windows()
        if added:
            self.last_sync_ms = (time.perf_counter() - t0) * 1000
            log.info("vector index: +%d summaries in %.0f ms", added, self.last_sync_ms)
        return added

    def _prune_windows(self) -> None:
        """The windows table keeps only the latest 200 per camera; drop vectors of windows that are gone."""
        if not self.counts["window"]:
            return
        live = set(self.store.window_ids())
        if len(live) >= self.counts["window"]:
            return
        m = self.m
        flt = m.Filter(must=[m.FieldCondition(key="kind", match=m.MatchValue(value="window"))])
        gone, offset = [], None
        while True:
            points, offset = self.client.scroll(self.collection, scroll_filter=flt, limit=1024, offset=offset,
                                                with_payload=["ref_id"], with_vectors=False)
            gone += [p.id for p in points if p.payload["ref_id"] not in live]
            if offset is None:
                break
        if gone:
            self.client.delete(self.collection, points_selector=m.PointIdsList(points=gone))
            self.counts["window"] -= len(gone)

    def start_background_sync(self, every_s: float) -> threading.Thread:
        """Keep the index current while the cameras run, so a question rarely has anything left to embed."""

        def loop():
            while True:
                try:
                    self.sync()
                except Exception as e:  # never let indexing take the web app down
                    log.warning("vector sync failed: %s", e)
                time.sleep(every_s)

        t = threading.Thread(target=loop, name="vector-sync", daemon=True)
        t.start()
        return t

    # ---------- search ----------

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.embed_query(text)

    def search(self, vector: list[float], kind: str, ids: list[int] | None = None, limit: int = 20,
               exclude: int | None = None) -> list[Hit]:
        """Nearest summaries of one kind, restricted to `ids` (None = any)."""
        if ids is not None and not ids:
            return []
        m = self.m
        must = [m.FieldCondition(key="kind", match=m.MatchValue(value=kind))]
        if ids is not None:
            must.append(m.FieldCondition(key="ref_id", match=m.MatchAny(any=list(ids))))
        must_not = [m.FieldCondition(key="ref_id", match=m.MatchValue(value=exclude))] if exclude is not None else []
        with self._lock:
            res = self.client.query_points(self.collection, query=vector, limit=limit,
                                           query_filter=m.Filter(must=must, must_not=must_not),
                                           with_payload=["ref_id"]).points
        return [Hit(kind, p.payload["ref_id"], float(p.score)) for p in res]

    def vector_of(self, kind: str, ref_id: int) -> list[float] | None:
        with self._lock:
            got = self.client.retrieve(self.collection, ids=[point_id(kind, ref_id)], with_vectors=True)
        return list(got[0].vector) if got else None

    def stats(self) -> dict:
        return {"engine": "qdrant", "mode": "server" if self.where.startswith("http") else "embedded",
                "collection": self.collection, "embedder": self.embedder.name, "dim": self.embedder.dim,
                "events": self.counts["event"], "windows": self.counts["window"],
                "last_sync_ms": round(self.last_sync_ms) if self.last_sync_ms is not None else None}

    def close(self) -> None:
        with self._lock:
            self.client.close()


def open_index(cfg: VectorCfg, store: EventStore, data_dir: Path, ollama_host: str) -> VectorIndex | None:
    """The configured index, or None (the chat then falls back to keyword search) if it can't be opened.
    Embedded Qdrant locks its folder, so a second process (`aequitas ask` while the web app runs) gets a
    private in-memory copy built from SQLite instead."""
    if not cfg.enabled:
        return None
    try:
        embedder = make_embedder(cfg, data_dir, ollama_host)
    except Exception as e:
        log.warning("vector search off: embedding model %s could not load (%s)", cfg.model, e)
        return None
    try:
        index = VectorIndex(store, embedder, path=None if cfg.url else data_dir / "vectors", url=cfg.url)
    except Exception as e:
        log.warning("vector index at %s is in use (%s); using an in-memory copy", data_dir / "vectors", e)
        index = VectorIndex(store, embedder)
    index.sync()
    return index
