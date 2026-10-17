from aequitas.triggers.base import Scene
from aequitas.types import TrackObs


def person(tid, x1, y1, x2, y2, conf=0.9):
    return TrackObs(tid, "person", conf, (float(x1), float(y1), float(x2), float(y2)))


def bag(tid, x1, y1, x2, y2, conf=0.8, label="backpack"):
    return TrackObs(tid, label, conf, (float(x1), float(y1), float(x2), float(y2)))


def scene(ts, persons, objects=(), motion=0.0, size=(1000, 1000), cam="cam1"):
    return Scene(cam, ts, size[0], size[1], list(persons), list(objects), lambda box: motion)


def run(engine, scenes):
    cands, ongoing = [], []
    for s in scenes:
        out = engine.step(s)
        cands.extend(out.candidates)
        ongoing.extend(out.ongoing)
    return cands, ongoing


class WordEmbedder:
    """A stand-in for the sentence-embedding model: hashed bag of word stems, unit length. Sentences that share
    words come out close, so vector-search tests run against real Qdrant with no model download."""

    name = "test-words"
    dim = 256

    def _vec(self, text):
        import re

        v = [0.0] * self.dim
        for w in re.findall(r"[a-z]{3,}", text.lower()):
            v[sum(map(ord, w[:5])) % self.dim] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def embed_docs(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)
