from minerva.triggers.base import Scene
from minerva.types import TrackObs


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
