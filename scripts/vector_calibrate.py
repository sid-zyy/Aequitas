"""Pick the chat's vector-match threshold (chat.vector.min_score) on SYNTHETIC sentences, never on the faculty clips.

The documents are written in the style of stage-1 summaries; each query lists which documents answer it. The
script embeds both with the configured model, then reports, for every threshold, how many relevant pairs pass
and how many irrelevant pairs pass. It also prints the per-question latency of embedding a query.

    .venv/Scripts/python scripts/vector_calibrate.py [--model BAAI/bge-base-en-v1.5]
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np

DOCS = [
    "Two men punch and shove each other outside a shop while others watch.",            # 0
    "A man walks a dog along the footpath past parked scooters.",                         # 1
    "People queue at a bus stop and board a city bus.",                                  # 2
    "A group stands quietly holding candles in a peaceful gathering.",                   # 3
    "Smoke and flames rise from a parked scooter at the roadside.",                      # 4
    "A woman leaves a black bag on a bench and walks away.",                             # 5
    "A worker carries a tall ladder across the road.",                                   # 6
    "An auto-rickshaw collides with a motorbike at the junction; the rider falls.",      # 7
    "Police officers hold a man by the arms and lead him to a jeep.",                    # 8
    "A man snatches a phone from a woman and runs off.",                                 # 9
    "A person lies motionless on the pavement as people gather around.",                 # 10
    "Traffic moves slowly on a wide road with cars and buses.",                          # 11
    "A boy rides a bicycle down an empty lane.",                                         # 12
    "Shopkeepers pull down shutters as a crowd runs past.",                              # 13
    "A man sprays paint on a wall and breaks a shop window.",                            # 14
    "Two people chat near a tea stall in the evening.",                                  # 15
    "A delivery rider waits on a motorbike outside a restaurant.",                       # 16
    "A man holds a long knife and waves it at passers-by.",                              # 17
    "Children in school uniforms cross the road with a guard.",                          # 18
    "A truck unloads boxes outside a warehouse.",                                        # 19
]
QUERIES = [
    ("anyone fighting", [0]), ("a brawl", [0]), ("someone with a pet", [1]), ("public transport", [2, 11]),
    ("a vigil", [3]), ("fire", [4]), ("unattended luggage", [5]), ("someone carrying a ladder", [6]),
    ("road accident", [7]), ("arrest", [8]), ("phone snatching", [9]), ("person collapsed", [10]),
    ("cyclist", [12]), ("people running away", [13, 9]), ("vandalism", [14]), ("weapon", [17]),
    ("kids crossing", [18]), ("goods being delivered", [19, 16]), ("helicopter landing", []), ("swimming pool", []),
    ("a horse", []), ("someone on a motorbike", [16, 7]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--cache", default="models/fastembed")
    args = ap.parse_args()
    from fastembed import TextEmbedding

    m = TextEmbedding(args.model, cache_dir=str(Path(args.cache)))
    D = np.array(list(m.embed(DOCS)))
    rel, irr, top1, lat = [], [], 0, []
    for q, want in QUERIES:
        t0 = time.perf_counter()
        v = np.array(list(m.query_embed([q])))[0]
        lat.append((time.perf_counter() - t0) * 1000)
        s = D @ v
        for i, x in enumerate(s):
            (rel if i in want else irr).append(float(x))
        top1 += bool(want) and int(np.argmax(s)) in want
    print(f"model {args.model}: top-1 right on {top1}/{sum(bool(w) for _, w in QUERIES)} answerable questions; "
          f"query embed p50 {statistics.median(lat):.1f} ms")
    print(f"relevant scores: min {min(rel):.3f} median {statistics.median(rel):.3f}; "
          f"irrelevant: median {statistics.median(irr):.3f} max {max(irr):.3f}")
    print("threshold  relevant_kept  irrelevant_kept")
    for t in np.arange(0.50, 0.80, 0.02):
        print(f"  {t:.2f}     {sum(x >= t for x in rel):>3}/{len(rel)}        {sum(x >= t for x in irr):>3}/{len(irr)}")


if __name__ == "__main__":
    main()
