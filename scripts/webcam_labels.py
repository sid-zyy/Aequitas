"""Re-label saved webcam windows (stage 2 only) to check false alarms on ordinary close-range scenes.

    .venv/Scripts/python scripts/webcam_labels.py --camera demo-webcam --since "2026-10-16 03:47:39" --repeats 3

Each window is labelled `repeats` times with the previous window's summary as context, like the live pipeline.
Prints how often each summary got an incident label."""

import argparse
import sqlite3
from collections import Counter
from datetime import datetime

from aequitas.config import Settings
from aequitas.understand.vlm_client import make_client


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="demo-webcam")
    ap.add_argument("--since", required=True, help="local time, YYYY-MM-DD HH:MM:SS")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--db", default="data/aequitas.db")
    a = ap.parse_args()
    s = Settings()
    client = make_client(s.vlm)
    since = int(datetime.fromisoformat(a.since).timestamp() * 1000)
    rows = sqlite3.connect(a.db).execute(
        "SELECT summary FROM windows WHERE camera_id=? AND start_ts>=? AND summary IS NOT NULL ORDER BY start_ts",
        (a.camera, since)).fetchall()
    flagged, total, prev = Counter(), 0, None
    for (summary,) in rows:
        labels = [client.classify_text(summary, "", s.categories, previous=prev)[0] for _ in range(a.repeats)]
        bad = [x for x in labels if x not in (None, "none")]
        total += len(labels)
        flagged.update(bad)
        print(f"{len(bad)}/{len(labels)} {','.join(sorted(set(bad))) or '-':22} {summary[:110]}")
        prev = summary
    print(f"\nincident labels: {sum(flagged.values())}/{total}  {dict(flagged)}")


if __name__ == "__main__":
    main()
