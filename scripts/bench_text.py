"""Times the text-only tasks shown in the presentation: the bodycam whole-recording summary and Hindi/Kannada
citizen-report translation. Uses the app's own code and prompts against Ollama.

    .venv/Scripts/python scripts/bench_text.py --model qwen3-vl:2b-instruct --db data/eval/demo_qwen2b/aequitas.db

The recording summaries are written from the stored 4 s window descriptions of six demo clips in --db (use a run
made with the same model). Each translation sentence gets a random hour, so Ollama never answers from its cache.
Prints every output, so the translations can be checked by eye."""

import argparse
import random
import sqlite3
import statistics
import sys

sys.path.insert(0, ".")
from aequitas.config import load_settings  # noqa: E402
from aequitas.translate import translate_to_english  # noqa: E402
from aequitas.understand.recording_summary import summarize_recording  # noqa: E402
from aequitas.understand.vlm_client import make_client  # noqa: E402

CAMS = ["arrest-1", "fire-rescue", "arrest-4", "punching", "brigade-road", "fac-v011"]
HI = ["कल रात लगभग {h} बजे एमजी रोड पर दो लोगों ने एक छात्र को घेरकर उसका फोन छीन लिया और उसे धक्का दिया। वह घायल है।",
      "आज शाम {h} बजे कोरमंगला बस स्टॉप के पास एक आदमी चाकू दिखाकर लोगों को डरा रहा था।",
      "{h} बजे ब्रिगेड रोड पर एक स्कूटर ने पैदल चलने वाले को टक्कर मारी और भाग गया।"]
KN = ["ನಿನ್ನೆ ರಾತ್ರಿ ಸುಮಾರು {h} ಗಂಟೆಗೆ ಎಂ.ಜಿ. ರಸ್ತೆಯಲ್ಲಿ ಇಬ್ಬರು ಒಬ್ಬ ವಿದ್ಯಾರ್ಥಿಯನ್ನು ಸುತ್ತುವರಿದು ಅವನ ಫೋನ್ ಕಸಿದುಕೊಂಡರು.",
      "ಇಂದು ಸಂಜೆ {h} ಗಂಟೆಗೆ ಕೋರಮಂಗಲ ಬಸ್ ನಿಲ್ದಾಣದ ಬಳಿ ಒಬ್ಬ ವ್ಯಕ್ತಿ ಚಾಕು ತೋರಿಸಿ ಜನರನ್ನು ಹೆದರಿಸುತ್ತಿದ್ದ.",
      "{h} ಗಂಟೆಗೆ ಬ್ರಿಗೇಡ್ ರಸ್ತೆಯಲ್ಲಿ ಸ್ಕೂಟರ್ ಒಬ್ಬ ಪಾದಚಾರಿಗೆ ಡಿಕ್ಕಿ ಹೊಡೆದು ಪರಾರಿಯಾಯಿತು."]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--db", required=True, help="a demo-set run's database (its window descriptions are summarised)")
    a = ap.parse_args()
    s = load_settings("config/settings.yaml")
    s.vlm = s.vlm.model_copy(update={"model": a.model})
    client = make_client(s.chat.endpoint(s.vlm))
    db = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    times = []
    for cam in CAMS:
        rows = [dict(r) for r in db.execute("SELECT * FROM windows WHERE camera_id=? ORDER BY start_ts", (cam,))]
        if not rows:
            continue
        t0 = rows[0]["start_ts"]
        wins = [{"t": (r["start_ts"] - t0) / 1000, "end": (r["end_ts"] - t0) / 1000, "summary": r["summary"],
                 "label": r["label"], "source": r.get("source"), "violence": r.get("violence")} for r in rows]
        out, ms = summarize_recording(client, wins)
        times.append(ms)
        print(f"{cam}: {len(wins)} windows -> {ms:.0f} ms | {(out or {}).get('overview', 'NO SUMMARY')}")
    print(f"RESULT recording summary p50 {statistics.median(times):.0f} ms (min {min(times):.0f}, max {max(times):.0f}) "
          f"over {len(times)}")
    for lang, texts in [("hi", HI), ("kn", KN)]:
        ts = []
        for r in range(2):
            for tx in texts:
                en, ms = translate_to_english(client, tx.format(h=random.randint(1, 11)), lang)
                ts.append(ms)
                if r == 0:
                    print(f"{lang} {ms:.0f} ms | {en}")
        print(f"RESULT translate {lang} p50 {statistics.median(ts):.0f} ms over {len(ts)}")


if __name__ == "__main__":
    main()
