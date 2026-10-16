"""Whole-recording summary: a short overview and timestamped key points, written from the 4 s window summaries.

Text only (the frames were already described window by window), so it is one cheap LLM call per recording.
Grounding: every key point must sit at the time of an analysed window (+- one window), otherwise it is dropped;
an overview without any window summaries is never written. Key points closer than two windows to the previous one
are dropped: on a 79 s fire-rescue clip, 7 of Gemma E2B's 8 points were consecutive windows of the same scene."""

from __future__ import annotations

import re
import time

from pydantic import BaseModel, Field

from aequitas.taxonomy import label as cat_label

MAX_LINES = 80  # longer recordings: consecutive windows with the same label and text are merged first


class KeyPoint(BaseModel):
    time: str = Field(description="mm:ss from the timeline")
    point: str = Field(max_length=200)


class RecordingSummary(BaseModel):
    overview: str = Field(max_length=700)
    key_points: list[KeyPoint] = Field(default_factory=list, max_length=8)


def mmss(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    return f"{s // 60:02d}:{s % 60:02d}"


def parse_mmss(text: str) -> float | None:
    parts = text.strip().split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    if len(nums) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    return None


def timeline_lines(windows: list[dict]) -> list[str]:
    """'mm:ss [label] summary' per window; runs of identical windows merged, so long recordings fit."""
    lines, prev = [], None
    for w in windows:
        if not w.get("summary"):
            continue
        key = (w.get("label"), w["summary"].strip())
        if key == prev:
            continue
        prev = key
        label = (w.get("label") or "unlabelled").replace("_", " ")
        if label != "none" and w.get("violence") is not None and (w.get("source") or "").startswith("clip_model"):
            # the violence model saw the fight in the motion even when the description missed it: say so
            label += f"; violence model {w['violence']:.2f}"
        lines.append(f"{mmss(w['t'])} [{label}] {w['summary'].strip()}")
    if len(lines) > MAX_LINES:  # keep every incident line, thin the rest evenly
        inc = [i for i, ln in enumerate(lines) if "[none]" not in ln]
        rest = [i for i in range(len(lines)) if i not in set(inc)]
        room = max(0, MAX_LINES - len(inc))
        keep = set(inc[:MAX_LINES]) | set(rest[:: max(1, len(rest) // room)][:room] if room else [])
        lines = [ln for i, ln in enumerate(lines) if i in keep]
    return lines


def prompt(lines: list[str], confirmed: list[str]) -> str:
    conf = ("\nAn officer has confirmed these moments: " + "; ".join(confirmed) + ".\n") if confirmed else ""
    return (
        "Below is the timeline of a video (a police body camera or a citizen's report). Each line is what an AI "
        "saw in 4 seconds of video: time (mm:ss), a label, and a description. A label marked 'violence model' "
        "came from a second model that watches motion: it saw that incident even where the description does not "
        "mention it, so state it as flagged by the violence model.\n\n" + "\n".join(lines) + "\n" + conf + "\n"
        "Write:\n"
        "- overview: 2 to 4 plain sentences telling what happens in the recording as one story, in order: the "
        "setting and who is there, then what changes (when an incident starts and how long it lasts: 'from mm:ss "
        "to mm:ss', or 'at mm:ss' for a single line), then how it ends. Merge lines that describe the same thing; do not retell every line. If "
        "nothing is labelled, say that no incident was seen. For an incident, use the most specific description "
        "among its lines (e.g. 'two men grapple and wrestle') and never filler such as 'physical movements', "
        "'various poses', 'in a confined space' or 'engaged in an activity'. Only use what the timeline says. Do "
        "not guess identities, names, intentions, or anything not described.\n"
        "- key_points: up to 8 DIFFERENT moments an officer reviewing the recording should look at, each with "
        "the time copied exactly from a timeline line and a short description. When several lines in a row "
        "describe the same thing, give one key point at the first of them. Prefer the start of each labelled "
        "incident, the start and end of physical contact, restraint or handcuffing, weapons, injuries, fire, "
        "and changes of scene.\n"
        "Never describe faces, religion, caste, political symbols or clothing as a reason for anything. "
        "Respond with JSON only."
    )


def ground(s: RecordingSummary, windows: list[dict], window_s: float = 4.0) -> RecordingSummary:
    """Drop key points whose time does not fall on an analysed window; sort them by time."""
    starts = [w["t"] for w in windows if w.get("summary")]
    kept = []
    for kp in s.key_points:
        t = parse_mmss(kp.time)
        if t is None or not any(abs(t - st) <= window_s for st in starts):
            continue
        kept.append((t, KeyPoint(time=mmss(t), point=kp.point.strip())))
    kept.sort(key=lambda x: x[0])
    spread = []  # one point per moment: consecutive windows describing the same scene come out as a run
    for t, k in kept:
        if spread and t - spread[-1][0] < 2 * window_s:
            continue
        spread.append((t, k))
    return RecordingSummary(overview=s.overview.strip(), key_points=[k for _, k in spread])


def _span(a: float, b: float) -> str:
    return f"at {mmss(a)}" if b - a <= 4.5 else f"from {mmss(a)} to {mmss(b)}"


def headline(windows: list[dict], events: list[dict]) -> str:
    """The first lines an officer reads, built from the labels and officer decisions, not written by the LLM:
    what was flagged, when, and whether an officer has confirmed it, then the stretches where nothing was found.
    Built when the page loads, so it follows the officer's latest decision. The LLM overview below it retells
    the window descriptions, which are often vague on CCTV (Gemma alone saw 0/5 faculty fights), so the flags
    must not depend on it."""
    ws = sorted((w for w in windows if w.get("summary") or w.get("label")), key=lambda w: w["t"])
    if not ws:
        return ""
    runs: list[dict] = []  # back-to-back windows that are all incidents, or all quiet
    for w in ws:
        inc = w.get("label") not in (None, "none")
        end = w.get("end", w["t"] + 4.0)
        if runs and runs[-1]["inc"] == inc and w["t"] - runs[-1]["b"] < 0.5:
            runs[-1]["b"] = end
            runs[-1]["ws"].append(w)
        else:
            runs.append({"inc": inc, "a": w["t"], "b": end, "ws": [w]})
    if not any(r["inc"] for r in runs):
        return f"No incident found in the {mmss(runs[-1]['b'] - runs[0]['a'])} analysed."
    out = []
    for r in runs:
        if not r["inc"]:
            out.append(f"Nothing found {_span(r['a'], r['b'])}.")
            continue
        counts: dict[str, int] = {}
        for w in r["ws"]:
            counts[w["label"]] = counts.get(w["label"], 0) + 1
        main = max(counts, key=lambda k: counts[k])  # ties: the first seen
        also = [cat_label(k) for k in counts if k != main]
        reviews = {e.get("review") for e in events if r["a"] - 0.5 <= e["t"] < r["b"]}
        state = ("confirmed by an officer" if "confirmed" in reviews else
                 "dismissed by an officer" if reviews == {"dismissed"} else "not yet confirmed by an officer")
        if all((w.get("source") or "").startswith("clip_model") for w in r["ws"]):
            state += "; seen by the violence model, which watches motion, so the descriptions may not show it"
        text = f"{cat_label(main).capitalize()} flagged {_span(r['a'], r['b'])}"
        out.append(text + (f" (also {', '.join(also)})" if also else "") + f" ({state}).")
    return " ".join(out)


# Filler the window model writes when it can't tell what people do (seen on CCTV fights): left in, the LLM copies
# it into the overview ("a person is performing physical movements in a confined space"), so it is swapped out.
VAGUE = re.compile(r"various poses|physical movements|physical activity|confined space|engaged in an activity|"
                   r"in motion or|appearing to be in motion|stretching", re.I)


def plain_incidents(windows: list[dict]) -> list[dict]:
    """Incident windows whose description is filler get 'The AI flagged <label> here.' instead, for the summary
    prompt only (stored windows keep the model's words)."""
    out = []
    for w in windows:
        if w.get("summary") and w.get("label") not in (None, "none") and VAGUE.search(w["summary"]):
            w = {**w, "summary": f"The AI flagged {cat_label(w['label'])} here."}
        out.append(w)
    return out


def summarize_recording(client, windows: list[dict], confirmed: list[str] | None = None) -> tuple[dict | None, float]:
    """Returns ({overview, key_points: [{time, t, point}], dropped}, ms); (None, ms) without window summaries
    or when the model gave no valid answer."""
    t0 = time.perf_counter()
    lines = timeline_lines(plain_incidents(windows))
    if not lines:
        return None, 0.0
    raw = client.structured([{"role": "user", "content": prompt(lines, confirmed or [])}], RecordingSummary,
                            max_tokens=600)
    ms = (time.perf_counter() - t0) * 1000
    if raw is None:
        return None, ms
    g = ground(raw, windows)
    return {"overview": g.overview,
            "key_points": [{"time": k.time, "t": parse_mmss(k.time), "point": k.point} for k in g.key_points],
            "dropped": len(raw.key_points) - len(g.key_points)}, ms
