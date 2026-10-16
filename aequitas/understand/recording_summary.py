"""Whole-recording summary: a short overview and timestamped key points, written from the 4 s window summaries.

Text only (the frames were already described window by window), so it is one cheap LLM call per recording.
Grounding: every key point must sit at the time of an analysed window (+- one window), otherwise it is dropped;
an overview without any window summaries is never written. Key points closer than two windows to the previous one
are dropped: on a 79 s fire-rescue clip, 7 of Gemma E2B's 8 points were consecutive windows of the same scene."""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

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
        "- overview: 2 to 4 plain sentences on what happens in the recording, in order. Only use what the "
        "timeline says. Do not guess identities, names, intentions, or anything not described.\n"
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


def motion_flags(windows: list[dict]) -> str:
    """A fixed sentence (not written by the LLM) listing incidents only the violence model saw. The description
    model often misses a fight on CCTV (Gemma alone: 0/5 faculty clips), and the LLM summary then reads as if
    nothing happened, so the flag is stated here in plain words."""
    by_label: dict[str, list[str]] = {}
    for w in windows:
        if w.get("label") not in (None, "none") and (w.get("source") or "").startswith("clip_model"):
            by_label.setdefault(w["label"].replace("_", " "), []).append(mmss(w["t"]))
    if not by_label:
        return ""
    parts = [f"{label} at {', '.join(ts)}" for label, ts in by_label.items()]
    return (f"The violence model, which watches motion, flagged {'; '.join(parts)}. The descriptions above may "
            "not mention it: check these moments in the video.")


def summarize_recording(client, windows: list[dict], confirmed: list[str] | None = None) -> tuple[dict | None, float]:
    """Returns ({overview, key_points: [{time, t, point}], dropped}, ms); (None, ms) without window summaries
    or when the model gave no valid answer."""
    t0 = time.perf_counter()
    lines = timeline_lines(windows)
    if not lines:
        return None, 0.0
    raw = client.structured([{"role": "user", "content": prompt(lines, confirmed or [])}], RecordingSummary,
                            max_tokens=600)
    ms = (time.perf_counter() - t0) * 1000
    if raw is None:
        return None, ms
    g = ground(raw, windows)
    return {"overview": " ".join(x for x in (g.overview, motion_flags(windows)) if x),
            "key_points": [{"time": k.time, "t": parse_mmss(k.time), "point": k.point} for k in g.key_points],
            "dropped": len(raw.key_points) - len(g.key_points)}, ms
