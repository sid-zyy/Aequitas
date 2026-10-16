"""Questions about ONE video (a citizen report or a bodycam recording), answered from that video's analysis only.

The model sees the recording summary, the key moments (with the officer's decision on each), the 4 s window
timeline and, for citizen reports, what the citizen wrote (marked unverified). Answers cite times like [01:12];
cited times outside the video are dropped, so the UI only offers seek buttons that exist. Text only: the frames were
already described window by window, so a question costs one short LLM call."""

from __future__ import annotations

import re
import time

from aequitas.taxonomy import label as cat_label
from aequitas.understand.recording_summary import mmss, parse_mmss, timeline_lines

MAX_HISTORY = 4  # earlier turns sent back so follow-ups ("what happened after that?") make sense
CITE = re.compile(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]")


def video_prompt(question: str, windows: list[dict], events: list[dict], summary: dict | None,
                 citizen_text: str | None = None, history: list[dict] | None = None) -> str:
    def decision(e: dict) -> str:
        return {"confirmed": "confirmed by an officer", "dismissed": "dismissed by an officer"}.get(
            e.get("review"), "not reviewed")

    parts = [
        "You answer a police officer's questions about ONE video, using ONLY the analysis below. "
        "Times are mm:ss from the start of the video.",
        "Rules: answer in 1-4 sentences. Write every time in square brackets, like [01:12]. Only the summary, "
        "key moments and timeline are evidence of what the video shows. A label in square brackets on the "
        "timeline (e.g. [fighting]) is evidence too, from the AI; [fighting; violence model 0.99] means a second "
        "model that watches motion saw fighting there even if the description misses it. When the AI flagged an "
        "incident, say so, e.g. 'the AI flagged fighting at [mm:ss] (not yet confirmed by an officer)' with that "
        "moment's own time, even if the description is vague. When asked whether the video shows or matches an "
        "incident, name the matching flags first, then say what the descriptions do and do not show. If the "
        "analysis does not mention something, answer that the analysis does not show it, then say what it does "
        "show. Report the AI's flags as they are: never call a flag wrong, mistaken or a misinterpretation, and never "
        "say the violence model did not confirm one; only an officer confirms or dismisses a flag. Describe actions "
        "only: never name, identify or guess who a person is, and never describe faces.",
    ]
    if citizen_text:
        parts.append("UNVERIFIED CLAIM from the person who reported it. This is NOT evidence: never say the video "
                     "shows something only because this claim says so.\nClaim: " + citizen_text.strip())
    if summary and summary.get("overview"):
        parts.append("Summary of the video (AI draft): " + summary["overview"])
    if events:
        parts.append("Key moments the AI flagged:\n" + "\n".join(
            f"{mmss(e['t'])} {cat_label(e['category'])} ({decision(e)}): {e.get('summary') or ''}" for e in events))
    lines = timeline_lines(windows)
    parts.append("Timeline, every analysed 4 s window:\n" + ("\n".join(lines) if lines else "(no window was analysed)"))
    for turn in (history or [])[-MAX_HISTORY:]:
        parts.append(f"Earlier question: {turn.get('q', '')}\nYour earlier answer: {turn.get('a', '')}")
    parts.append(f"Question: {question.strip()}")
    return "\n\n".join(parts)


def cited_times(text: str, duration_s: float) -> list[float]:
    """Seconds cited as [mm:ss], in order, without duplicates or times past the end of the video."""
    out: list[float] = []
    for m in CITE.finditer(text):
        t = parse_mmss(m.group(1))
        if t is not None and 0 <= t <= duration_s + 1 and t not in out:
            out.append(t)
    return out


def answer_video(llm, question: str, windows: list[dict], events: list[dict], summary: dict | None,
                 duration_s: float, citizen_text: str | None = None,
                 history: list[dict] | None = None) -> dict:
    """-> {"answer", "cited_s", "ms"}. Raises if the model fails (the route turns that into a 503)."""
    t0 = time.perf_counter()
    prompt = video_prompt(question, windows, events, summary, citizen_text, history)
    text = "".join(llm.stream_text([{"role": "user", "content": prompt}], max_tokens=256)).strip()
    if not text:
        raise RuntimeError("empty answer")
    return {"answer": text, "cited_s": cited_times(text, duration_s), "ms": round((time.perf_counter() - t0) * 1000, 1)}
