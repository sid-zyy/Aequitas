"""Gradio app: a chat panel with keyframes, and a recent-events table."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import gradio as gr

from minerva.chat.answer import ChatService, _fmt_time
from minerva.store.query import QueryFilter
from minerva.types import Status

EXAMPLES = [
    "Show all tailgating events from camera 1 today",
    "How many people fell this week?",
    "Any fights in the corridor in the last 2 hours?",
    "Was a bag left unattended near the entrance this morning?",
    "Show disputed and rejected events from today",
]


def _chatbot() -> gr.Chatbot:
    try:
        return gr.Chatbot(type="messages", height=520, label="Minerva")
    except TypeError:  # newer Gradio: messages format is the only format
        return gr.Chatbot(height=520, label="Minerva")


def build_ui(service: ChatService) -> gr.Blocks:
    store = service.store
    tz = service.parser.tz

    def respond(question: str, history: list):
        question = (question or "").strip()
        if not question:
            return "", history, []
        ans = service.ask(question)
        method = ans.parsed.method if ans.parsed else "?"
        history = (history or []) + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": ans.text + f"\n\n_(parsed with: {method})_"},
        ]
        gallery = [
            (r["keyframe_path"], f"{_fmt_time(r['start_ts'], tz)} · {r.get('camera_name') or r['camera_id']} · "
                                 f"{(r.get('subtype') or r['category']).replace('_', ' ')}")
            for r in ans.rows if Path(r["keyframe_path"]).exists()
        ]
        return "", history, gallery

    def recent(show_all: bool):
        statuses = list(Status) if show_all else [Status.CONFIRMED]
        rows = store.query(QueryFilter(statuses=statuses, limit=100))
        return [[_fmt_time(r["start_ts"], tz), r.get("camera_name") or r["camera_id"],
                 (r.get("subtype") or r["category"]).replace("_", " "), r["status"], round(r["confidence"], 2),
                 r.get("summary") or ""] for r in rows]

    with gr.Blocks(title="Minerva") as demo:
        gr.Markdown("## Minerva · ask about recorded events")
        with gr.Tab("Ask"):
            with gr.Row():
                with gr.Column(scale=3):
                    chat = _chatbot()
                    box = gr.Textbox(placeholder="e.g. Show all tailgating events from camera 1 today",
                                     show_label=False, autofocus=True)
                    gr.Examples(EXAMPLES, inputs=box)
                with gr.Column(scale=2):
                    gallery = gr.Gallery(label="Key frames", columns=1, height=620, object_fit="contain")
            box.submit(respond, [box, chat], [box, chat, gallery])
        with gr.Tab("Recent events"):
            show_all = gr.Checkbox(label="Include disputed and rejected", value=False)
            refresh = gr.Button("Refresh")
            table = gr.Dataframe(
                headers=["time", "camera", "event", "status", "confidence", "summary"],
                value=recent(False), wrap=True, interactive=False,
            )
            refresh.click(recent, show_all, table)
            show_all.change(recent, show_all, table)
        gr.Markdown(f"<small>Loaded {datetime.now():%Y-%m-%d %H:%M}. All inference runs locally.</small>")
    return demo
