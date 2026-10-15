"""Command line: init-db, run, ask, ui, spike, stats."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from minerva.config import load_cameras, load_settings


def _tz(name: str | None):
    if not name:
        return None
    from zoneinfo import ZoneInfo

    return ZoneInfo(name)


def _service(settings, use_llm: bool):
    from minerva.chat.answer import ChatService
    from minerva.chat.parser import QuestionParser
    from minerva.store.db import EventStore

    llm = None
    if use_llm:
        from minerva.understand.vlm_client import make_client

        llm = make_client(settings.chat.endpoint(settings.vlm))
        if not llm.ping():
            logging.warning("chat model %s not reachable at %s; using the keyword parser", llm.model, llm.cfg.host)
            llm = None
    return ChatService(EventStore(settings.db_path), QuestionParser(llm, _tz(settings.chat.timezone)),
                       summarize=settings.chat.summarize)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="minerva")
    ap.add_argument("--settings", default="config/settings.yaml")
    ap.add_argument("--cameras", default="config/cameras.yaml")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-db", help="create the database and register cameras")

    run = sub.add_parser("run", help="process camera streams and record events")
    run.add_argument("--camera", action="append", help="only these camera IDs (repeatable)")
    run.add_argument("--no-vlm", action="store_true", help="trigger-only mode, no VLM verification")
    run.add_argument("--max-seconds", type=float, default=None)

    ask = sub.add_parser("ask", help="ask a question about stored events")
    ask.add_argument("question")
    ask.add_argument("--no-llm", action="store_true", help="use the keyword parser only")

    ui = sub.add_parser("ui", help="start the chat web app")
    ui.add_argument("--port", type=int, default=7860)
    ui.add_argument("--no-llm", action="store_true")

    spike = sub.add_parser("spike", help="M0: measure detector/VLM latency and memory on this machine")
    spike.add_argument("--models", nargs="+",
                       default=["qwen2.5vl:3b", "gemma4:e2b", "gemma4:e4b", "qwen3-vl:2b", "qwen3-vl:4b"])
    spike.add_argument("--video", default=None, help="sample frames from this video instead of a stock image")
    spike.add_argument("--vlm-iters", type=int, default=3)

    sub.add_parser("stats", help="per-stage latency summary of recorded events")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    settings = load_settings(args.settings)

    if args.cmd == "init-db":
        from minerva.store.db import EventStore

        store = EventStore(settings.db_path)
        for cam in load_cameras(args.cameras):
            store.upsert_camera(cam)
        print(f"Database ready at {settings.db_path}")
        return 0

    if args.cmd == "run":
        from minerva.pipeline import Pipeline
        from minerva.store.db import EventStore

        cams = load_cameras(args.cameras)
        if args.camera:
            cams = [c for c in cams if c.camera_id in set(args.camera)]
            if not cams:
                print("No matching cameras.", file=sys.stderr)
                return 2
        store = EventStore(settings.db_path)
        Pipeline(settings, cams, store, use_vlm=not args.no_vlm).run(args.max_seconds)
        return 0

    if args.cmd == "ask":
        print(_service(settings, not args.no_llm).ask(args.question).text)
        return 0

    if args.cmd == "ui":
        from minerva.ui.app import build_ui

        demo = build_ui(_service(settings, not args.no_llm))
        demo.launch(server_name="127.0.0.1", server_port=args.port,
                    allowed_paths=[str(Path(settings.data_dir).resolve())])
        return 0

    if args.cmd == "spike":
        from minerva.eval.spike import run_spike, summarize

        print(summarize(run_spike(settings, args.models, args.video, vlm_iters=args.vlm_iters)))
        return 0

    if args.cmd == "stats":
        from minerva.eval.report import latency_table
        from minerva.store.db import EventStore

        print(latency_table(EventStore(settings.db_path)))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
