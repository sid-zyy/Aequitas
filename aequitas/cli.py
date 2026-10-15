"""Command line: init-db, run, ask, ui, spike, stats."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from aequitas.config import load_cameras, load_settings


def _tz(name: str | None):
    if not name:
        return None
    from zoneinfo import ZoneInfo

    return ZoneInfo(name)


def _service(settings, use_llm: bool):
    from aequitas.chat.answer import ChatService
    from aequitas.chat.parser import QuestionParser
    from aequitas.store.db import EventStore

    llm = None
    if use_llm:
        from aequitas.understand.vlm_client import make_client

        llm = make_client(settings.chat.endpoint(settings.vlm))
        if not llm.ping():
            logging.warning("chat model %s not reachable at %s; using the keyword parser", llm.model, llm.cfg.host)
            llm = None
    parser = QuestionParser(llm, _tz(settings.chat.timezone), settings.categories)
    return ChatService(EventStore(settings.db_path), parser, summarize=settings.chat.summarize)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aequitas")
    ap.add_argument("--settings", default="config/settings.yaml")
    ap.add_argument("--cameras", default="config/cameras.yaml")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-db", help="create the database and register cameras")

    cu = sub.add_parser("create-user", help="create a police account (prompts for the password)")
    cu.add_argument("--username", required=True)
    cu.add_argument("--role", choices=["admin", "officer"], default="officer")

    run = sub.add_parser("run", help="process camera streams and record events")
    run.add_argument("--camera", action="append", help="only these camera IDs (repeatable)")
    run.add_argument("--no-vlm", action="store_true", help="trigger-only mode, no VLM verification")
    run.add_argument("--max-seconds", type=float, default=None)

    ask = sub.add_parser("ask", help="ask a question about stored events")
    ask.add_argument("question")
    ask.add_argument("--no-llm", action="store_true", help="use the keyword parser only")

    ui = sub.add_parser("web", aliases=["ui"], help="start the web app (upload videos, browse events, chat)")
    ui.add_argument("--port", type=int, default=8000)
    ui.add_argument("--no-llm", action="store_true")

    spike = sub.add_parser("spike", help="M0: measure detector/VLM latency and memory on this machine")
    spike.add_argument("--models", nargs="+",
                       default=["qwen2.5vl:3b", "gemma4:e2b", "gemma4:e4b", "qwen3-vl:2b", "qwen3-vl:4b"])
    spike.add_argument("--video", default=None, help="sample frames from this video instead of a stock image")
    spike.add_argument("--vlm-iters", type=int, default=3)

    bench = sub.add_parser("bench", help="latency of summarize/classify/chat on an OpenAI-compatible server")
    bench.add_argument("--host", default=None, help="server URL (default: vlm.host from settings)")
    bench.add_argument("--model", default=None, help="served model name (default: vlm.model)")
    bench.add_argument("--tasks", nargs="+", default=["summarize", "classify", "chat"],
                       choices=["summarize", "classify", "chat"])
    bench.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 16])
    bench.add_argument("--requests", type=int, default=16, help="requests per concurrency level")
    bench.add_argument("--label", default="", help="e.g. w4a16-mtp, used in the report name")

    ev = sub.add_parser("eval-classify", help="stage-2 accuracy/F1/latency on labelled summaries")
    ev.add_argument("--labels", default="config/classify_eval.yaml")
    ev.add_argument("--host", default=None, help="server URL (default: vlm.host)")
    ev.add_argument("--model", default=None, help="model name (default: vlm.model)")
    ev.add_argument("--backend", choices=["ollama", "openai"], default=None)
    ev.add_argument("--concurrency", type=int, default=4)
    ev.add_argument("--label", default="", help="e.g. w4a16, used in the report name")

    sub.add_parser("stats", help="per-stage latency summary of recorded events")

    rep = sub.add_parser("eval-report", help="score recorded events per clip against expected labels")
    rep.add_argument("--expected", required=True, help="YAML of expected labels per camera/clip")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    settings = load_settings(args.settings)

    if args.cmd == "init-db":
        from aequitas.store.db import EventStore

        store = EventStore(settings.db_path)
        for cam in load_cameras(args.cameras):
            store.upsert_camera(cam)
        print(f"Database ready at {settings.db_path}")
        return 0

    if args.cmd == "create-user":
        import getpass

        from aequitas.auth import Auth
        from aequitas.store.db import EventStore

        password = getpass.getpass("Password (10+ characters): ")
        if password != getpass.getpass("Repeat password: "):
            print("Passwords do not match.", file=sys.stderr)
            return 2
        try:
            Auth(EventStore(settings.db_path)).create_user(args.username, password, args.role)
        except Exception as e:  # bad input, or the username exists
            print(f"Could not create the account: {e}", file=sys.stderr)
            return 2
        print(f"Created {args.role} account '{args.username}'.")
        return 0

    if args.cmd == "run":
        from aequitas.pipeline import Pipeline
        from aequitas.store.db import EventStore

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

    if args.cmd in ("web", "ui"):
        from aequitas.web.server import serve

        cams = load_cameras(args.cameras) if Path(args.cameras).exists() else []
        serve(settings, _service(settings, not args.no_llm), cams, port=args.port)
        return 0

    if args.cmd == "spike":
        from aequitas.eval.spike import run_spike, summarize

        print(summarize(run_spike(settings, args.models, args.video, vlm_iters=args.vlm_iters)))
        return 0

    if args.cmd == "bench":
        from aequitas.eval.bench import run_bench

        host = args.host or settings.vlm.host
        if args.host is None and settings.vlm.backend == "ollama":
            host = host.rstrip("/") + "/v1"  # Ollama's OpenAI-compatible endpoint
        report = run_bench(host, args.model or settings.vlm.model, args.tasks, args.concurrency, args.requests,
                           label=args.label, out_dir=settings.data_dir / "reports",
                           allow_remote=settings.vlm.allow_remote)
        print(Path(report["_path"]).read_text(encoding="utf-8"))
        print(f"Report: {report['_path']}")
        return 0

    if args.cmd == "eval-classify":
        from aequitas.eval.classify_eval import load_items, run_eval, summary_line
        from aequitas.understand.vlm_client import make_client

        ep = settings.vlm.model_copy(update={k: v for k, v in
                                             {"host": args.host, "model": args.model, "backend": args.backend}.items()
                                             if v is not None})
        items, cats = load_items(args.labels)
        report = run_eval(make_client(ep), items, cats or settings.categories, args.concurrency, args.label,
                          settings.data_dir / "reports")
        print(summary_line(report))
        return 0

    if args.cmd == "stats":
        from aequitas.eval.report import latency_table
        from aequitas.store.db import EventStore

        print(latency_table(EventStore(settings.db_path)))
        return 0

    if args.cmd == "eval-report":
        from aequitas.eval.clip_report import clip_report
        from aequitas.store.db import EventStore

        print(clip_report(EventStore(settings.db_path), args.expected))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
