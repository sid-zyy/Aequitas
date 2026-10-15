"""Pipeline v2 (faculty requirement R5): summarize and classify as separate, separately timed stages,
concurrent understanding workers, and per-stage timings in the store."""

import json
import random
import sqlite3
import threading
import time

import httpx
import pytest

from aequitas.config import CameraCfg, FusionCfg, LlmEndpoint, Settings
from aequitas.store.db import EventStore
from aequitas.store.query import QueryFilter
from aequitas.types import Candidate, Category, Status
from aequitas.understand import vlm_client
from aequitas.understand.fusion import fuse
from aequitas.understand.prompts import (
    CHOICES,
    LabelAnswer,
    SummaryAnswer,
    classify_text_prompt,
    summarize_prompt,
)
from aequitas.understand.stages import InOrder, understand
from aequitas.understand.vlm_client import (
    LlmClient,
    OpenAICompatClient,
    choice_payload,
    label_confidence,
)

from test_store_chat import store  # noqa: F401  (fixture)


class FakeVlm:
    """Stage 1 fake: returns a fixed summary, or None to simulate a failure."""

    def __init__(self, summary="A man falls and lies on the floor.", fail=False, delay=0.0):
        self.summary, self.fail, self.delay, self.calls = summary, fail, delay, 0

    def summarize(self, grid_jpeg, highlighted=True, view="cctv"):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            return None, 12.0
        return SummaryAnswer(summary=self.summary, visible_evidence="person lying down"), 1500.0


class FakeClassifier:
    def __init__(self, label="person_down", conf=0.9):
        self.label, self.conf, self.seen = label, conf, []

    def classify_text(self, summary, evidence="", categories=None, previous=None):
        self.seen.append((summary, evidence))
        return self.label, self.conf, 40.0


# ---------- prompts ----------

def test_summarize_prompt_never_shows_the_category_list():
    for highlighted in (True, False):
        p = summarize_prompt(highlighted)
        assert not any(c in p for c in CHOICES), "the summary must not be steered towards a label"
        assert "only what is visible" in p
    assert "Red boxes" in summarize_prompt(True) and "Red boxes" not in summarize_prompt(False)


def test_classify_prompt_is_text_only_and_lists_every_choice():
    p = classify_text_prompt("Two people shove each other.", "pushing")
    assert "Two people shove each other." in p and "pushing" in p
    assert all(c in p for c in CHOICES)
    assert "grid" not in p  # it classifies the text, not the image


# ---------- the two stages ----------

def test_understand_runs_summary_then_classifies_the_summary_text():
    vlm, clf = FakeVlm(), FakeClassifier("person_down", 0.83)
    u = understand(vlm, clf, b"jpeg", highlighted=True)
    assert clf.seen == [("A man falls and lies on the floor.", "person lying down")]
    assert u.answer.category == "person_down" and u.answer.confidence == 0.83
    assert u.summarize_ms == 1500.0 and u.classify_ms == 40.0 and u.vlm_ms == 1540.0


def test_failed_summary_skips_classification():
    clf = FakeClassifier()
    u = understand(FakeVlm(fail=True), clf, b"jpeg", highlighted=False)
    assert u.answer is None and u.summary is None and clf.seen == [] and u.classify_ms is None


def test_failed_classification_keeps_the_summary():
    u = understand(FakeVlm(), FakeClassifier(label=None, conf=None), b"jpeg", highlighted=False)
    assert u.answer is None and u.summary == "A man falls and lies on the floor."


def test_fusion_uses_the_classifier_label():
    cand = Candidate("cam1", Category.PERSON_DOWN, None, 0, 3000, 0.7, (1,))
    agree = understand(FakeVlm(), FakeClassifier("person_down"), b"j", True).answer
    assert fuse(cand, agree, FusionCfg()).status == Status.CONFIRMED
    none = understand(FakeVlm(), FakeClassifier("none"), b"j", True).answer
    assert fuse(cand, none, FusionCfg()).status == Status.REJECTED
    other = understand(FakeVlm(), FakeClassifier("fighting"), b"j", True).answer
    assert fuse(cand, other, FusionCfg()).status == Status.DISPUTED


class SchemaOnlyClient(LlmClient):
    """Backend without choice decoding: the label comes from a JSON-schema enum."""

    def __init__(self, reply):
        super().__init__(LlmEndpoint())
        self.reply = reply

    def _chat(self, messages, schema, name, max_tokens=None):
        assert name == "LabelAnswer" and "images" not in messages[0]
        return self.reply


def test_classify_text_with_a_schema_enum_has_no_confidence():
    label, conf, ms = SchemaOnlyClient('{"category": "fighting"}').classify_text("Two men grapple.")
    assert label == "fighting" and conf is None and ms >= 0
    assert set(LabelAnswer.model_json_schema()["properties"]["category"]["enum"]) == set(CHOICES)


def test_classify_text_rejects_invalid_output():
    assert SchemaOnlyClient('{"category": "alien_landing"}').classify_text("x")[0] is None


# ---------- vLLM structured outputs ----------

def test_choice_payload_uses_vllm_structured_outputs_and_logprobs():
    p = choice_payload("gemma", "label this", CHOICES)
    assert p["structured_outputs"] == {"choice": CHOICES}
    assert p["logprobs"] is True and p["temperature"] == 0
    assert "guided_choice" not in p  # removed in vLLM 0.12
    assert "images" not in json.dumps(p) and "image_url" not in json.dumps(p)


def test_label_confidence_is_the_product_of_token_probabilities():
    choice = {"logprobs": {"content": [{"token": "person", "logprob": -0.1}, {"token": "_down", "logprob": -0.2}]}}
    assert label_confidence(choice) == pytest.approx(round(2.718281828 ** -0.3, 4), abs=1e-4)
    assert label_confidence({"logprobs": None}) is None


def _openai_client(handler, choice_decoding=True):
    c = OpenAICompatClient(LlmEndpoint(backend="openai", host="http://127.0.0.1:8000", model="m",
                                       choice_decoding=choice_decoding))
    c.http = httpx.Client(transport=httpx.MockTransport(handler))
    return c


def test_openai_client_classifies_with_choice_and_confidence():
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "fighting"},
                                                      "logprobs": {"content": [{"token": "fighting", "logprob": -0.05}]}}]})

    label, conf, _ = _openai_client(handler).classify_text("Two men exchange punches.")
    assert label == "fighting" and conf == pytest.approx(0.9512, abs=1e-4)
    assert bodies[0]["structured_outputs"]["choice"] == CHOICES


def test_openai_client_falls_back_when_server_ignores_choice():
    bodies = []

    def handler(req):
        body = json.loads(req.content)
        bodies.append(body)
        content = '{"category": "tailgating"}' if "response_format" in body else "I think tailgating."
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    label, conf, _ = _openai_client(handler).classify_text("Two people enter one behind the other.")
    assert label == "tailgating" and conf is None
    assert "structured_outputs" in bodies[0] and "response_format" in bodies[1]


# ---------- in-order results under concurrency ----------

def test_in_order_applies_results_in_submission_order():
    order, applied = InOrder(), []
    seqs = [order.ticket("cam1") for _ in range(4)]
    order.finish("cam1", seqs[2], "c", applied.append)
    order.finish("cam1", seqs[1], "b", applied.append)
    assert applied == []  # waiting for 0
    order.skip("cam1", seqs[0], applied.append)  # dropped job
    assert applied == ["b", "c"]
    order.finish("cam1", seqs[3], "d", applied.append)
    assert applied == ["b", "c", "d"]


def test_in_order_keys_are_independent():
    order, applied = InOrder(), []
    a, b = order.ticket("cam1"), order.ticket("cam2")
    order.ticket("cam1")
    order.finish("cam2", b, "cam2-0", applied.append)
    order.finish("cam1", a, "cam1-0", applied.append)
    assert applied == ["cam2-0", "cam1-0"]


def test_in_order_under_real_threads():
    order, applied, lock = InOrder(), [], threading.Lock()
    tickets = [order.ticket("cam") for _ in range(60)]
    random.Random(1).shuffle(tickets)

    def apply(x):
        with lock:
            applied.append(x)

    def work(seq):
        time.sleep(random.random() * 0.005)
        order.finish("cam", seq, seq, apply)

    threads = [threading.Thread(target=work, args=(s,)) for s in tickets]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert applied == list(range(60))


# ---------- pipeline: periodic mode with concurrent workers ----------

@pytest.fixture
def periodic_pipeline(tmp_path, monkeypatch):
    from aequitas.pipeline import Pipeline

    class Fake(FakeVlm, FakeClassifier):
        def __init__(self, cfg):
            FakeVlm.__init__(self)
            FakeClassifier.__init__(self, "fighting", 0.7)
            self.cfg, self.model = cfg, cfg.model

        def ping(self):
            return True

    monkeypatch.setattr(vlm_client, "make_client", lambda cfg: Fake(cfg))
    s = Settings(data_dir=tmp_path)
    s.analysis.mode = "periodic"
    s.vlm.concurrency = 4
    store = EventStore(s.db_path)
    p = Pipeline(s, [CameraCfg(camera_id="cam1", name="Gate", source="x.mp4")], store, run_id="up1")
    return p, store


def test_periodic_windows_finishing_out_of_order_still_merge_into_one_event(periodic_pipeline):
    from aequitas.pipeline import VlmJob

    p, store = periodic_pipeline
    jobs = [VlmJob("cam1", b"jpeg", time.perf_counter_ns(), window=(i * 4000, (i + 1) * 4000),
                   seq=p.order.ticket("cam1")) for i in range(6)]
    random.Random(3).shuffle(jobs)
    threads = [threading.Thread(target=p._process_window, args=(j, 5.0)) for j in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rows = store._conn.execute("SELECT * FROM events").fetchall()
    assert len(rows) == 1  # one fighting event, extended by every window in order
    assert rows[0]["status"] == "confirmed" and rows[0]["start_ts"] == 0 and rows[0]["end_ts"] == 24_000
    t = store._conn.execute("SELECT * FROM event_timings").fetchone()
    assert t["summarize_ms"] == 1500.0 and t["classify_ms"] == 40.0 and t["vlm_ms"] == 1540.0
    summary = store.stage_summary("up1")
    assert summary["jobs"] == 6 and summary["ok"] == 6
    assert summary["summarize_ms"]["total"] == 9000.0 and summary["classify_ms"]["mean"] == 40.0


def test_pipeline_starts_one_thread_per_concurrency_slot(periodic_pipeline, monkeypatch):
    p, _ = periodic_pipeline
    p.cameras = []  # no streams: run() starts the workers and returns once the queue is empty
    started = []
    real = threading.Thread

    def spy(*a, **kw):
        started.append(kw.get("name"))
        return real(*a, **kw)

    monkeypatch.setattr(threading, "Thread", spy)
    p.run()
    assert sorted(n for n in started if n.startswith("vlm-")) == ["vlm-0", "vlm-1", "vlm-2", "vlm-3"]


# ---------- store: new columns and migration ----------

def test_old_database_gets_the_new_columns(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE event_timings (event_id INTEGER PRIMARY KEY, detect_ms REAL, trigger_ms REAL, queue_ms REAL,"
        " vlm_ms REAL, db_ms REAL, proc_ms REAL, e2e_ms REAL);"
        "CREATE TABLE uploads (upload_id TEXT PRIMARY KEY, camera_id TEXT NOT NULL, filename TEXT NOT NULL,"
        " path TEXT NOT NULL, start_ts INTEGER NOT NULL, duration_ms INTEGER, mode TEXT NOT NULL, vlm_model TEXT,"
        " status TEXT NOT NULL, frames_done INTEGER NOT NULL DEFAULT 0, frames_total INTEGER,"
        " events_found INTEGER NOT NULL DEFAULT 0, error TEXT, created_at INTEGER NOT NULL, finished_at INTEGER);"
    )
    con.close()
    store = EventStore(path)
    cols = {r["name"] for r in store._conn.execute("PRAGMA table_info(event_timings)")}
    assert {"summarize_ms", "classify_ms"} <= cols
    assert "started_at" in {r["name"] for r in store._conn.execute("PRAGMA table_info(uploads)")}
    EventStore(path)  # opening again is a no-op


def test_upload_json_reports_total_execution_time_and_stages(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from aequitas.chat.answer import ChatService
    from aequitas.chat.parser import QuestionParser
    from aequitas.web import server as web
    from aequitas.web.jobs import JobManager

    monkeypatch.setattr(JobManager, "submit", lambda self, upload_id: None)
    s = Settings(data_dir=tmp_path, auth={"required": False})
    store = EventStore(s.db_path)
    app = web.create_app(s, ChatService(store, QuestionParser(None)),
                         [CameraCfg(camera_id="cam1", name="Gate", source="x.mp4")])
    store.insert_upload({"upload_id": "u1", "camera_id": "cam1", "filename": "a.mp4", "path": "a.mp4",
                         "start_ts": 0, "duration_ms": 8000, "mode": "periodic", "vlm_model": None,
                         "status": "done", "created_at": 1000, "started_at": 2000, "finished_at": 14_500})
    store.record_stage("u1", "cam1", "window", 3.0, 1000.0, 50.0, True)
    store.record_stage("u1", "cam1", "window", 1.0, 3000.0, 30.0, False)
    u = next(x for x in TestClient(app).get("/api/uploads").json() if x["id"] == "u1")
    assert u["total_s"] == 12.5
    assert u["stages"]["jobs"] == 2 and u["stages"]["ok"] == 1
    assert u["stages"]["summarize_ms"] == {"total": 4000.0, "mean": 2000.0}


# ---------- stage 3: streamed, grounded chat ----------

class StreamingLlm:
    """Filter parsing via `result`; summary text streamed in pieces."""

    def __init__(self, result, pieces):
        self.result, self.pieces = result, pieces

    def structured(self, messages, model_cls, retries=1):
        return self.result if model_cls.__name__ == "LlmFilter" else None

    def stream_text(self, messages, max_tokens=256):
        yield from self.pieces


def _chat(store, llm, monkeypatch):
    from aequitas.chat.answer import ChatService
    from aequitas.chat.parser import QuestionParser
    from test_store_chat import NOW

    parser = QuestionParser(llm)
    monkeypatch.setattr(parser, "now", lambda: NOW)
    return ChatService(store, parser)


def _tailgating_filter():
    from aequitas.chat.parser import LlmFilter
    from test_store_chat import NOW

    return LlmFilter(categories=["tailgating"], cameras=["camera 2"], start=NOW.strftime("%Y-%m-%dT00:00"),
                     end=NOW.strftime("%Y-%m-%dT%H:%M"))


def test_stream_sends_rows_first_then_tokens_then_a_grounded_summary(store, monkeypatch):
    row_id = store.query(QueryFilter(
        categories=[Category.TAILGATING], camera_ids=["cam2"]))[0]["event_id"]
    llm = StreamingLlm(_tailgating_filter(), ["One tailgating ", f"event at the door [#{row_id}]."])
    events = list(_chat(store, llm, monkeypatch).ask_stream("Show all tailgating events from camera 2 today"))
    kinds = [e["type"] for e in events]
    assert kinds == ["retrieved", "token", "token", "summary", "done"]
    assert events[0]["total"] >= 1 and events[0]["rows"]
    assert events[3]["summary"] == f"One tailgating event at the door [#{row_id}]."
    assert events[4]["summary_ttft_ms"] is not None and events[4]["elapsed_ms"] >= 0


def test_stream_drops_a_summary_that_cites_unretrieved_events(store, monkeypatch):
    llm = StreamingLlm(_tailgating_filter(), ["Something happened [#999]."])
    events = list(_chat(store, llm, monkeypatch).ask_stream("Show all tailgating events from camera 2 today"))
    assert next(e for e in events if e["type"] == "summary")["summary"] is None


def test_stream_endpoint_emits_server_sent_events(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from aequitas.chat.answer import ChatService
    from aequitas.chat.parser import QuestionParser
    from aequitas.web import server as web
    from aequitas.web.jobs import JobManager

    monkeypatch.setattr(JobManager, "submit", lambda self, upload_id: None)
    s = Settings(data_dir=tmp_path, auth={"required": False})
    st = EventStore(s.db_path)
    app = web.create_app(s, ChatService(st, QuestionParser(None)), [CameraCfg(camera_id="cam1", name="Gate",
                                                                               source="x.mp4")])
    r = TestClient(app).post("/api/ask/stream", json={"question": "any fights today?"})
    assert r.headers["content-type"].startswith("text/event-stream")
    names = [line.split(": ", 1)[1] for line in r.text.splitlines() if line.startswith("event: ")]
    assert names == ["retrieved", "done"]  # no LLM: no summary stage


def test_sse_parser_reads_openai_stream_chunks():
    from aequitas.understand.vlm_client import sse_deltas

    lines = ['data: {"choices":[{"delta":{"role":"assistant"}}]}', "",
             'data: {"choices":[{"delta":{"content":"Two "}}]}', ": keep-alive",
             'data: {"choices":[{"delta":{"content":"events."}}]}', "data: [DONE]",
             'data: {"choices":[{"delta":{"content":"ignored"}}]}']
    assert "".join(sse_deltas(lines)) == "Two events."


# ---------- latency levers ----------

def test_summary_calls_are_capped_and_short():
    from aequitas.understand.vlm_client import openai_payload

    assert openai_payload("m", [{"role": "user", "content": "x"}], {}, "S", 128)["max_tokens"] == 128
    assert "max_tokens" not in openai_payload("m", [{"role": "user", "content": "x"}], {}, "S")
    assert "at most 35 words" in summarize_prompt(False)

    seen = []

    class Cap(LlmClient):
        def _chat(self, messages, schema, name, max_tokens=None):
            seen.append((name, max_tokens))
            return '{"summary": "A man walks.", "visible_evidence": "walking"}'

    c = Cap(LlmEndpoint(summary_max_tokens=100, max_tokens=300))
    c.summarize(b"jpeg")
    c.structured([{"role": "user", "content": "q"}], SummaryAnswer)
    assert seen == [("SummaryAnswer", 100), ("SummaryAnswer", 300)]


def test_window_motion_separates_static_and_moving_scenes():
    import numpy as np

    from aequitas.ingest.ring_buffer import BufferedFrame
    from aequitas.understand.keyframes import window_motion

    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    static = [BufferedFrame(i, base.copy(), 1.0, {}) for i in range(4)]
    moving = [BufferedFrame(i, np.roll(base, 20 * i, axis=1), 1.0, {}) for i in range(4)]
    assert window_motion(static) < 0.001 < 0.05 < window_motion(moving)
    assert window_motion(static[:1]) == 1.0  # cannot judge: never skip


def test_static_windows_skip_the_vlm(periodic_pipeline):
    import numpy as np

    from aequitas.ingest.ring_buffer import RingBuffer
    from aequitas.types import Frame

    p, store = periodic_pipeline
    p.s.analysis.skip_static_below = 0.01
    p.s.analysis.window_s = 1.0
    p.s.analysis.max_skip_s = 3.0
    cam = p.cameras[0]
    step = p._periodic_step(cam, RingBuffer(12, 320))
    img = np.random.default_rng(1).integers(0, 255, (60, 80, 3), dtype=np.uint8)  # textured, unchanging
    for i in range(41):  # 10 s of a static scene at 4 fps
        step(Frame("cam1", i, i * 250, 0, img))
        while len(p.queue):  # play the worker: take queued windows so new ones can be emitted
            p.queue.get(0)
    s = store.stage_summary("up1")
    assert s["skipped_static"] >= 5  # most windows skipped...
    rows = store._conn.execute("SELECT COUNT(*) FROM stage_timings WHERE kind='skipped'").fetchone()[0]
    assert rows == s["skipped_static"]
    # ...but the VLM still looks at least every max_skip_s: windows queued = 10 s / ~3 s + first one
    assert p.order.ticket("cam1") >= 3


def test_blank_windows_never_reach_the_models(periodic_pipeline):
    """Seen live: all-black webcam frames gave a 'confirmed fighting' (clip model 0.96, Gemma invented people)."""
    import numpy as np

    from aequitas.ingest.ring_buffer import BufferedFrame, RingBuffer
    from aequitas.types import Frame
    from aequitas.understand.keyframes import window_detail

    assert window_detail([BufferedFrame(0, np.zeros((48, 64, 3), np.uint8), 1.0, {})]) == 0.0
    noise = np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    assert window_detail([BufferedFrame(0, np.zeros((48, 64, 3), np.uint8), 1.0, {}),
                          BufferedFrame(1, noise, 1.0, {})]) > 40  # one real frame is enough to look
    p, store = periodic_pipeline
    p.s.analysis.window_s = 1.0
    step = p._periodic_step(p.cameras[0], RingBuffer(12, 320))
    for i in range(21):  # 5 s of a covered camera
        step(Frame("cam1", i, i * 250, 0, np.zeros((60, 80, 3), np.uint8)))
    assert len(p.queue) == 0 and store.stage_summary("up1")["skipped_blank"] >= 4
    for i in range(21, 30):  # the lens is uncovered: windows go to the VLM again
        step(Frame("cam1", i, i * 250, 0, noise))
    assert len(p.queue) == 1


def test_file_sources_analyse_every_window_even_with_a_slow_vlm(periodic_pipeline):
    """Regression: uploads used to stretch windows while the VLM was busy, so most of a video was never
    analysed (and total time looked fast for the wrong reason)."""
    import numpy as np

    from aequitas.ingest.ring_buffer import RingBuffer
    from aequitas.types import Frame

    p, store = periodic_pipeline
    p.s.analysis.window_s = 1.0
    p.s.vlm.concurrency = 2
    p.vlm.delay = 0.05  # VLM much slower than reading frames
    done = threading.Event()
    workers = [threading.Thread(target=p._vlm_loop, args=(done,)) for _ in range(2)]
    for w in workers:
        w.start()
    step = p._periodic_step(p.cameras[0], RingBuffer(12, 320), live=False)
    rng = np.random.default_rng(1)
    for i in range(41):  # 10 s at 4 fps, fed as fast as possible
        step(Frame("cam1", i, i * 250, 0, rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)))
    done.set()
    for w in workers:
        w.join()
    n = store._conn.execute("SELECT COUNT(*) FROM stage_timings WHERE kind='window'").fetchone()[0]
    assert n == 10  # one per second of video, none stretched or dropped
    assert store._conn.execute("SELECT COUNT(*) FROM dropped_candidates").fetchone()[0] == 0


def test_chat_reports_its_response_time_per_step(store, monkeypatch):
    svc = _chat(store, StreamingLlm(_tailgating_filter(), ["One event [#1]."]), monkeypatch)
    ans = svc.ask("Show all tailgating events from camera 2 today")
    assert {"parse_ms", "query_ms", "total_ms"} <= set(ans.timings)
    done = list(svc.ask_stream("Show all tailgating events from camera 2 today"))[-1]
    assert {"parse_ms", "query_ms", "summary_ttft_ms", "summary_ms", "elapsed_ms"} <= set(done)


# ---------- violence clip model as the second signal (periodic / live mode) ----------

def test_fuse_window_rules():
    from aequitas.understand.fusion import fuse_window
    from aequitas.understand.prompts import VlmAnswer

    vague = VlmAnswer(category="none", summary="People are gathered in an outdoor area.")
    ans, src = fuse_window(vague, 0.93, 0.6)
    assert ans.category == "fighting" and ans.confidence == 0.93 and src == "clip_model"
    assert ans.summary == vague.summary  # the description is kept; only the label is upgraded
    fight = VlmAnswer(category="fighting", summary="Two men exchange punches.")
    assert fuse_window(fight, 0.9, 0.6) == (fight, "both")
    assert fuse_window(fight, 0.1, 0.6) == (fight, "vlm")  # a fight the VLM saw is never downgraded
    assert fuse_window(vague, 0.3, 0.6) == (vague, "vlm")
    assert fuse_window(vague, None, 0.6) == (vague, "vlm")
    ans, src = fuse_window(None, 0.95, 0.6)  # VLM failed but the clip model is sure
    assert ans.category == "fighting" and "clip model" in ans.summary and src == "clip_model"


def test_fuse_window_keeps_specific_labels_and_needs_more_from_the_clip_model_alone():
    from aequitas.understand.fusion import fuse_window
    from aequitas.understand.prompts import VlmAnswer

    arrest = VlmAnswer(category="arrest", summary="Officers hold a man's arms behind his back.")
    assert fuse_window(arrest, 0.98, 0.51, 0.9) == (arrest, "vlm+clip")  # a struggle is not relabelled fighting
    assert fuse_window(arrest, 0.2, 0.51, 0.9) == (arrest, "vlm")
    busy = VlmAnswer(category="none", summary="A crowded shopping street.")
    assert fuse_window(busy, 0.85, 0.51, 0.9) == (busy, "vlm")  # clip model alone, not sure enough
    ans, src = fuse_window(busy, 0.97, 0.51, 0.9)
    assert ans.category == "fighting" and src == "clip_model"
    fight = VlmAnswer(category="fighting", summary="Two men exchange punches.")
    assert fuse_window(fight, 0.6, 0.51, 0.9) == (fight, "both")  # agreement needs only the calibrated threshold


def test_window_with_high_violence_score_becomes_a_fighting_event(periodic_pipeline):
    from aequitas.pipeline import VlmJob
    from aequitas.understand.stages import Understanding
    from aequitas.understand.prompts import VlmAnswer

    p, store = periodic_pipeline

    class FakeViolence:
        threshold = 0.6

    p.violence_model = FakeViolence()
    for i in range(2):
        job = VlmJob("cam1", b"jpeg", time.perf_counter_ns(), window=(i * 4000, (i + 1) * 4000),
                     seq=p.order.ticket("cam1"), violence=0.9)
        u = Understanding(VlmAnswer(category="none", summary="People are gathered."), "People are gathered.", 800, 40)
        p._apply_window_result(job, u, 1.0)
    w = store.recent_windows("cam1", 2)
    assert [x["label"] for x in w] == ["fighting", "fighting"] and w[0]["source"] == "clip_model"
    assert w[0]["violence"] == 0.9
    ev = store._conn.execute("SELECT category, status FROM events").fetchall()
    assert [tuple(e) for e in ev] == [("fighting", "confirmed")]


# ---------- cascade: a bigger model re-checks flagged windows ----------

def test_escalation_rule_only_rechecks_when_it_matters():
    from aequitas.understand.stages import Understanding, needs_escalation
    from aequitas.understand.prompts import VlmAnswer

    calm = Understanding(VlmAnswer(category="none", summary="Traffic."), "Traffic.", 800, 40)
    flagged = Understanding(VlmAnswer(category="arrest", summary="x"), "x", 800, 40)
    failed = Understanding(None, None, 800, None)
    assert not needs_escalation(calm, None, 0.35)
    assert not needs_escalation(calm, 0.2, 0.35)
    assert needs_escalation(calm, 0.5, 0.35)  # clip model unsure or worse
    assert needs_escalation(flagged, None, 0.35) and needs_escalation(failed, None, 0.35)


def test_escalation_replaces_the_verdict_and_adds_the_time():
    from aequitas.understand.stages import Understanding, escalate
    from aequitas.understand.prompts import VlmAnswer

    class Big(FakeVlm, FakeClassifier):
        def __init__(self, label, fail=False):
            FakeVlm.__init__(self, summary="A man is handcuffed by two officers.", fail=fail)
            FakeClassifier.__init__(self, label, 0.95)
            self.model = "big-model"

    small = Understanding(VlmAnswer(category="fighting", summary="People interacting."), "People interacting.", 800, 40)
    u = escalate(small, Big("arrest"), b"jpeg", False, None)
    assert u.answer.category == "arrest" and u.escalated == "big-model"
    assert u.summarize_ms == 2300 and u.classify_ms == 80  # both models really ran
    kept = escalate(small, Big("arrest", fail=True), b"jpeg", False, None)
    assert kept.answer.category == "fighting" and kept.escalated is None


def test_pipeline_rechecks_flagged_windows_with_the_big_model(periodic_pipeline):
    from aequitas.pipeline import VlmJob

    p, store = periodic_pipeline  # the fake small model labels everything 'fighting'

    class Big(FakeVlm, FakeClassifier):
        def __init__(self):
            FakeVlm.__init__(self, summary="Two friends hug.")
            FakeClassifier.__init__(self, "none", 0.9)
            self.model = "big-model"

    p.big = Big()
    job = VlmJob("cam1", b"jpeg", time.perf_counter_ns(), window=(0, 4000), seq=p.order.ticket("cam1"))
    p._process_window(job, 1.0)
    w = store.recent_windows("cam1", 1)[0]
    assert w["label"] == "none" and "rechecked:big-model" in w["source"]
    assert store._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0  # false alarm avoided


def test_missing_violence_model_folder_does_not_stop_the_pipeline(tmp_path, monkeypatch):
    from aequitas.pipeline import Pipeline

    class Fake(FakeVlm):
        def __init__(self, cfg):
            FakeVlm.__init__(self)
            self.cfg, self.model = cfg, cfg.model

        def ping(self):
            return True

    monkeypatch.setattr(vlm_client, "make_client", lambda cfg: Fake(cfg))
    s = Settings(data_dir=tmp_path, analysis={"mode": "periodic"}, models={"violence_dir": str(tmp_path / "nope")})
    p = Pipeline(s, [], EventStore(s.db_path))
    assert p.violence_model is None


def test_label_step_can_see_the_previous_window_and_guided_checks_reach_it():
    from aequitas.understand.prompts import SummaryAnswer, classify_text_prompt, summarize_prompt
    from aequitas.understand.stages import understand

    p = classify_text_prompt("Officers lead a man to a van.", previous="Two officers hold a man's arms.")
    assert "The 4 seconds before" in p and "hold a man's arms" in p
    assert "4 seconds before" not in classify_text_prompt("Officers lead a man to a van.")
    assert "Check questions" in summarize_prompt(False, guided=True) and "Check questions" not in summarize_prompt(False)

    class V:
        def summarize(self, grid, highlighted, view):
            return SummaryAnswer(summary="Two men near a car.", visible_evidence="car", checks="punching"), 10.0

    class C:
        seen = None

        def classify_text(self, summary, evidence, categories, previous=None):
            C.seen = (evidence, previous)
            return "fighting", None, 5.0

    u = understand(V(), C(), b"x", False, previous="earlier")
    assert C.seen == ("car. Yes to: punching", "earlier") and u.answer.category == "fighting"


def test_scene_gate_skips_only_when_every_signal_says_ordinary():
    from aequitas.understand.gate import should_skip

    kw = dict(below=0.3, violence_band_low=0.35, max_skip_ms=30_000)
    assert should_skip(0.05, None, 4000, **kw)  # clearly normal, VLM looked 4 s ago
    assert not should_skip(0.05, None, None, **kw)  # a camera's first window always goes to the VLM
    assert not should_skip(0.9, None, 4000, **kw)  # looks like an incident
    assert not should_skip(0.05, 0.5, 4000, **kw)  # the violence clip model sees something
    assert not should_skip(0.05, None, 30_000, **kw)  # the VLM looks at least every 30 s
    assert not should_skip(None, None, 4000, **kw)  # gate failed to score
