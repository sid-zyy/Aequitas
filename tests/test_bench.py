"""`minerva bench` against a fake OpenAI-compatible streaming server (no models)."""

import base64
import json

import httpx
import pytest

from minerva.eval import bench
from minerva.understand.prompts import CHOICES


def test_every_summarize_request_gets_a_different_image():
    imgs = set()
    for i in range(5):
        body = bench.build_request("summarize", "m", i, seed=7, max_tokens=64)
        imgs.add(body["messages"][0]["content"][1]["image_url"]["url"])
    assert len(imgs) == 5
    other_run = bench.build_request("summarize", "m", 0, seed=8, max_tokens=64)
    assert other_run["messages"][0]["content"][1]["image_url"]["url"] not in imgs
    assert base64.b64decode(next(iter(imgs)).split(",", 1)[1])[:2] == b"\xff\xd8"  # a JPEG


def test_requests_use_each_task_shape():
    c = bench.build_request("classify", "m", 0, 1, 16)
    assert c["structured_outputs"] == {"choice": CHOICES} and isinstance(c["messages"][0]["content"], str)
    s = bench.build_request("summarize", "m", 0, 1, 160)
    assert s["response_format"]["json_schema"]["name"] == "SummaryAnswer"
    for task in bench.TASKS:
        b = bench.build_request(task, "m", 0, 1, 10)
        assert b["stream"] is True and b["stream_options"] == {"include_usage": True}


def test_spec_metrics_parsing_and_acceptance_rate():
    before = bench.parse_spec_metrics(
        'vllm:spec_decode_num_drafts_total{model_name="g"} 10\n'
        'vllm:spec_decode_num_draft_tokens_total{model_name="g"} 40\n'
        'vllm:spec_decode_num_accepted_tokens_total{model_name="g"} 20\n'
        '# HELP something else\nvllm:other 5\n')
    assert before == {"drafts": 10, "draft_tokens": 40, "accepted_tokens": 20}
    after = {"drafts": 30, "draft_tokens": 120, "accepted_tokens": 80}
    acc = bench.acceptance(before, after)
    assert acc["acceptance_rate"] == 0.75 and acc["mean_accepted_per_draft"] == 3.0
    assert bench.acceptance({}, {}) is None  # no speculation


def _fake_server(pieces=("Hello", " there", "."), fail=False):
    def handler(req):
        if req.url.path.endswith("/metrics"):
            return httpx.Response(404)
        if fail:
            return httpx.Response(500, text="boom")
        lines = [f'data: {json.dumps({"choices": [{"delta": {"content": p}}]})}' for p in pieces]
        lines.append(f'data: {json.dumps({"choices": [], "usage": {"completion_tokens": len(pieces)}})}')
        lines.append("data: [DONE]")
        return httpx.Response(200, text="\n\n".join(lines) + "\n\n",
                              headers={"content-type": "text/event-stream"})
    return handler


def test_run_one_measures_ttft_tpot_and_tokens():
    http = httpx.Client(transport=httpx.MockTransport(_fake_server()))
    s = bench.run_one(http, "http://127.0.0.1:8000/v1", bench.build_request("chat", "m", 0, 1, 10), "chat")
    assert s.ok and s.out_tokens == 3 and s.ttft_ms is not None and s.tpot_ms is not None
    assert s.total_ms >= s.ttft_ms


def test_run_one_reports_errors():
    http = httpx.Client(transport=httpx.MockTransport(_fake_server(fail=True)))
    s = bench.run_one(http, "http://127.0.0.1:8000/v1", bench.build_request("chat", "m", 0, 1, 10), "chat")
    assert not s.ok and s.error


def test_run_bench_writes_json_and_markdown(tmp_path, monkeypatch):
    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(_fake_server())))
    rep = bench.run_bench("http://127.0.0.1:8000", "m", ["classify", "chat"], [1, 2], 3, label="bf16 none",
                          out_dir=tmp_path)
    assert rep["results"]["chat"]["2"]["ok"] == 3 and rep["results"]["classify"]["1"]["speculation"] is None
    md = (tmp_path / rep["_path"].split("\\")[-1].split("/")[-1]).read_text(encoding="utf-8")
    assert "| chat | 2 | 3/3 |" in md
    assert list(tmp_path.glob("bench-bf16_none-*.json"))


def test_bench_refuses_remote_hosts():
    with pytest.raises(ValueError):
        bench.run_bench("https://api.example.com", "m", ["chat"], [1], 1)
