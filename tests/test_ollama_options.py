"""Ollama request options: the MTP drafter setting is sent only when configured; call stats are kept."""

from aequitas.config import LlmEndpoint
from aequitas.understand.vlm_client import OllamaClient


class FakeOllama:
    def __init__(self):
        self.calls = []

    def chat(self, **kw):
        self.calls.append(kw)
        if kw.get("stream"):
            return iter([{"message": {"content": "a"}, "done": False},
                         {"message": {"content": "b"}, "done": True, "eval_count": 2, "eval_duration": 10}])
        return {"message": {"content": "{}"}, "eval_count": 5, "eval_duration": 50, "prompt_eval_count": 9}


def client(**cfg) -> tuple[OllamaClient, FakeOllama]:
    c = OllamaClient(LlmEndpoint(**cfg))
    c.client = FakeOllama()
    return c, c.client


def test_draft_option_omitted_by_default():
    c, fake = client()
    c._chat([{"role": "user", "content": "x"}], {}, "x")
    assert "draft_num_predict" not in fake.calls[0]["options"]
    assert c.last_stats["eval_count"] == 5 and c.last_stats["prompt_eval_count"] == 9


def test_draft_option_sent_when_set_including_zero():
    for n in (0, 4):
        c, fake = client(draft_num_predict=n)
        c._chat([{"role": "user", "content": "x"}], {}, "x")
        assert fake.calls[0]["options"]["draft_num_predict"] == n
        assert "".join(c.stream_text([{"role": "user", "content": "x"}])) == "ab"
        assert fake.calls[1]["options"]["draft_num_predict"] == n
        assert c.last_stats["eval_count"] == 2
