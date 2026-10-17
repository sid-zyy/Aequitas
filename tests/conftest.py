import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(autouse=True)
def _no_embedding_model_download(monkeypatch):
    """The web app opens the chat's vector index on start; tests get the word-hash embedder, never the real model."""
    from helpers import WordEmbedder

    monkeypatch.setattr("aequitas.chat.vectors.make_embedder", lambda *a, **k: WordEmbedder())
