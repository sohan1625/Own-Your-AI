"""Shared test setup: API tests never call the real embedding model."""

import hashlib
import math
import re

import pytest

from src import api


def fake_embedding(text: str, dims: int = 64) -> list[float]:
    """Deterministic bag-of-words vector, so texts sharing words score as similar."""
    vector = [0.0] * dims
    for token in re.findall(r"\w+", text.lower()):
        vector[int(hashlib.sha256(token.encode()).hexdigest(), 16) % dims] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        vector[0] = 1.0
        return vector
    return [value / norm for value in vector]


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    monkeypatch.setattr(api, "embed_text", lambda text, **kwargs: fake_embedding(text))
    monkeypatch.setattr(
        api,
        "build_chunk_embeddings",
        lambda chunks, **kwargs: [fake_embedding(chunk["text"]) for chunk in chunks],
    )
