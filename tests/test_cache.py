"""Answer cache tests. Each test runs in an isolated PostgreSQL schema (never the main one)."""

import psycopg
import pytest
from fastapi.testclient import TestClient

from src import api, cache
from tests.test_api import _make_pdf, isolated_database  # noqa: F401 (fixture)

REFUSAL = "not found in your documents"


@pytest.fixture
def client(isolated_database, monkeypatch, tmp_path):  # noqa: F811
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(cache, "is_enabled", lambda _: True)
    calls = {"generate": 0, "search": 0}
    real_search = api.bm25_search

    def counting_search(*args, **kwargs):
        calls["search"] += 1
        return real_search(*args, **kwargs)

    def generate(question, chunks, **kwargs):
        calls["generate"] += 1
        return f"The fact is stated here [{chunks[0]['chunk_id']}]"

    monkeypatch.setattr(api, "bm25_search", counting_search)
    monkeypatch.setattr(api, "generate_answer", generate)
    with TestClient(api.app) as test_client:
        test_client.calls = calls
        test_client.database_url = isolated_database
        yield test_client


def _upload(client, *names):
    files = [
        ("files", (f"{name}.pdf", _make_pdf(f"The fact appears in {name}."), "application/pdf"))
        for name in names
    ]
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    return [document["id"] for document in response.json()["documents"]]


def _ask(client, question, doc_ids):
    response = client.post("/ask", json={"question": question, "doc_ids": doc_ids})
    assert response.status_code == 200
    return response.json()


def _cache_rows(client):
    with psycopg.connect(client.database_url) as connection:
        return connection.execute("SELECT key, doc_ids FROM answer_cache").fetchall()


def test_same_question_twice_is_served_from_cache(client):
    doc_ids = _upload(client, "alpha")

    first = _ask(client, "What is the fact?", doc_ids)
    second = _ask(client, "  what IS   the fact?", doc_ids)

    assert first["cached"] is False
    assert second["cached"] is True
    assert second["answer"] == first["answer"]
    assert second["citations"] == first["citations"]
    assert client.calls == {"generate": 1, "search": 1}
    # A cache hit is still recorded in its conversation history.
    messages = client.get(f"/conversations/{second['conversation_id']}/messages").json()
    assert [message["content"] for message in messages] == ["what IS   the fact?", first["answer"]]


def test_different_document_selection_is_a_miss(client):
    alpha, beta = _upload(client, "alpha", "beta")

    first = _ask(client, "What is the fact?", [alpha])
    second = _ask(client, "What is the fact?", [alpha, beta])
    reordered = _ask(client, "What is the fact?", [beta, alpha])

    assert first["cached"] is False
    assert second["cached"] is False
    assert reordered["cached"] is True
    assert client.calls["generate"] == 2
    assert len(_cache_rows(client)) == 2


def test_refusal_is_not_stored(client, monkeypatch):
    doc_ids = _upload(client, "alpha")
    monkeypatch.setattr(api, "generate_answer", lambda *args, **kwargs: REFUSAL)

    first = _ask(client, "What is the capital of Mars?", doc_ids)
    second = _ask(client, "What is the capital of Mars?", doc_ids)

    assert first["answer"] == REFUSAL
    assert first["cached"] is False
    assert second["cached"] is False
    assert _cache_rows(client) == []


def test_deleting_a_document_removes_its_cached_answers(client):
    alpha, beta = _upload(client, "alpha", "beta")
    _ask(client, "What is the fact?", [alpha])
    _ask(client, "What is the fact?", [alpha, beta])
    _ask(client, "What is the fact?", [beta])
    assert len(_cache_rows(client)) == 3

    assert client.delete(f"/documents/{alpha}").status_code == 200

    remaining = _cache_rows(client)
    assert len(remaining) == 1
    assert alpha not in remaining[0][1]
    assert _ask(client, "What is the fact?", [beta])["cached"] is True


def test_follow_up_with_history_bypasses_cache(client):
    doc_ids = _upload(client, "alpha")
    first = _ask(client, "What is the fact?", doc_ids)

    response = client.post(
        "/ask",
        json={
            "question": "What is the fact?",
            "doc_ids": doc_ids,
            "conversation_id": first["conversation_id"],
        },
    )

    assert response.json()["cached"] is False
    assert client.calls["generate"] == 2


def test_disabled_cache_is_skipped(client, monkeypatch):
    monkeypatch.setattr(cache, "is_enabled", lambda _: False)
    doc_ids = _upload(client, "alpha")

    first = _ask(client, "What is the fact?", doc_ids)
    second = _ask(client, "What is the fact?", doc_ids)

    assert first["cached"] is False
    assert second["cached"] is False
    assert _cache_rows(client) == []


def test_config_enables_cache():
    assert cache.is_enabled(api.ROOT / "config.toml") is True
