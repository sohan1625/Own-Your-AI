"""Hybrid retrieval tests. Database tests run in an isolated PostgreSQL schema."""

import psycopg
import pytest
from fastapi.testclient import TestClient

from src import api, cache
from src.fuse import reciprocal_rank_fusion
from tests.conftest import fake_embedding
from tests.test_api import _make_pdf, isolated_database  # noqa: F401 (fixture)


def test_rrf_rewards_items_ranked_well_by_both_lists():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["d", "b", "e"]], k=60)
    assert fused[0] == "b"  # 2nd in both beats 1st in only one list
    assert fused[1:3] == ["a", "d"]
    assert set(fused) == {"a", "b", "c", "d", "e"}


def test_rrf_keeps_first_seen_order_on_ties():
    assert reciprocal_rank_fusion([["a"], ["b"]], k=60) == ["a", "b"]


@pytest.fixture
def client(isolated_database, monkeypatch, tmp_path):  # noqa: F811
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(cache, "is_enabled", lambda _: False)
    captured = {}

    def generate(question, chunks, **kwargs):
        captured["chunk_ids"] = [chunk["chunk_id"] for chunk in chunks]
        captured["top_score"] = kwargs.get("top_score")
        return f"Answer [{chunks[0]['chunk_id']}]"

    monkeypatch.setattr(api, "generate_answer", generate)
    with TestClient(api.app) as test_client:
        test_client.captured = captured
        test_client.database_url = isolated_database
        yield test_client


def _upload(client, *texts):
    files = [
        ("files", (f"doc{number}.pdf", _make_pdf(text), "application/pdf"))
        for number, text in enumerate(texts)
    ]
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    return [document["id"] for document in response.json()["documents"]]


def _chunk_ids_by_doc(client):
    with psycopg.connect(client.database_url) as connection:
        rows = connection.execute("SELECT doc_id, id FROM chunks").fetchall()
    return {doc_id: chunk_id for doc_id, chunk_id in rows}


def _missing_embeddings(client):
    with psycopg.connect(client.database_url) as connection:
        return connection.execute(
            "SELECT count(*) FROM chunks WHERE embedding IS NULL"
        ).fetchone()[0]


def test_upload_stores_embeddings_and_vector_search_ranks_them(client):
    firewall, router = _upload(
        client,
        "A firewall filters incoming and outgoing network traffic.",
        "A router forwards packets between networks using IP addresses.",
    )
    assert _missing_embeddings(client) == 0

    chunk_ids = _chunk_ids_by_doc(client)
    results = api.db.vector_search(
        fake_embedding("firewall traffic filters"), [firewall, router], top_n=2
    )
    assert [chunk_id for chunk_id, _ in results] == [chunk_ids[firewall], chunk_ids[router]]
    assert results[0][1] > results[1][1]
    # Restricted to the selected documents.
    only_router = api.db.vector_search(fake_embedding("firewall"), [router], top_n=5)
    assert [chunk_id for chunk_id, _ in only_router] == [chunk_ids[router]]


def test_ask_fuses_keyword_and_vector_results(client, monkeypatch):
    keyword_doc, vector_doc = _upload(
        client,
        "Firewalls block traffic.",
        "Packet filtering devices guard a network perimeter.",
    )
    chunk_ids = _chunk_ids_by_doc(client)
    # Vector search finds a chunk that shares no words with the question.
    monkeypatch.setattr(
        api,
        "_vector_ranking",
        lambda query, doc_ids, top_n: [(chunk_ids[vector_doc], 0.83), (chunk_ids[keyword_doc], 0.41)],
    )
    monkeypatch.setattr(api, "get_generation_top_k", lambda _: 2)

    response = client.post(
        "/ask",
        json={"question": "explain firewalls", "doc_ids": [keyword_doc, vector_doc]},
    )

    assert response.status_code == 200
    assert set(client.captured["chunk_ids"]) == {chunk_ids[keyword_doc], chunk_ids[vector_doc]}
    assert client.captured["top_score"] == 0.83


def test_ask_falls_back_to_keyword_search_when_embedding_fails(client, monkeypatch):
    (doc_id,) = _upload(client, "Firewalls block traffic.")

    def embedding_down(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(api, "embed_text", embedding_down)

    response = client.post("/ask", json={"question": "explain firewalls", "doc_ids": [doc_id]})

    assert response.status_code == 200
    assert client.captured["chunk_ids"] == [_chunk_ids_by_doc(client)[doc_id]]
    assert client.captured["top_score"] is None  # no vector score, so no threshold check


def test_keyword_method_skips_vector_search(client, monkeypatch):
    (doc_id,) = _upload(client, "Firewalls block traffic.")
    monkeypatch.setattr(api, "_retrieval_method", lambda: "keyword")

    def unexpected(*args):
        raise AssertionError("vector search should not run in keyword mode")

    monkeypatch.setattr(api, "_vector_ranking", unexpected)

    response = client.post("/ask", json={"question": "explain firewalls", "doc_ids": [doc_id]})

    assert response.status_code == 200
    assert client.captured["top_score"] is None


def test_startup_backfills_missing_embeddings(isolated_database, monkeypatch, tmp_path):  # noqa: F811
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path / "uploads")

    def embedding_down(chunks, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(api, "build_chunk_embeddings", embedding_down)
    with TestClient(api.app) as client:
        response = client.post(
            "/upload",
            files=[("files", ("doc.pdf", _make_pdf("Firewalls block traffic."), "application/pdf"))],
        )
        assert response.status_code == 200  # upload still works without embeddings

    def count_missing():
        with psycopg.connect(isolated_database) as connection:
            return connection.execute(
                "SELECT count(*) FROM chunks WHERE embedding IS NULL"
            ).fetchone()[0]

    assert count_missing() == 1

    monkeypatch.setattr(
        api,
        "build_chunk_embeddings",
        lambda chunks, **kwargs: [fake_embedding(chunk["text"]) for chunk in chunks],
    )
    with TestClient(api.app):
        pass

    assert count_missing() == 0
