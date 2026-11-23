"""scripts/rechunk.py: rebuild a stored document's chunks in place (isolated schema, fake embeddings)."""

import psycopg
import pytest
from fastapi.testclient import TestClient

from scripts import rechunk
from src import api, cache
from tests.test_api import _make_pdf, isolated_database  # noqa: F401 (fixture)


@pytest.fixture
def client(isolated_database, monkeypatch, tmp_path):  # noqa: F811
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(cache, "is_enabled", lambda _: True)
    monkeypatch.setattr(api, "generate_answer", lambda q, chunks, **kw: f"Stated [{chunks[0]['chunk_id']}]")
    with TestClient(api.app) as test_client:
        test_client.database_url = isolated_database
        yield test_client


def _rows(client, sql, *params):
    with psycopg.connect(client.database_url) as connection:
        return connection.execute(sql, params).fetchall()


def test_rechunk_replaces_chunks_and_clears_cache_keeps_history(client):
    upload = client.post("/upload", files=[("files", ("a.pdf", _make_pdf("Oak was renamed to Java."), "application/pdf"))])
    doc_id = upload.json()["documents"][0]["id"]
    first = client.post("/ask", json={"question": "Why was Oak renamed?", "doc_ids": [doc_id]}).json()
    old_ids = {r[0] for r in _rows(client, "SELECT id FROM chunks WHERE doc_id = %s", doc_id)}
    assert _rows(client, "SELECT count(*) FROM answer_cache")[0][0] == 1

    assert rechunk.main([]) == 0

    new = _rows(client, "SELECT id, embedding IS NOT NULL FROM chunks WHERE doc_id = %s", doc_id)
    assert new and {r[0] for r in new}.isdisjoint(old_ids)
    assert all(has_embedding for _, has_embedding in new)
    assert _rows(client, "SELECT chunk_count FROM documents WHERE id = %s", doc_id)[0][0] == len(new)
    assert _rows(client, "SELECT count(*) FROM answer_cache")[0][0] == 0
    assert len(client.get(f"/conversations/{first['conversation_id']}/messages").json()) == 2
    api._invalidate_document_cache()
    again = client.post("/ask", json={"question": "Why was Oak renamed?", "doc_ids": [doc_id]}).json()
    assert again["cached"] is False and again["citations"][0]["file"] == "a.pdf"


def test_rechunk_matches_source_by_hash_when_upload_missing(client, tmp_path):
    pdf = _make_pdf("Oak was renamed to Java.")
    doc_id = client.post("/upload", files=[("files", ("a.pdf", pdf, "application/pdf"))]).json()["documents"][0]["id"]
    (api.UPLOAD_DIR / f"{doc_id}.pdf").unlink()
    assert rechunk.find_sources(api.db.list_documents(), None) == {}
    source_dir = tmp_path / "pdfs"
    source_dir.mkdir()
    (source_dir / "renamed.pdf").write_bytes(pdf)
    (source_dir / "other.pdf").write_bytes(_make_pdf("Something else."))
    assert rechunk.find_sources(api.db.list_documents(), source_dir) == {doc_id: source_dir / "renamed.pdf"}
    assert rechunk.main(["--source-dir", str(source_dir)]) == 0
