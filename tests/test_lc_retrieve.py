"""LangChain retrieval engine: BM25Retriever + pgvector wrapper fused by EnsembleRetriever."""

import pytest

from src import fuse
from src import lc_retrieve
from tests.conftest import fake_embedding
from tests.test_api import isolated_database  # noqa: F401 (fixture)

CHUNKS = [
    {"chunk_id": "a", "doc_id": "d1", "file": "x.pdf", "page": 1, "start_char": 0, "end_char": 10,
     "text": "Oak was renamed to Java because of a trademark."},
    {"chunk_id": "b", "doc_id": "d1", "file": "x.pdf", "page": 2, "start_char": 0, "end_char": 10,
     "text": "The JVM runs bytecode on every platform."},
    {"chunk_id": "c", "doc_id": "d1", "file": "x.pdf", "page": 3, "start_char": 0, "end_char": 10,
     "text": "PreparedStatement prevents SQL injection."},
]


@pytest.fixture
def fake_vectors(monkeypatch):
    calls = {}

    def vector_search(query_vector, doc_ids, top_n):
        calls["doc_ids"] = doc_ids
        calls["top_n"] = top_n
        # Fixed ranking: c, b, a.
        return [("c", 0.9), ("b", 0.5), ("a", 0.1)][:top_n]

    monkeypatch.setattr(lc_retrieve, "embed_text", lambda text, **kw: fake_embedding(text))
    monkeypatch.setattr(lc_retrieve.db, "vector_search", vector_search)
    return calls


def test_pgvector_retriever_wraps_db_vector_search(fake_vectors):
    retriever = lc_retrieve.PgVectorRetriever(
        chunks_by_id={c["chunk_id"]: c for c in CHUNKS}, doc_ids=["d1"], k=2
    )
    docs = retriever.invoke("anything")
    assert [d.metadata["chunk_id"] for d in docs] == ["c", "b"]
    assert fake_vectors == {"doc_ids": ["d1"], "top_n": 2}
    assert retriever._last_top_score == 0.9


def test_hybrid_returns_chunks_and_top_vector_score(fake_vectors):
    engine = lc_retrieve.LangChainHybrid(CHUNKS, ["d1"], top_n=3, rrf_k=60)
    chunks, top_score = engine.retrieve("Why was Oak renamed to Java?")
    assert top_score == 0.9
    assert {c["chunk_id"] for c in chunks} == {"a", "b", "c"}
    assert all(c in CHUNKS for c in chunks)


def test_equal_weight_ensemble_orders_like_our_rrf(fake_vectors):
    engine = lc_retrieve.LangChainHybrid(CHUNKS, ["d1"], top_n=3, rrf_k=60)
    query = "Why was Oak renamed to Java?"
    bm25_ids = [d.metadata["chunk_id"] for d in engine.bm25.invoke(query)]
    vector_ids = [d.metadata["chunk_id"] for d in engine.vector.invoke(query)]
    chunks, _ = engine.retrieve(query)
    assert [c["chunk_id"] for c in chunks] == fuse.reciprocal_rank_fusion([bm25_ids, vector_ids], k=60)


def test_bm25_retriever_uses_project_tokenizer(fake_vectors):
    engine = lc_retrieve.LangChainHybrid(CHUNKS, ["d1"], top_n=1)
    # Lower-case query still matches the capitalised "PreparedStatement".
    assert engine.bm25.invoke("preparedstatement injection")[0].metadata["chunk_id"] == "c"


def test_ask_uses_langchain_engine_when_configured(monkeypatch, tmp_path, isolated_database):
    from fastapi.testclient import TestClient

    from src import api
    from tests.test_api import _make_pdf

    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(api, "_retrieval_engine", lambda: "langchain")
    monkeypatch.setattr(lc_retrieve, "embed_text", lambda text, **kw: fake_embedding(text))
    used = {}
    real_retrieve = lc_retrieve.LangChainHybrid.retrieve

    def spy(self, query):
        used["query"] = query
        return real_retrieve(self, query)

    monkeypatch.setattr(lc_retrieve.LangChainHybrid, "retrieve", spy)
    monkeypatch.setattr(
        api, "generate_answer", lambda q, chunks, **kw: f"Stated here [{chunks[0]['chunk_id']}]"
    )
    with TestClient(api.app) as client:
        upload = client.post(
            "/upload",
            files=[("files", ("a.pdf", _make_pdf("Oak was renamed to Java."), "application/pdf"))],
        )
        assert upload.status_code == 200
        response = client.post("/ask", json={"question": "Why was Oak renamed?"})
    assert response.status_code == 200
    assert used == {"query": "Why was Oak renamed?"}
    assert response.json()["citations"][0]["file"] == "a.pdf"


def test_project_config_uses_langchain_engine():
    from src import api

    assert api._retrieval_engine() == "langchain"


def test_query_embedder_is_used_when_given(monkeypatch):
    seen = []
    monkeypatch.setattr(lc_retrieve.db, "vector_search", lambda v, d, n: [("a", 0.7)])
    monkeypatch.setattr(lc_retrieve, "embed_text", lambda *a, **k: pytest.fail("default embedder used"))
    engine = lc_retrieve.LangChainHybrid(
        CHUNKS, ["d1"], top_n=3, embed_query=lambda q: seen.append(q) or fake_embedding(q)
    )
    _, top = engine.retrieve("Oak")
    assert seen == ["Oak"] and top == 0.7
