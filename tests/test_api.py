from pathlib import Path

from fastapi.testclient import TestClient

from src import api


def test_upload_bundled_pdf_survives_cache_clear_without_readable_text_warning(
    monkeypatch, tmp_path
):
    pdf_path = Path(__file__).resolve().parents[1] / "data" / "Core_Java_Complete_Answers.pdf"
    monkeypatch.setattr(api, "DOCUMENT_INDEX_DIR", tmp_path)
    monkeypatch.setattr(
        api,
        "build_chunk_embeddings",
        lambda chunks, **kwargs: [[1.0] for _ in chunks],
    )
    monkeypatch.setattr(
        api,
        "vector_retrieve",
        lambda question, chunks, vectors, top_n, **kwargs: [(chunks[0], 0.9)],
    )
    monkeypatch.setattr(
        api,
        "generate_answer",
        lambda question, chunks, **kwargs: f"JVM runs bytecode [{chunks[0]['chunk_id']}]",
    )

    api.documents.clear()
    with TestClient(api.app) as client:
        response = client.post(
            "/upload",
            files={"file": (pdf_path.name, pdf_path.read_bytes(), "application/pdf")},
        )

    assert response.status_code == 200
    result = response.json()
    assert result["pages"] > 0
    assert "detail" not in result
    assert "The PDF contains no readable text." not in response.text
    assert (tmp_path / f"{result['doc_id']}.json").is_file()

    api.documents.clear()
    with TestClient(api.app) as client:
        ask_response = client.post(
            "/ask",
            json={"doc_id": result["doc_id"], "question": "What is the JVM?"},
        )

    assert ask_response.status_code == 200
    assert "JVM runs bytecode" in ask_response.json()["answer"]

    missing_response = client.post(
        "/ask",
        json={"doc_id": "00000000-0000-0000-0000-000000000000", "question": "What is the JVM?"},
    )
    assert missing_response.status_code == 404
    assert missing_response.json()["detail"] == "Document not found. Re-upload the PDF."
    api.documents.clear()
