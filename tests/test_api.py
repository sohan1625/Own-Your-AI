import importlib
import os
import uuid
from pathlib import Path

import psycopg
import pytest
from dotenv import dotenv_values
from fastapi.testclient import TestClient
from psycopg import sql
from psycopg.conninfo import make_conninfo

from src import api

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def isolated_database(monkeypatch):
    env_values = dotenv_values(ROOT / ".env")
    example_values = dotenv_values(ROOT / ".env.example")
    base_url = (
        os.environ.get("DATABASE_URL")
        or env_values.get("DATABASE_URL")
        or example_values.get("DATABASE_URL")
    )
    if not base_url:
        pytest.fail("Set DATABASE_URL in .env or .env.example to run PostgreSQL API tests.")

    schema = f"test_{uuid.uuid4().hex}"
    with psycopg.connect(base_url, autocommit=True) as connection:
        connection.execute(
            sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema))
        )

    test_url = make_conninfo(base_url, options=f"-c search_path={schema}")
    monkeypatch.setenv("DATABASE_URL", test_url)
    try:
        yield test_url
    finally:
        with psycopg.connect(base_url, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema))
            )


def _make_pdf(text: str) -> bytes:
    escaped_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 14 Tf 50 740 Td ({escaped_text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n"
        + stream
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_number} 0 obj\n".encode("ascii"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(pdf)


def test_upload_ask_delete_and_restart_persisted_state(
    isolated_database, monkeypatch, tmp_path
):
    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(api, "UPLOAD_DIR", upload_dir)

    def answer_with_citations(question, chunks, **kwargs):
        filenames = {chunk["file"] for chunk in chunks}
        assert filenames == {"alpha.pdf", "beta.pdf"}
        first_id, second_id = (chunk["chunk_id"] for chunk in chunks)
        return f"The shared fact is present in both files [{first_id}] [{second_id}]"

    monkeypatch.setattr(api, "generate_answer", answer_with_citations)
    pdfs = [
        ("files", ("alpha.pdf", _make_pdf("The shared fact appears in alpha."), "application/pdf")),
        ("files", ("beta.pdf", _make_pdf("The shared fact appears in beta."), "application/pdf")),
    ]

    with TestClient(api.app) as client:
        upload_response = client.post("/upload", files=pdfs)
        assert upload_response.status_code == 200
        uploaded = upload_response.json()["documents"]
        assert len(uploaded) == 2
        assert {document["filename"] for document in uploaded} == {"alpha.pdf", "beta.pdf"}
        assert all(not document["already_indexed"] for document in uploaded)
        assert {path.stem for path in upload_dir.glob("*.pdf")} == {
            document["id"] for document in uploaded
        }

        ask_response = client.post(
            "/ask",
            json={
                "question": "What is the shared fact?",
                "doc_ids": [document["id"] for document in uploaded],
            },
        )
        assert ask_response.status_code == 200
        answer = ask_response.json()
        assert {citation["file"] for citation in answer["citations"]} == {
            "alpha.pdf",
            "beta.pdf",
        }

        conversation_id = answer["conversation_id"]
        messages = client.get(
            f"/conversations/{conversation_id}/messages"
        ).json()
        assert [message["role"] for message in messages] == ["user", "assistant"]
        assert messages[0]["content"] == "What is the shared fact?"

        removed_id = uploaded[0]["id"]
        delete_response = client.delete(f"/documents/{removed_id}")
        assert delete_response.status_code == 200
        assert not (upload_dir / f"{removed_id}.pdf").exists()

    with psycopg.connect(isolated_database) as connection:
        chunk_count = connection.execute(
            "SELECT count(*) FROM chunks WHERE doc_id = %s",
            (uploaded[0]["id"],),
        ).fetchone()[0]
    assert chunk_count == 0

    restarted_api = importlib.reload(api)
    with TestClient(restarted_api.app) as client:
        documents_response = client.get("/documents")
        assert documents_response.status_code == 200
        remaining_documents = documents_response.json()
        assert len(remaining_documents) == 1
        assert remaining_documents[0]["id"] == uploaded[1]["id"]

        conversations_response = client.get("/conversations")
        assert conversations_response.status_code == 200
        assert conversations_response.json()[0]["id"] == conversation_id
        messages_response = client.get(
            f"/conversations/{conversation_id}/messages"
        )
        assert messages_response.status_code == 200
        assert [message["content"] for message in messages_response.json()] == [
            "What is the shared fact?",
            answer["answer"],
        ]


def test_startup_error_explains_how_to_start_postgres(monkeypatch):
    def database_is_unavailable():
        raise psycopg.OperationalError("database is down")

    monkeypatch.setattr(api.db, "connect", database_is_unavailable)
    with pytest.raises(RuntimeError, match="Start Docker and the rag-pgvector container"), TestClient(api.app):
        pass


def test_followup_retrieves_and_answers_with_recent_conversation(monkeypatch):
    history = [
        {"role": "user", "content": "What are the different types of delays?"},
        {
            "role": "assistant",
            "content": (
                "Transmission Delay, Propagation Delay, Queuing Delay, "
                "and Processing Delay."
            ),
        },
    ]
    chunk = {
        "chunk_id": "delay-definitions",
        "doc_id": "doc-1",
        "text": "Transmission delay is the time to put bits on a link.",
    }
    captured = {}

    monkeypatch.setattr(api.db, "list_documents", lambda: [{"id": "doc-1"}])
    monkeypatch.setattr(api.db, "conversation_exists", lambda _: True)
    monkeypatch.setattr(api.db, "get_conversation_messages", lambda _: history)
    monkeypatch.setattr(api.db, "save_exchange", lambda *args: None)
    monkeypatch.setattr(api, "_get_bm25_index", lambda: {"chunks": [chunk]})
    monkeypatch.setattr(api, "get_retrieval_params", lambda _: {"top_n": 5})
    monkeypatch.setattr(api, "get_generation_top_k", lambda _: 1)

    def search(query, index, top_n):
        captured["retrieval_query"] = query
        return [(chunk, 1.0)]

    def generate(question, chunks, **kwargs):
        captured["question"] = question
        captured["history"] = kwargs["conversation_history"]
        return "Transmission delay is the time to put bits on a link. [delay-definitions]"

    monkeypatch.setattr(api, "bm25_search", search)
    monkeypatch.setattr(api, "generate_answer", generate)
    monkeypatch.setattr(
        api,
        "resolve_citations",
        lambda *_: type("CitationResult", (), {"valid_citations": []})(),
    )

    response = api.ask_question(
        api.AskRequest(
            question="Explain those.",
            doc_ids=["doc-1"],
            conversation_id="conversation-1",
        )
    )

    assert response["answer"].startswith("Transmission delay")
    # Previous question expands the search; the previous answer does not.
    assert captured["retrieval_query"] == "What are the different types of delays?\nExplain those."
    assert captured["question"] == "Explain those."
    assert captured["history"] == history


def test_followup_search_is_not_buried_by_previous_answer(monkeypatch):
    firewall = {
        "chunk_id": "firewall",
        "doc_id": "doc-1",
        "text": "Explain Firewalls? A firewall is a network security device that "
        "monitors incoming and outgoing traffic and accepts, rejects or drops it.",
    }
    osi = {
        "chunk_id": "osi",
        "doc_id": "doc-1",
        "text": "The OSI model has seven layers. The TCP/IP model is a simplified "
        "version of the OSI model with four layers.",
    }
    # Distractors share many words with the previous answer, but not with either question.
    distractors = [
        {
            "chunk_id": f"tcp-ip-{number}",
            "doc_id": "doc-1",
            "text": "TCP/IP layers map onto each other; the simplified TCP/IP version "
            "is discussed, though not explicitly mentioned, with TCP/IP layers.",
        }
        for number in range(3)
    ]
    history = [
        {"role": "user", "content": "what is osi model"},
        {
            "role": "assistant",
            "content": "The OSI model is not explicitly mentioned, but the TCP/IP model, "
            "a simplified version of the OSI model, is discussed. The OSI model "
            "layers and the TCP/IP model layers map onto each other.",
        },
    ]
    captured = {}

    monkeypatch.setattr(api.db, "list_documents", lambda: [{"id": "doc-1"}])
    monkeypatch.setattr(api.db, "conversation_exists", lambda _: True)
    monkeypatch.setattr(api.db, "get_conversation_messages", lambda _: history)
    monkeypatch.setattr(api.db, "save_exchange", lambda *args: None)
    index = api.build_bm25_index([firewall, osi, *distractors])
    monkeypatch.setattr(api, "_get_bm25_index", lambda: index)
    monkeypatch.setattr(api, "get_retrieval_params", lambda _: {"top_n": 2})
    monkeypatch.setattr(api, "get_generation_top_k", lambda _: 2)

    def generate(question, chunks, **kwargs):
        captured["chunk_ids"] = [chunk["chunk_id"] for chunk in chunks]
        return "A firewall monitors traffic. [firewall]"

    monkeypatch.setattr(api, "generate_answer", generate)
    monkeypatch.setattr(
        api,
        "resolve_citations",
        lambda *_: type("CitationResult", (), {"valid_citations": []})(),
    )

    api.ask_question(
        api.AskRequest(
            question="explain firewalls",
            doc_ids=["doc-1"],
            conversation_id="conversation-1",
        )
    )

    assert "firewall" in captured["chunk_ids"]
