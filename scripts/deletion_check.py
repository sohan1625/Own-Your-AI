"""Deletion test (docs/PRIVACY.md): after a document is removed, its content is
gone from everything the app stores and from answers.

Runs the real app (PostgreSQL + Ollama) in a temporary schema and upload folder.
Needs Docker (container "rag-pgvector", or set PG_CONTAINER) and Ollama running.

    python -m scripts.deletion_check
"""

from __future__ import annotations

import os
import random
import subprocess

import psycopg

from scripts.check_utils import Report, isolated_app, make_pdf

CONTAINER = os.environ.get("PG_CONTAINER", "rag-pgvector")
PG_DATA = "/var/lib/postgresql/data"


def files_on_disk_containing(text: str) -> list[str]:
    """Paths inside the PostgreSQL data directory whose raw bytes contain ``text``."""
    result = subprocess.run(
        ["docker", "exec", CONTAINER, "grep", "-rlaF", text, PG_DATA],
        capture_output=True,
        text=True,
    )
    return [line.replace(PG_DATA + "/", "") for line in result.stdout.splitlines() if line]


def main() -> int:
    number = random.randint(100000, 999999)
    protocol = f"zorblax{number}"  # asked about in the question
    secret = f"velmora{number}"  # only in the document, never in a question
    document_text = (
        f"The {protocol} protocol encrypts every packet with a rotating quartz key. "
        f"Its emergency backup code is {secret}."
    )
    question = f"What does the {protocol} protocol do?"
    report = Report()
    print(f"Unique strings for this run: {protocol} (in the question), {secret} (document only)\n")

    with isolated_app() as (client, api, schema_url):

        def count(query: str, *params) -> int:
            with psycopg.connect(schema_url) as connection:
                return connection.execute(query, params).fetchone()[0]

        # ---------------------------------------------------------------- setup
        response = client.post(
            "/upload",
            files=[
                ("files", ("deletion-test.pdf", make_pdf(document_text), "application/pdf")),
                ("files", ("control.pdf", make_pdf("A router forwards packets between networks."), "application/pdf")),
            ],
        )
        response.raise_for_status()
        test_doc, control_doc = (document["id"] for document in response.json()["documents"])
        upload_path = api._upload_path(test_doc)

        # ---------------------------------------------------------------- before
        print("Before deletion:")
        report.check(upload_path.exists(), "uploaded PDF is stored")
        report.check(
            count("SELECT count(*) FROM chunks WHERE doc_id = %s AND text LIKE %s", test_doc, f"%{secret}%") > 0,
            "chunk text contains the document's content",
        )
        report.check(
            count("SELECT count(*) FROM chunks WHERE doc_id = %s AND embedding IS NOT NULL", test_doc) > 0,
            "chunk embeddings are stored",
        )
        first = client.post("/ask", json={"question": question, "doc_ids": [test_doc, control_doc]}).json()
        report.check(
            any(citation["file"] == "deletion-test.pdf" for citation in first["citations"]),
            f"a query returns the document (answer: {first['answer'][:90]!r})",
        )
        second = client.post("/ask", json={"question": question, "doc_ids": [test_doc, control_doc]}).json()
        report.check(second["cached"] and count("SELECT count(*) FROM answer_cache") > 0, "the answer is cached")
        before_disk = files_on_disk_containing(secret)
        report.note(f"PostgreSQL files containing the document-only string before deletion: {len(before_disk)}")

        # ---------------------------------------------------------------- delete
        report.check(client.delete(f"/documents/{test_doc}").status_code == 200, "DELETE /documents/{id} succeeds")

        # ---------------------------------------------------------------- after
        print("\nAfter deletion:")
        report.check(not upload_path.exists(), "uploaded PDF is deleted")
        report.check(count("SELECT count(*) FROM documents WHERE id = %s", test_doc) == 0, "document row is gone")
        report.check(count("SELECT count(*) FROM chunks WHERE doc_id = %s", test_doc) == 0, "chunks and embeddings are gone")
        report.check(
            count("SELECT count(*) FROM chunks WHERE text LIKE %s OR text LIKE %s", f"%{secret}%", f"%{protocol}%") == 0,
            "no remaining chunk contains the document's content",
        )
        report.check(
            count("SELECT count(*) FROM answer_cache WHERE doc_ids::jsonb ? %s", test_doc) == 0,
            "cached answers that used the document are gone",
        )
        index = api._get_bm25_index()
        report.check(
            not any(secret in chunk["text"] or protocol in chunk["text"] for chunk in index["chunks"]),
            "keyword (BM25) index no longer contains it",
        )
        vector_hits = api._vector_ranking(document_text, [control_doc], 20)
        report.check(
            all(count("SELECT count(*) FROM chunks WHERE id = %s AND doc_id = %s", chunk_id, control_doc) == 1
                for chunk_id, _ in vector_hits),
            "vector search returns only the remaining document",
        )
        report.check(
            client.post("/ask", json={"question": question, "doc_ids": [test_doc]}).status_code == 404,
            "asking about the deleted document is rejected (404)",
        )
        after = client.post("/ask", json={"question": question, "doc_ids": [control_doc]}).json()
        report.check(
            not after["cached"]
            and "quartz" not in after["answer"].lower()
            and secret not in after["answer"]
            and not any(citation["file"] == "deletion-test.pdf" for citation in after["citations"]),
            f"answers no longer use it (answer: {after['answer'][:90]!r})",
        )

        # ---------------------------------------------------------------- what remains by design
        print("\nRetained by design:")
        report.note(
            f"chat history keeps past questions and answers: {count('SELECT count(*) FROM messages WHERE content LIKE %s', f'%{protocol}%')} "
            "messages mention the protocol, "
            f"{count('SELECT count(*) FROM messages WHERE content LIKE %s', f'%{secret}%')} contain the document-only string"
        )

        # ---------------------------------------------------------------- raw files on disk
        print("\nRaw PostgreSQL files (strictest check):")
        after_disk = files_on_disk_containing(secret)
        report.note(f"files still containing the document-only string right after deletion: {after_disk or 'none'}")
        with psycopg.connect(schema_url, autocommit=True) as connection:
            connection.execute("VACUUM FULL chunks")
            connection.execute("VACUUM FULL answer_cache")
            connection.execute("VACUUM FULL documents")
            connection.execute("CHECKPOINT")
        after_vacuum = files_on_disk_containing(secret)
        report.note(f"after VACUUM FULL and CHECKPOINT: {after_vacuum or 'none'}")

    print(f"\n{len([s for s, _ in report.lines if s == 'PASS'])} passed, {report.failed} failed")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
