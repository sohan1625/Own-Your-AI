"""PostgreSQL persistence for uploaded documents and chat history."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATABASE_UNAVAILABLE_MESSAGE = (
    "PostgreSQL is unavailable. Start Docker and the rag-pgvector container "
    "with `docker compose up -d`, then verify DATABASE_URL in .env."
)


def connect() -> psycopg.Connection[dict[str, Any]]:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env, start Docker "
            "and the rag-pgvector container with `docker compose up -d`."
        )
    return psycopg.connect(database_url, row_factory=dict_row)


def initialize_database() -> None:
    try:
        with connect() as connection:
            statements = (
                # pgvector lives in public and is always schema-qualified below, so it
                # resolves under any search_path (the tests use an isolated schema).
                "CREATE EXTENSION IF NOT EXISTS vector SCHEMA public",
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    sha256 TEXT UNIQUE NOT NULL,
                    pages INT NOT NULL,
                    chunk_count INT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    page INT NOT NULL,
                    start_char INT NOT NULL,
                    end_char INT NOT NULL,
                    text TEXT NOT NULL
                )
                """,
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding public.vector",
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citations JSONB NOT NULL DEFAULT '[]'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """,
            )
            for statement in statements:
                connection.execute(statement)
    except psycopg.OperationalError as exc:
        raise RuntimeError(DATABASE_UNAVAILABLE_MESSAGE) from exc


def list_documents() -> list[dict[str, Any]]:
    with connect() as connection:
        return connection.execute(
            """
            SELECT id, filename, sha256, pages, chunk_count, created_at
            FROM documents
            ORDER BY created_at, id
            """
        ).fetchall()


def get_document_by_sha256(sha256: str) -> dict[str, Any] | None:
    with connect() as connection:
        return connection.execute(
            """
            SELECT id, filename, sha256, pages, chunk_count, created_at
            FROM documents
            WHERE sha256 = %s
            """,
            (sha256,),
        ).fetchone()


def _vector_literal(vector: list[float] | None) -> str | None:
    if not vector:
        return None
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


def insert_document(
    document: dict[str, Any],
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]] | None = None,
) -> None:
    """Insert a document and its chunks. ``embeddings`` aligns with ``chunks``;
    when it is missing, chunk embeddings stay NULL and are filled in later."""
    if embeddings is not None and len(embeddings) != len(chunks):
        raise ValueError("embeddings must align with chunks")
    vectors = embeddings if embeddings is not None else [None] * len(chunks)
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO documents (id, filename, sha256, pages, chunk_count)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                document["id"],
                document["filename"],
                document["sha256"],
                document["pages"],
                len(chunks),
            ),
        )
        connection.cursor().executemany(
            """
            INSERT INTO chunks (id, doc_id, page, start_char, end_char, text, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s::public.vector)
            """,
            [
                (
                    chunk["chunk_id"],
                    document["id"],
                    chunk["page"],
                    chunk["start_char"],
                    chunk["end_char"],
                    chunk["text"],
                    _vector_literal(vector),
                )
                for chunk, vector in zip(chunks, vectors)
            ],
        )


def get_chunks_without_embeddings() -> list[dict[str, Any]]:
    with connect() as connection:
        return connection.execute(
            "SELECT id AS chunk_id, text FROM chunks WHERE embedding IS NULL ORDER BY id"
        ).fetchall()


def set_chunk_embeddings(embeddings: dict[str, list[float]]) -> None:
    """Store embeddings keyed by chunk ID."""
    with connect() as connection:
        connection.cursor().executemany(
            "UPDATE chunks SET embedding = %s::public.vector WHERE id = %s",
            [(_vector_literal(vector), chunk_id) for chunk_id, vector in embeddings.items()],
        )


def vector_search(
    query_vector: list[float], doc_ids: list[str], top_n: int
) -> list[tuple[str, float]]:
    """Return ``(chunk_id, cosine similarity)`` for the closest chunks in ``doc_ids``."""
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT id, 1 - (embedding OPERATOR(public.<=>) %(query)s::public.vector) AS score
            FROM chunks
            WHERE doc_id = ANY(%(doc_ids)s) AND embedding IS NOT NULL
            ORDER BY embedding OPERATOR(public.<=>) %(query)s::public.vector, id
            LIMIT %(top_n)s
            """,
            {"query": _vector_literal(query_vector), "doc_ids": doc_ids, "top_n": top_n},
        ).fetchall()
    return [(row["id"], float(row["score"])) for row in rows]


def get_chunks() -> list[dict[str, Any]]:
    with connect() as connection:
        return connection.execute(
            """
            SELECT chunks.id AS chunk_id, chunks.doc_id, documents.filename AS file,
                   chunks.page, chunks.start_char, chunks.end_char, chunks.text
            FROM chunks
            JOIN documents ON documents.id = chunks.doc_id
            ORDER BY documents.id, chunks.page, chunks.start_char, chunks.id
            """
        ).fetchall()


def delete_document(document_id: str) -> dict[str, Any] | None:
    with connect() as connection:
        document = connection.execute(
            """
            DELETE FROM documents
            WHERE id = %s
            RETURNING id, filename, sha256, pages, chunk_count, created_at
            """,
            (document_id,),
        ).fetchone()
        return document


def create_conversation(title: str) -> str:
    from uuid import uuid4

    conversation_id = str(uuid4())
    with connect() as connection:
        connection.execute(
            "INSERT INTO conversations (id, title) VALUES (%s, %s)",
            (conversation_id, title),
        )
    return conversation_id


def conversation_exists(conversation_id: str) -> bool:
    with connect() as connection:
        row = connection.execute(
            "SELECT 1 FROM conversations WHERE id = %s",
            (conversation_id,),
        ).fetchone()
        return row is not None


def save_exchange(
    conversation_id: str,
    question: str,
    answer: str,
    citations: list[dict[str, Any]],
) -> None:
    from uuid import uuid4

    with connect() as connection:
        connection.execute(
            """
            INSERT INTO messages (id, conversation_id, role, content, citations)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (str(uuid4()), conversation_id, "user", question, Jsonb([])),
        )
        connection.execute(
            """
            INSERT INTO messages (id, conversation_id, role, content, citations, created_at)
            VALUES (%s, %s, %s, %s, %s, clock_timestamp() + interval '1 microsecond')
            """,
            (
                str(uuid4()),
                conversation_id,
                "assistant",
                answer,
                Jsonb(citations),
            ),
        )


def list_conversations() -> list[dict[str, Any]]:
    with connect() as connection:
        return connection.execute(
            """
            SELECT id, title, created_at
            FROM conversations
            ORDER BY created_at DESC, id
            """
        ).fetchall()


def get_conversation_messages(conversation_id: str) -> list[dict[str, Any]]:
    with connect() as connection:
        return connection.execute(
            """
            SELECT id, role, content, citations, created_at
            FROM messages
            WHERE conversation_id = %s
            ORDER BY created_at, id
            """,
            (conversation_id,),
        ).fetchall()
