"""PostgreSQL-backed cache of grounded answers keyed by question and document set."""

from __future__ import annotations

import json
import tomllib
from hashlib import sha256
from pathlib import Path
from typing import Any

from psycopg.types.json import Jsonb

from src import db

REFUSAL = "not found in your documents"


def is_enabled(config_path: str | Path) -> bool:
    with open(config_path, "rb") as config_file:
        config = tomllib.load(config_file)
    return bool(config.get("cache", {}).get("enabled", False))


def initialize_cache() -> None:
    with db.connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS answer_cache (
                key TEXT PRIMARY KEY,
                question TEXT,
                doc_ids TEXT,
                doc_hashes TEXT,
                answer TEXT,
                citations JSONB,
                created_at TIMESTAMPTZ DEFAULT now()
            )
            """
        )


def normalize_question(question: str) -> str:
    return " ".join(question.lower().split())


def make_key(question: str, doc_hashes: dict[str, str]) -> str:
    """Hash the normalized question, sorted doc IDs, and each document's sha256.

    ``doc_hashes`` maps each selected document ID to that document's sha256.
    """
    doc_ids = sorted(doc_hashes)
    payload = json.dumps(
        {
            "question": normalize_question(question),
            "doc_ids": doc_ids,
            "doc_hashes": [doc_hashes[doc_id] for doc_id in doc_ids],
        },
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def is_refusal(answer: str) -> bool:
    return answer.strip().rstrip(".").strip().lower() == REFUSAL


def get(key: str) -> dict[str, Any] | None:
    with db.connect() as connection:
        return connection.execute(
            "SELECT answer, citations FROM answer_cache WHERE key = %s",
            (key,),
        ).fetchone()


def store(
    key: str,
    question: str,
    doc_hashes: dict[str, str],
    answer: str,
    citations: list[dict[str, Any]],
) -> bool:
    """Store an answer unless it is a refusal. Returns True when stored."""
    if is_refusal(answer):
        return False
    doc_ids = sorted(doc_hashes)
    with db.connect() as connection:
        connection.execute(
            """
            INSERT INTO answer_cache (key, question, doc_ids, doc_hashes, answer, citations)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (key) DO UPDATE
            SET answer = EXCLUDED.answer,
                citations = EXCLUDED.citations,
                created_at = now()
            """,
            (
                key,
                question,
                # JSON arrays, so entries referencing a document can be found exactly
                # with the jsonb ? operator (no substring false matches).
                json.dumps(doc_ids),
                json.dumps([doc_hashes[doc_id] for doc_id in doc_ids]),
                answer,
                Jsonb(citations),
            ),
        )
    return True


def delete_for_document(document_id: str) -> int:
    """Delete every cached answer whose document selection includes ``document_id``."""
    with db.connect() as connection:
        return connection.execute(
            "DELETE FROM answer_cache WHERE doc_ids::jsonb ? %s",
            (document_id,),
        ).rowcount
