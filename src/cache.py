"""PostgreSQL-backed cache of grounded answers keyed by question and document set.

Two lookups, in order:

1. Exact: the normalized question plus the selected documents' IDs and sha256 hashes.
2. Semantic (``[cache] semantic = true``), only when the exact lookup misses: the
   question's embedding is compared (cosine) with the embeddings of cached questions
   for the *same* document set (same IDs and hashes), and the closest cached answer is
   served only if its similarity is at or above ``[cache] semantic_threshold``.

Refusals are never stored, follow-ups with conversation history never use the cache
(enforced in ``src.api``), and deleting a document deletes every entry that used it.
"""

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


def semantic_settings(config_path: str | Path) -> tuple[bool, float]:
    """``(enabled, threshold)`` for the semantic lookup."""
    with open(config_path, "rb") as config_file:
        cache_cfg = tomllib.load(config_file).get("cache", {})
    threshold = float(cache_cfg.get("semantic_threshold", 1.0))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("[cache] semantic_threshold must be between 0 and 1")
    return bool(cache_cfg.get("semantic", False)), threshold


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
        # Semantic lookup: entries are compared only within one document set.
        connection.execute("ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS doc_set_key TEXT")
        connection.execute(
            "ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS question_embedding public.vector"
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


def make_doc_set_key(doc_hashes: dict[str, str]) -> str:
    """Hash of the sorted document IDs and their sha256 hashes (no question)."""
    doc_ids = sorted(doc_hashes)
    payload = json.dumps(
        {"doc_ids": doc_ids, "doc_hashes": [doc_hashes[doc_id] for doc_id in doc_ids]},
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


def is_refusal(answer: str) -> bool:
    return answer.strip().rstrip(".").strip().lower() == REFUSAL


def get(key: str) -> dict[str, Any] | None:
    with db.connect() as connection:
        return connection.execute(
            "SELECT answer, citations FROM answer_cache WHERE key = %s",
            (key,),
        ).fetchone()


def get_semantic(
    question_embedding: list[float],
    doc_hashes: dict[str, str],
    threshold: float,
) -> dict[str, Any] | None:
    """Closest cached answer for the same document set, if similarity >= threshold.

    Returns ``answer``, ``citations``, ``question`` (the cached one) and ``similarity``.
    """
    if not question_embedding:
        return None
    with db.connect() as connection:
        row = connection.execute(
            """
            SELECT answer, citations, question,
                   1 - (question_embedding OPERATOR(public.<=>) %(q)s::public.vector) AS similarity
            FROM answer_cache
            WHERE doc_set_key = %(doc_set)s AND question_embedding IS NOT NULL
            ORDER BY question_embedding OPERATOR(public.<=>) %(q)s::public.vector, key
            LIMIT 1
            """,
            {"q": _vector_literal(question_embedding), "doc_set": make_doc_set_key(doc_hashes)},
        ).fetchone()
    if row is None or float(row["similarity"]) < threshold:
        return None
    return {**row, "similarity": float(row["similarity"])}


def store(
    key: str,
    question: str,
    doc_hashes: dict[str, str],
    answer: str,
    citations: list[dict[str, Any]],
    question_embedding: list[float] | None = None,
) -> bool:
    """Store an answer unless it is a refusal. Returns True when stored."""
    if is_refusal(answer):
        return False
    doc_ids = sorted(doc_hashes)
    with db.connect() as connection:
        connection.execute(
            """
            INSERT INTO answer_cache
                (key, question, doc_ids, doc_hashes, answer, citations, doc_set_key, question_embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::public.vector)
            ON CONFLICT (key) DO UPDATE
            SET answer = EXCLUDED.answer,
                citations = EXCLUDED.citations,
                doc_set_key = EXCLUDED.doc_set_key,
                question_embedding = COALESCE(EXCLUDED.question_embedding, answer_cache.question_embedding),
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
                make_doc_set_key(doc_hashes),
                _vector_literal(question_embedding) if question_embedding else None,
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
