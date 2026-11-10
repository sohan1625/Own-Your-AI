# src/index.py
"""Indexing utilities: Keyword (BM25) and Vector (nomic-embed-text) embeddings.

This module provides:
1. Pure-Python BM25 keyword index over chunk records.
2. Vector embeddings for chunks using local Ollama endpoint (/api/embed with nomic-embed-text).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import tomllib
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple
import urllib.request


# ---------------------------------------------------------------------------
# Configuration & Host Validation helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path = "config.toml") -> dict[str, Any]:
    """Load the TOML configuration file and return the full dictionary."""
    cfg_file = Path(config_path)
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(cfg_file, "rb") as f:
        return tomllib.load(f)


def _validate_host(host: str) -> str:
    """Ensure the Ollama host points to localhost.

    Acceptable forms are ``http://localhost`` (with optional port) or ``http://127.0.0.1``.
    Returns the host string without trailing slash.
    """
    host = host.rstrip("/")
    if host.startswith("http://localhost") or host.startswith("http://127.0.0.1"):
        return host
    raise ValueError("Ollama host must be localhost (127.0.0.1); outbound network calls are prohibited.")


def apply_prefix(prefix: str, text: str) -> str:
    """Apply prefix ensuring nomic-embed-text expected format (e.g. search_query: text)."""
    if not prefix:
        return text
    if prefix.endswith(": "):
        return f"{prefix}{text}"
    if prefix.endswith(":"):
        return f"{prefix} {text}"
    if prefix.endswith(" "):
        return f"{prefix}{text}"
    return f"{prefix}: {text}"


def get_embedding_prefix(kind: str = "document", config_path: str | Path = "config.toml") -> str:
    """Read the prefix for embedding from config.toml."""
    try:
        cfg = load_config(config_path)
        emb_cfg = cfg.get("embedding", {})
        if kind in ("query", "search_query", "question"):
            return emb_cfg.get(
                "search_query_prefix",
                emb_cfg.get("query_prefix", "search_query: "),
            )
        else:
            return emb_cfg.get(
                "search_document_prefix",
                emb_cfg.get("document_prefix", "search_document: "),
            )
    except Exception:
        return "search_query: " if kind in ("query", "search_query", "question") else "search_document: "


# ---------------------------------------------------------------------------
# Vector embedding utilities (local Ollama /api/embed)
# ---------------------------------------------------------------------------

def embed_text(
    text: str,
    *,
    prefix: str = "",
    config_path: str | Path = "config.toml",
    host: str | None = None,
    model: str | None = None,
) -> List[float]:
    """Call local Ollama /api/embed to compute an embedding vector for text.

    Ensures the host is localhost (127.0.0.1); raises ValueError otherwise.
    Applies the prefix if given.
    """
    try:
        cfg = load_config(config_path)
    except Exception:
        cfg = {}

    emb_cfg = cfg.get("embedding", {})
    ollama_cfg = cfg.get("ollama", {})

    target_model = model or emb_cfg.get("model") or "nomic-embed-text"
    target_host = host or emb_cfg.get("host") or ollama_cfg.get("host") or "http://localhost:11434"

    valid_host = _validate_host(target_host)

    prefixed_text = apply_prefix(prefix, text)
    payload = json.dumps({"model": target_model, "input": prefixed_text}).encode("utf-8")
    req = urllib.request.Request(f"{valid_host}/api/embed", data=payload, method="POST")
    req.add_header("Content-Type", "application/json")

    with urllib.request.urlopen(req) as resp:
        resp_data = resp.read()
        resp_json = json.loads(resp_data)

        # Ollama /api/embed returns "embeddings": [[...]]
        embeddings = resp_json.get("embeddings")
        if embeddings and len(embeddings) > 0:
            return embeddings[0]
        # Backwards compatibility if Ollama returns "embedding": [...]
        if "embedding" in resp_json:
            return resp_json["embedding"]
        return []


def build_chunk_embeddings(
    chunks: List[Dict[str, Any]],
    *,
    config_path: str | Path = "config.toml",
    host: str | None = None,
    model: str | None = None,
) -> List[List[float]]:
    """Build embedding vectors for a list of chunks using Ollama /api/embed.

    Reads model name, host, and document prefix from config.toml.
    Returns vectors in memory as a list aligned with the input chunk list.
    """
    if not chunks:
        return []
    doc_prefix = get_embedding_prefix("document", config_path=config_path)
    vectors: List[List[float]] = []
    for chunk in chunks:
        text = chunk.get("text", "")
        vec = embed_text(
            text,
            prefix=doc_prefix,
            config_path=config_path,
            host=host,
            model=model,
        )
        vectors.append(vec)
    return vectors


# Alias
build_vector_index = build_chunk_embeddings

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Simple whitespace‑based tokenizer.

    The RAG pipeline uses a word‑based token approximation, so splitting on
    whitespace is sufficient for the BM25 stage.
    """
    # Lower‑case and split on any non‑word characters to get clean tokens.
    return re.findall(r"\b\w+\b", text.lower())

# ---------------------------------------------------------------------------
# BM25 index implementation
# ---------------------------------------------------------------------------

def build_bm25_index(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a BM25 index from a list of chunk records.

    Returns a dictionary containing the necessary statistics:
        * ``doc_len`` – list of document (chunk) lengths (in tokens)
        * ``avgdl`` – average document length
        * ``term_freqs`` – list of ``Counter`` objects for each chunk
        * ``doc_freq`` – ``Counter`` of document frequencies per term
        * ``chunks`` – the original chunk list (preserved for lookup)
    """
    doc_len: List[int] = []
    term_freqs: List[Counter[str]] = []
    doc_freq: Counter[str] = Counter()

    for chunk in chunks:
        tokens = _tokenize(chunk.get("text", ""))
        tf = Counter(tokens)
        term_freqs.append(tf)
        doc_len.append(len(tokens))
        # Update document frequency (unique terms per document)
        for term in tf.keys():
            doc_freq[term] += 1

    avgdl = sum(doc_len) / len(doc_len) if doc_len else 0.0

    return {
        "doc_len": doc_len,
        "avgdl": avgdl,
        "term_freqs": term_freqs,
        "doc_freq": doc_freq,
        "chunks": chunks,
        "N": len(chunks),
    }


def _idf(term: str, index: Dict[str, Any], k1: float = 1.5, b: float = 0.75) -> float:
    """Compute the inverse document frequency for a term.

    Uses the BM25‑style smoothing:
        idf = log( (N - df + 0.5) / (df + 0.5) + 1 )
    """
    df = index["doc_freq"].get(term, 0)
    N = index["N"]
    return math.log((N - df + 0.5) / (df + 0.5) + 1.0)


def _score(query_terms: List[str], index: Dict[str, Any], doc_idx: int, k1: float = 1.5, b: float = 0.75) -> float:
    """Score a single document (chunk) against the query terms.
    """
    tf = index["term_freqs"][doc_idx]
    doc_len = index["doc_len"][doc_idx]
    avgdl = index["avgdl"]
    score = 0.0
    for term in query_terms:
        if term not in tf:
            continue
        idf_val = _idf(term, index, k1, b)
        freq = tf[term]
        numerator = freq * (k1 + 1)
        denominator = freq + k1 * (1 - b + b * doc_len / avgdl)
        score += idf_val * (numerator / denominator)
    return score


def bm25_search(query: str, index: Dict[str, Any], top_n: int = 20) -> List[Tuple[Dict[str, Any], float]]:
    """Return the top‑N chunks matching *query* sorted by BM25 score.

    The return value is a list of ``(chunk, score)`` tuples.
    """
    query_terms = _tokenize(query)
    if not query_terms:
        return []
    scores: List[Tuple[int, float]] = []  # (doc_idx, score)
    for doc_idx in range(index["N"]):
        s = _score(query_terms, index, doc_idx)
        if s > 0:
            scores.append((doc_idx, s))
    # Sort by score descending and keep top_n
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:top_n]
    return [(index["chunks"][doc_idx], score) for doc_idx, score in top]

# ---------------------------------------------------------------------------
# Convenience wrapper used by higher‑level code (future stages)
# ---------------------------------------------------------------------------

def build_and_search(chunks: List[Dict[str, Any]], query: str, top_n: int = 20) -> List[Dict[str, Any]]:
    """One‑shot helper: build an index from *chunks* and return the top matches.
    """
    idx = build_bm25_index(chunks)
    results = bm25_search(query, idx, top_n=top_n)
    # Return only the chunk dictionaries (scores can be inspected separately if needed)
    return [chunk for chunk, _ in results]

# End of file
