# src/index.py
"""Keyword (BM25) indexing utilities.

This module provides a lightweight pure‑Python implementation of BM25 suitable for the
M1 milestone. It deliberately avoids external dependencies to keep the project offline‑
first. The implementation follows the classic BM25 formula (Robertson & Jones, 2004)
with the standard parameters k1=1.5 and b=0.75.

The public API consists of:
    * ``build_bm25_index(chunks)`` – builds an index from a list of chunk records.
    * ``bm25_search(query, index, top_n=20)`` – returns the top‑N matching chunk
      dictionaries sorted by BM25 score (descending).

Chunk records are dictionaries with at least the keys ``file``, ``page``, ``text`` and
``parent_id`` (as produced by ``src.chunk``). The function does **not** modify the
input records.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple

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
