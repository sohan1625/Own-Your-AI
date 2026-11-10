"""Retrieval module that offers keyword (BM25) and vector (cosine similarity) search."""
from __future__ import annotations

import math
from pathlib import Path
import tomllib
from typing import Any, Mapping, List, Dict, Tuple

# Re‑use indexing and embedding utilities from src.index
from src.index import bm25_search, embed_text, get_embedding_prefix

def load_config(config_path: str | Path = "config.toml") -> dict[str, Any]:
    """Load the TOML configuration file and return the full dictionary.

    Raises ``FileNotFoundError`` if the file does not exist.
    """
    cfg_file = Path(config_path)
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(cfg_file, "rb") as f:
        return tomllib.load(f)

def get_retrieval_params(config_path: str | Path = "config.toml") -> Mapping[str, int]:
    """Extract the ``top_n_per_retriever`` value from the ``[retrieval]`` section.

    Returns a mapping with the key ``top_n``.
    """
    cfg = load_config(config_path)
    retrieval_cfg = cfg.get("retrieval", {})
    top_n = int(retrieval_cfg.get("top_n_per_retriever", 20))
    if top_n <= 0:
        raise ValueError("top_n_per_retriever must be a positive integer")
    return {"top_n": top_n}

def keyword_retrieve(question: str, bm25_index: dict[str, Any], *, config_path: str | Path = "config.toml") -> List[Dict[str, Any]] | str:
    """Retrieve the top‑N chunks for *question* using a BM25 index.

    * If the BM25 search returns no matching chunks, the function returns the
      placeholder string ``"not found"`` – the real threshold is set later on the
      eval set (RAG_PIPELINE section 7).
    * The number of candidates ``N`` is taken from ``config.toml`` under the
      ``[retrieval]`` section (default 20).
    """
    params = get_retrieval_params(config_path)
    top_n = params["top_n"]

    # bm25_search returns a list of (chunk, score) tuples.
    results = bm25_search(question, bm25_index, top_n=top_n)
    if not results:
        return "not found"
    # Return only the chunk dictionaries, preserving order.
    return [chunk for chunk, _ in results]


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two float vectors using only standard library math."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def vector_retrieve(
    question: str,
    chunks: List[Dict[str, Any]],
    vectors: List[List[float]],
    top_n: int | None = None,
    *,
    config_path: str | Path = "config.toml",
) -> List[Tuple[Dict[str, Any], float]]:
    """Retrieve top_n chunks by cosine similarity with the embedded question.

    1. Embeds question using local Ollama nomic-embed-text with the search_query prefix.
    2. Computes cosine similarity between question vector and chunk vectors.
    3. Returns top_n (chunk, score) tuples sorted descending by score.
    Uses standard library only, with no numpy.
    """
    if not chunks or not vectors:
        return []
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors must have the same length.")

    if top_n is None:
        try:
            params = get_retrieval_params(config_path)
            top_n = params.get("top_n", 20)
        except Exception:
            top_n = 20

    query_prefix = get_embedding_prefix("query", config_path=config_path)
    q_vec = embed_text(question, prefix=query_prefix, config_path=config_path)
    if not q_vec:
        return []

    scored: List[Tuple[Dict[str, Any], float]] = []
    for chunk, vec in zip(chunks, vectors):
        sim = cosine_similarity(q_vec, vec)
        scored.append((chunk, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


# End of retrieve module
