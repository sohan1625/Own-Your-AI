"""Retrieval module that offers keyword-only BM25 search for the RAG pipeline."""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Mapping, List, Dict

# Re‑use the BM25 utilities from src.index
from src.index import bm25_search

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

# End of retrieve module
