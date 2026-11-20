"""Hybrid fusion module.

Merges ranked candidate lists from keyword (BM25) and vector retrieval using
Reciprocal Rank Fusion (RRF): score(d) = sum over lists of 1 / (k + rank_i(d)),
with k read from ``[fusion] rrf_k`` in config.toml.
"""

from __future__ import annotations

import tomllib
from pathlib import Path


def get_rrf_k(config_path: str | Path) -> int:
    with open(config_path, "rb") as config_file:
        config = tomllib.load(config_file)
    k = int(config.get("fusion", {}).get("rrf_k", 60))
    if k <= 0:
        raise ValueError("rrf_k must be a positive integer")
    return k


def reciprocal_rank_fusion(ranked_ids: list[list[str]], k: int = 60) -> list[str]:
    """Fuse ranked lists of IDs (best first) into one list, best first.

    Ties keep first-seen order, so the earlier list wins an exact tie.
    """
    scores: dict[str, float] = {}
    for ranking in ranked_ids:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda item_id: scores[item_id], reverse=True)
