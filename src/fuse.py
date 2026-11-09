"""Hybrid fusion module.

Responsible for:
- Merging ranked candidate lists from keyword (BM25) and vector retrieval using Reciprocal Rank Fusion (RRF).
- Calculating fused scores: score(d) = Σ weight_i / (k + rank_i(d)) with configurable k and weights.
- Providing unified ranked candidates for downstream threshold verification and answer generation.
"""

