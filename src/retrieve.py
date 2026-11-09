"""Retrieval module.

Responsible for:
- Executing custom BM25 keyword scoring over chunk term statistics.
- Executing dense vector similarity search (cosine distance) using chunk embeddings via pgvector.
- Returning top-N candidate chunk lists from both keyword and vector retrievers for every query.
"""

