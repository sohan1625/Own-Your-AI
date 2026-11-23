"""LangChain retrieval engine ([retrieval] engine = "langchain").

BM25 (``langchain_community`` ``BM25Retriever``, backed by ``rank_bm25``) and
pgvector (a retriever that wraps ``db.vector_search``) are combined with
``EnsembleRetriever``, which fuses the two lists with *weighted* Reciprocal Rank
Fusion: score(d) = sum_i w_i / (c + rank_i(d)). With equal weights and c = rrf_k
this orders documents like our RRF (k = 60), but the inputs differ:

- ``rank_bm25.BM25Okapi`` uses a different IDF (with an epsilon floor) from
  ``src.index``, and ``get_top_n`` returns the top k even when their score is 0,
  while ``src.index.bm25_search`` drops zero-score chunks.
- Exact RRF ties are broken by first appearance across the lists, as ours is,
  but the lists themselves can differ for the reasons above.

Nothing here makes network calls: embeddings go to the local Ollama through
``src.index.embed_text`` (localhost only) and vectors come from PostgreSQL.
LangChain tracing (LangSmith) is only active when its environment variables are
set; this module never sets them. LangChain is imported only when this engine
is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr

from src import db
from src.index import _tokenize, embed_text, get_embedding_prefix


def _to_document(chunk: Dict[str, Any], **extra: Any) -> Document:
    return Document(
        page_content=chunk.get("text", ""),
        metadata={"chunk_id": chunk["chunk_id"], **extra},
    )


class PgVectorRetriever(BaseRetriever):
    """Cosine-similarity search over stored chunk embeddings via ``db.vector_search``."""

    chunks_by_id: Dict[str, Dict[str, Any]]
    doc_ids: List[str]
    k: int = 20
    config_path: str = "config.toml"
    # Embeds the query; defaults to the local model with the search_query prefix.
    embed_query: Callable[[str], List[float]] | None = None
    _last_top_score: float | None = PrivateAttr(default=None)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.embed_query is not None:
            vector = self.embed_query(query)
        else:
            prefix = get_embedding_prefix("query", config_path=self.config_path)
            vector = embed_text(query, prefix=prefix, config_path=self.config_path)
        if not vector:
            self._last_top_score = None
            return []
        results = db.vector_search(vector, self.doc_ids, self.k)
        self._last_top_score = results[0][1] if results else None
        return [
            _to_document(self.chunks_by_id[chunk_id], score=score)
            for chunk_id, score in results
            if chunk_id in self.chunks_by_id
        ]


class LangChainHybrid:
    """BM25 + pgvector, fused by ``EnsembleRetriever`` (weighted RRF)."""

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        doc_ids: List[str],
        *,
        top_n: int = 20,
        rrf_k: int = 60,
        config_path: str | Path = "config.toml",
        embed_query: Callable[[str], List[float]] | None = None,
    ) -> None:
        self.chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        self.top_n = top_n
        self.bm25 = BM25Retriever.from_documents(
            [_to_document(chunk) for chunk in chunks],
            preprocess_func=_tokenize,
            k=top_n,
        )
        self.vector = PgVectorRetriever(
            chunks_by_id=self.chunks_by_id,
            doc_ids=list(doc_ids),
            k=top_n,
            config_path=str(config_path),
            embed_query=embed_query,
        )
        self.ensemble = EnsembleRetriever(
            retrievers=[self.bm25, self.vector],
            weights=[0.5, 0.5],
            c=rrf_k,
            id_key="chunk_id",
        )

    def retrieve(self, query: str) -> Tuple[List[Dict[str, Any]], float | None]:
        """Return the fused chunks (best first, at most ``top_n``) and the top vector score."""
        documents = self.ensemble.invoke(query)
        chunks = [
            self.chunks_by_id[doc.metadata["chunk_id"]]
            for doc in documents
            if doc.metadata.get("chunk_id") in self.chunks_by_id
        ]
        return chunks[: self.top_n], self.vector._last_top_score
