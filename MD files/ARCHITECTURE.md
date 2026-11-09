# Architecture

## Pipeline

```
Ingest → Chunk → Index → [Query] → Keyword search (BM25)  ─┐
                                  → Vector search          ─┼→ Fuse (RRF) → Rerank (optional) → Answer → Cite
                                                             ┘
```

## Components

| Component | Responsibility | Notes |
|-----------|---------------|-------|
| Ingestion | Read PDF, markdown, text; extract text with page and position | Parsing library allowed |
| Chunker | Split text into overlapping chunks, keep source metadata | Custom |
| Index store | Store chunks, metadata, BM25 index, vectors | Stack TBD (see DECISIONS.md) |
| Embedding model | Turn chunks and queries into vectors | Local |
| Keyword retriever | BM25 scoring over chunks | Custom |
| Vector retriever | Nearest-neighbor search over embeddings | Custom query logic |
| Fusion | Reciprocal rank fusion of both result lists | Custom, weights configurable |
| Reranker | Cross-encoder over top-N candidates | Optional, measured on and off |
| Generator | Local LLM answers using only retrieved context | Local runtime |
| Citation builder | Map answer claims to source passages | Custom |
| UI | Query, answer with citations, document list, retrieval inspector | Simple |

## Data flow

1. A folder is added. Files are hashed, parsed, and chunked.
2. Chunks are embedded and written to the index store.
3. A query is run through both retrievers. Results are fused and optionally reranked.
4. If the top scores fall below a threshold, return "not found."
5. Otherwise the generator answers from the top chunks, and the citation builder attaches sources.

## Boundaries

- Everything runs on the user's machine. No network calls during normal use (see PRIVACY.md).
- Models are loaded from local files after an explicit download step.

## Open decisions

- Language and framework for the backend and UI
- Index store and vector storage
- Local LLM runtime and model size
- Embedding model
- Reranker model
