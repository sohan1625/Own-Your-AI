# Decisions

Record each technical choice as it's made: the options considered, what was picked, and why. Include rejected options. Interviewers ask about these.

Format for each entry:

```
## D-00X: Title
Date:
Status: Proposed / Accepted / Superseded by D-00Y
Context:
Options considered:
Decision:
Reasons:
Tradeoffs accepted:
```

---

## D-001: Project scope and privacy model
Date: 2026-11-09
Status: Accepted
Context: Choosing the core idea for the portfolio project.
Options considered: Cloud RAG chatbot; enterprise private chatbot; local-first document assistant.
Decision: Local-first document assistant with a verifiable offline guarantee.
Reasons: Clear privacy story, measurable quality, feasible in the time available.
Tradeoffs accepted: Smaller local models; weaker reasoning than cloud models. Crowded category, so differentiation comes from measurement and custom retrieval.

## D-002: Retrieval built in-house
Date: 2026-11-09
Status: Accepted
Context: Whether to use a framework (e.g., LangChain) for the RAG pipeline.
Options considered: Framework wrapper; custom pipeline with libraries only for model calls and parsing.
Decision: Custom chunking, BM25, vector search, fusion, and citation logic.
Reasons: Each stage can be explained and measured. Framework abstractions hide the decisions interviewers ask about.
Tradeoffs accepted: More code to write and test.

## D-003: Hybrid retrieval with reciprocal rank fusion
Date: 2026-11-09
Status: Accepted
Context: Keyword search alone misses paraphrases; vector search alone misses exact terms.
Options considered: Keyword only; vector only; hybrid with weighted score sum; hybrid with reciprocal rank fusion.
Decision: Hybrid with RRF.
Reasons: No need to normalize scores across retrievers.
Tradeoffs accepted: Two indexes to maintain.

## D-004: Backend language is Python
Date: 2026-11-09
Status: Accepted
Context: Choosing the language for ingestion, retrieval, and the API.
Options considered: Python + FastAPI; Java + Spring Boot; Node + TypeScript.
Decision: Python, with FastAPI for the API.
Reasons: The PDF parsing, embedding, and reranker libraries are Python-first, so more time goes to retrieval and less to glue code.
Tradeoffs accepted: Does not use the Spring Boot skills on this project. Java remains the focus for other portfolio work.

## D-005: Database is PostgreSQL with pgvector
Date: 2026-11-09
Status: Accepted
Context: Choosing where chunks, metadata, keyword index, and vectors are stored.
Options considered: SQLite + sqlite-vec; Postgres + pgvector; Qdrant (local).
Decision: PostgreSQL with the pgvector extension, run locally in Docker.
Reasons: Vectors, relational metadata, and full-text search in one system. Matches existing Postgres experience from the inventory project. Stronger resume signal and scales to multi-user later.
Tradeoffs accepted: Needs Docker running, which is heavier on Windows than SQLite's single file. Postgres's built-in full-text ranking (ts_rank) is not BM25, so BM25 is implemented in our own code on top of stored term statistics, per D-002.

## D-006: Target platform is Windows
Date: 2026-11-09
Status: Accepted
Context: PRD section 11 recommends Linux or macOS for v1. The developer's machine runs Windows.
Options considered: Windows; Linux; macOS.
Decision: Windows is the primary and only v1 target. Setup guide and benchmark machine are Windows.
Reasons: The development and benchmark machine is Windows, so testing on the same platform avoids guesswork. Ollama, Python, and PostgreSQL all run on Windows.
Tradeoffs accepted: PostgreSQL runs in Docker Desktop with WSL2, which is heavier than on Linux. Benchmarks on Windows may be slower than on Linux and should be stated in the write-up. Other platforms are untested.

## D-007: Local LLM runtime is Ollama; main model is Mistral 7B Instruct
Date: 2026-11-09
Status: Accepted
Context: Choosing the runtime and model for answer generation on a laptop with a 6 GB GPU and 24 GB RAM.
Options considered: Runtime: Ollama, llama.cpp, LM Studio. Models: Qwen2.5 7B and 14B, Mistral 7B Instruct, Llama 3.1 8B Instruct, Phi-4 14B.
Decision: Ollama as the runtime. Mistral 7B Instruct (4-bit) as the main model. Phi-4 14B (4-bit) as the comparison model for the quality-vs-speed benchmark.
Reasons: Ollama is simple on Windows and handles GPU use and quantization. Mistral 7B fits fully in 6 GB VRAM and is Apache 2.0, the simplest license for a portfolio project. Phi-4 is MIT licensed and stronger, but too large for the GPU, so it runs partly on CPU and is used only for comparison.
Tradeoffs accepted: Mistral 7B is older than the newest small models and may trail Llama 3.1 8B on general tasks. Qwen2.5 was considered and dropped at the owner's choice. Model choice is to be confirmed by the faithfulness eval; if it fails the target, the model will be superseded with a recorded reason.

## D-008: Embedding model is nomic-embed-text
Date: 2026-11-09
Status: Accepted
Context: Choosing the model that turns chunks and queries into vectors for semantic search.
Options considered: nomic-embed-text; bge-m3; all-MiniLM-L6-v2.
Decision: nomic-embed-text, run through Ollama.
Reasons: Small and fast on a laptop, Apache 2.0 license. Indexing will not strain the 24 GB RAM or the 6 GB GPU. Retrieval quality is to be confirmed by the eval.
Tradeoffs accepted: Less accurate than bge-m3 on multilingual or hard queries. Changing it later means re-embedding every chunk and rebuilding the vector index, so the eval should confirm it before the index is built.

## D-009: Reranker is an optional M3 experiment, not in v1
Date: 2026-11-09
Status: Accepted
Context: Whether to include a cross-encoder reranker in v1.
Options considered: Reranker in v1 (MiniLM cross-encoder or bge-reranker-v2-m3); reranker as a measured M3 experiment; no reranker.
Decision: v1 ships without a reranker. In M3, test the MiniLM cross-encoder against the no-reranker baseline on the frozen eval set. Keep it only if it measurably improves recall or faithfulness.
Reasons: The PRD requires measuring with and without reranking. Starting without it keeps the pipeline simpler and makes the first working version faster to reach.
Tradeoffs accepted: If the reranker helps, the v1 numbers come from a later milestone. It is also the easiest component to drop if time runs short.

## Pending
- None. All stack decisions are recorded. Next: write the eval set and run the M1 tasks.
