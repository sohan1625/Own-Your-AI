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
Implementation note (2026-11-20): the app now uses hybrid retrieval (BM25 + pgvector, RRF k = 60). On the eval set it raised Recall@3 from 0.867 (keyword) to 0.967. `[retrieval] method = "keyword"` switches it off.

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
Implementation note: PostgreSQL now stores document metadata, chunks, and chat history. Docker and the rag-pgvector container must be running for the application to start.

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

## D-010: Answer cache in PostgreSQL
Date: 2026-11-20
Status: Accepted
Context: Each answer takes about 7 s on the target laptop. Repeated questions are common in a personal Q&A tool.
Options considered: No cache; in-memory cache; cache table in PostgreSQL.
Decision: An `answer_cache` table keyed by a SHA-256 of the normalized question, the sorted document IDs, and each document's hash. Refusals and follow-up questions are not cached. Deleting a document deletes every cached answer that used it.
Reasons: Survives restarts, uses the database already in place, and including document hashes in the key means a changed document can never return a stale answer. A repeat question returns in about 0.13 s.
Tradeoffs accepted: Follow-ups are never cached, because the key does not include the chat. A cached answer is fixed even if the model or prompt changes; the cache must be cleared after such changes.

## D-011: Follow-up search uses the previous question, not the previous answer
Date: 2026-11-20
Status: Accepted
Context: Follow-ups such as "Explain those." need context to search. The first version added the previous question and answer to the search query.
Options considered: Search with the new question only; add the previous question and answer; add only the previous question; run both searches and fuse them.
Decision: Add only the previous user question. The full recent chat still goes into the prompt.
Reasons: A long previous answer outweighed the new question: "explain firewalls" asked after "what is osi model" ranked the right page 9th (only 4 chunks reach the model); with the previous question only it ranked 2nd. The new question alone finds nothing for "Explain those.", and fusing both searches dropped the right page for that case to 13th.
Tradeoffs accepted: A previous question that strongly matches its own topic can still compete with the new one.

## D-012: Keep the citation fallback and the vector-score not-found threshold
Date: 2026-11-20
Status: Accepted (revisit with a reranker or a model that cites reliably)
Context: On the eval set, 2 of 10 unanswerable questions were answered from general knowledge, and 4 of 30 answerable ones were refused by the threshold. The citation fallback (append the top chunk's ID when the model cites nothing) let the general-knowledge answers look cited.
Options considered: Drop the fallback; drop the threshold; drop both; keep both.
Decision: Keep both. Measured on one set of 40 answers: both on gives 2 hallucinations and 4 false refusals; threshold off gives 7 and 0; fallback off gives 0 and 21; both off gives 1 and 18.
Reasons: mistral leaves out citations on most correct answers, so without the fallback most correct answers become refusals. Without the threshold most unanswerable questions get answered.
Tradeoffs accepted: The hallucination target (< 10%) is still missed. Reworded refusals are normalized to "not found in your documents" so they are neither cached nor shown with a citation.

## D-013: Reach Ollama at 127.0.0.1, not localhost
Date: 2026-11-20
Status: Accepted
Context: Every embedding and generation request took about 2.1 s longer than it should.
Options considered: Keep `localhost`; use `127.0.0.1`; make Ollama listen on IPv6 as well.
Decision: `http://127.0.0.1:11434` in `config.toml` and the code defaults.
Reasons: On Windows, `localhost` resolves to IPv6 `::1` first, and Ollama listens only on IPv4, so each request waited for the IPv6 attempt to fail. One embedding went from about 2,150 ms to 25 ms; indexing a 47-page PDF from 98.7 s to 3.6 s.
Tradeoffs accepted: None found. The localhost-only privacy guard accepts both forms.

## D-014: App uploads live in storage/uploads, not data/
Date: 2026-11-20
Status: Accepted
Context: Uploads were saved in `data/uploads/`, inside `data/`, which the eval scripts scan. An uploaded PDF silently became part of the eval corpus.
Options considered: Make the eval skip `data/uploads/`; move uploads elsewhere.
Decision: Save uploads in `storage/uploads/` (git-ignored). `data/` holds only the eval corpus.
Reasons: Keeps the app's data and the eval corpus separate without special cases in the eval code.
Tradeoffs accepted: Existing uploads had to be moved once.

## Pending
- Whether to add a reranker (D-009), and whether a different model would cite reliably enough to drop the citation fallback (D-012).
