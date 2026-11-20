# RAG Pipeline

This file describes each stage as implemented, its inputs and outputs, and the parameters to tune (all in `config.toml`). Every stage is custom code unless noted. Items marked *not implemented* are planned but not built.

## 1. Ingestion (`src/ingest.py`)

- **In:** an uploaded PDF (the app), or the PDFs in `data/` (the eval scripts).
- **Out:** plain text per page, with page number and source file.
- Text is extracted with `pypdf` (a parsing library is allowed here). Repeated headers and footers (titles, "Page N") are removed, and table-of-contents pages are skipped.
- Each file is hashed (SHA-256). Uploading a file with the same hash again is skipped. *Not implemented:* replacing a changed file. An edited PDF is added as a new document and the old version stays.

## 2. Chunking (`src/chunk.py`)

- **In:** page text.
- **Out:** chunks with `chunk_id`, `file`, `page`, `start_char`, `end_char`, `text`.
- Chunk size 400 tokens, overlap 50 tokens. Tokens are approximated from word counts (about 0.75 words per token) to avoid a tokenizer dependency. Chunks never cross a page boundary.
- *Not implemented:* a parent reference so a matched small chunk can return its surrounding section.

## 3. Indexing (`src/db.py`, `src/index.py`)

- Chunks are stored in PostgreSQL.
- Vector index: each chunk gets one 768-dimension embedding from `nomic-embed-text` (with the `search_document:` prefix), stored in a pgvector column on `chunks`. New uploads are embedded before they are saved. If Ollama is unavailable, the upload still succeeds without embeddings, and missing embeddings are filled in at the next startup.
- Keyword index: BM25 term statistics, built in memory from the stored chunks on first use and rebuilt after every upload or delete.
- Deleting a document removes its file, chunks, embeddings, and cached answers, and the BM25 index is rebuilt.

## 4. Retrieval (`src/api.py`, `src/retrieve.py`)

- Both retrievers search only the documents selected in the UI.
- Keyword retriever: BM25 top-N (N = 20).
- Vector retriever: the question is embedded (with the `search_query:` prefix) and pgvector returns the top-N chunks by cosine similarity (N = 20).
- **Follow-up questions:** in an existing chat, the search query is the previous user question plus the new question, so "Explain those." still finds the right chunks. The previous *answer* is left out of the search. A long answer outweighed the new question and buried the chunks it needed (D-011).
- `[retrieval] method = "keyword"` turns vector search off. If embedding fails at query time, retrieval falls back to keyword only.

## 5. Fusion (`src/fuse.py`)

- Reciprocal rank fusion: score(d) = Σ 1 / (k + rank_i(d)), with k = 60.
- Both retrievers have equal weight. *Not implemented:* per-retriever weights.

## 6. Reranking (optional)

- *Not implemented* (D-009). Planned: a cross-encoder scores the top 20 fused chunks and keeps the top 5, measured with and without reranking.

## 7. Threshold check (`src/answer.py`)

- If the best **vector** (cosine) score is below 0.605, the answer is "not found in your documents" and the model is not called.
- The threshold was chosen on `eval/calibration.jsonl`, not on the eval set. On the eval set it wrongly refuses 4 of 30 answerable questions and lets 2 of 10 unanswerable ones through; see EVAL.md.
- With no vector score (keyword-only mode), no threshold check is made.

## 8. Answer generation (`src/answer.py`)

- The top 4 fused chunks, each labelled with its ID, go to `mistral` via Ollama on 127.0.0.1.
- The prompt includes the last 2 chat messages, for resolving follow-up references only; factual claims must come from the chunks.
- The model is told to answer only from the chunks, cite chunk IDs, and say "not found in your documents" otherwise.
- **Reworded refusals:** if the answer contains "not found in your documents" and cites no chunk, it is replaced by exactly "not found in your documents" (the model often rewords its refusal).
- **Citation fallback:** if an answer cites no chunk, the top chunk's ID is appended. The model leaves out citations on most correct answers, and removing the fallback refused 18–21 of 30 answerable eval questions (EVAL.md, D-012).

## 9. Citation (`src/cite.py`)

- Chunk IDs in the answer are mapped to file, page, and span.
- Citations to chunks that were not in the provided set are dropped and the answer is flagged.

## 10. Answer cache (`src/cache.py`)

- Key: SHA-256 of the normalized question (lowercase, collapsed whitespace), the sorted document IDs, and each document's SHA-256.
- Checked before retrieval. A hit returns the stored answer and citations (`cached: true`) and is still saved to the chat.
- Not cached: refusals, and follow-up questions (their answer depends on the chat, which the key does not include).
- Deleting a document deletes every cached answer whose document selection included it.
- `[cache] enabled = false` turns it off.

## Parameters

| Parameter | Value | Set from |
|-----------|-------|----------|
| Chunk size | 400 tokens | Default, not yet tuned |
| Overlap | 50 tokens | Default, not yet tuned |
| Top-N per retriever | 20 | Default |
| RRF k | 60 | Default |
| Chunks sent to the model | 4 | Default |
| Rerank top-N / keep | 20 / 5 | Not implemented |
| Not-found threshold | 0.605 (top vector score) | `eval/calibration.jsonl` |
