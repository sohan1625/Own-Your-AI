# Tasks

Status key: [ ] todo, [~] in progress, [x] done

## Before coding

- [x] Write 30–50 eval questions from your own notes (EVAL.md) — 40 in `eval/questions.jsonl`
- [x] Check laptop RAM and GPU (BENCHMARKS.md)
- [x] Choose database and index store (DECISIONS.md)
- [x] Choose local LLM runtime and model size
- [x] Choose embedding model
- [x] Choose backend language and UI approach
- [x] Set up repo, lint, and a basic test runner
- [x] Test local model speed on target hardware (go/no-go check) — go: ~7 s per answer (14 s p95) on an RTX 3050 6 GB; see BENCHMARKS.md

## M1: One answer with a citation

- [x] Parse one PDF folder into pages with text
- [x] Implement chunker with size and overlap, unit tests
- [x] Implement BM25 over chunks, unit tests
- [x] Build keyword-only query path
- [x] Generate an answer from top chunks with chunk IDs
- [x] Map chunk IDs to file, page, and span
- [x] Demo one question end to end (`scripts/demo.py`)

## M2: Hybrid retrieval and measurement

- [x] Add embeddings and vector search — nomic-embed-text, stored in PostgreSQL with pgvector
- [x] Implement RRF fusion — `src/fuse.py`, k = 60; used by the app
- [x] Implement "not found" threshold — 0.605 on the top vector score, from `eval/calibration.jsonl` (on the eval set it refuses 4 answerable questions and passes 2 unanswerable ones; see EVAL.md)
- [x] Freeze eval set — `eval/questions.jsonl` unchanged since 2026-11-10
- [x] Run retrieval metrics for keyword, vector, and hybrid — `eval/results_*.json`
- [x] Record results in EVAL.md — includes misses and their causes

## M3: Quality and management

- [ ] Add reranker; measure with and without
- [~] Faithfulness and citation evaluation — hallucination on unanswerable measured (hybrid 20%, target < 10%); faithfulness and citation validity not measured
- [~] Incremental re-indexing by file hash — re-uploading an identical file is skipped; a changed file is added as a new document and the old version is not replaced
- [x] Deletion removes chunks, vectors, and index entries — also removes cached answers and the stored PDF
- [ ] Retrieval inspector view
- [x] Simple UI: query, answer, citations, document list
- [x] Draft README

## M4: Proof and packaging

- [x] Run offline test and record in PRIVACY.md — 9 of 9 with Wi-Fi off; online audit: no outbound connections from the app or Ollama server (`python -m scripts.offline_check`)
- [x] Run deletion test — 15 of 15 checks pass (`python -m scripts.deletion_check`); chat history and PostgreSQL's write-ahead log retain text, see PRIVACY.md
- [x] Run benchmarks and record in BENCHMARKS.md
- [ ] Quantized vs. full-precision comparison
- [ ] Setup under 15 minutes, verified by a clean run
- [ ] Final README with results
- [ ] Demo video (network off)
- [ ] Write-up of what worked and what didn't
- [ ] Write LIMITATIONS.md (scanned PDFs, tables, failure cases)

## Built outside the plan

- [x] PDF upload, document list, and delete from the UI
- [x] Chats saved in PostgreSQL; sidebar with New chat and chat history
- [x] Follow-up questions use the previous question for search and recent turns in the prompt
- [x] Answer cache in PostgreSQL (repeat questions return in ~0.2 s instead of ~19 s)

## Later (v1.1+)

- [x] Conversation memory — chats persist; follow-ups use recent turns
- [ ] Document summaries
- [~] Filters — choose which documents to search; no other filters
- [ ] Export Q&A as markdown
