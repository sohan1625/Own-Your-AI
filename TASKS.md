# Tasks

Status key: [ ] todo, [~] in progress, [x] done

## Before coding

- [ ] Write 30–50 eval questions from your own notes (EVAL.md)
- [x] Check laptop RAM and GPU (BENCHMARKS.md)
- [x] Choose database and index store (DECISIONS.md)
- [x] Choose local LLM runtime and model size
- [x] Choose embedding model
- [x] Choose backend language and UI approach
- [x] Set up repo, lint, and a basic test runner
- [ ] Test local model speed on target hardware (go/no-go check)

## M1: One answer with a citation

- [x] Parse one PDF folder into pages with text
- [x] Implement chunker with size and overlap, unit tests
- [x] Implement BM25 over chunks, unit tests
- [x] Build keyword-only query path
- [ ] Generate an answer from top chunks with chunk IDs
- [ ] Map chunk IDs to file, page, and span
- [ ] Demo one question end to end

## M2: Hybrid retrieval and measurement

- [ ] Add embeddings and vector search
- [ ] Implement RRF fusion
- [ ] Implement "not found" threshold
- [ ] Freeze eval set
- [ ] Run retrieval metrics for keyword, vector, and hybrid
- [ ] Record results in EVAL.md

## M3: Quality and management

- [ ] Add reranker; measure with and without
- [ ] Faithfulness and citation evaluation
- [ ] Incremental re-indexing by file hash
- [ ] Deletion removes chunks, vectors, and index entries
- [ ] Retrieval inspector view
- [ ] Simple UI: query, answer, citations, document list
- [ ] Draft README

## M4: Proof and packaging

- [ ] Run offline test and record in PRIVACY.md
- [ ] Run deletion test
- [ ] Run benchmarks and record in BENCHMARKS.md
- [ ] Quantized vs. full-precision comparison
- [ ] Setup under 15 minutes, verified by a clean run
- [ ] Final README with results
- [ ] Demo video (network off)
- [ ] Write-up of what worked and what didn't
- [ ] Write LIMITATIONS.md (scanned PDFs, tables, failure cases)

## Later (v1.1+)

- [ ] Conversation memory
- [ ] Document summaries
- [ ] Filters
- [ ] Export Q&A as markdown
