# PRD: Own Your Intelligence (working title)

**Status:** Draft v0.2
**Owner:** Sohan
**Last updated:** 2026-11-09

---

## 1. Summary

A personal AI assistant that answers questions over a user's own documents, runs fully on their machine, and cites the exact source for every answer. No data leaves the device. The retrieval pipeline is built in-house rather than assembled from a framework, so every stage can be explained and measured.

## 2. Problem

- Students have lecture notes, PDFs, and slides spread across folders. Finding a specific answer means searching manually or re-reading.
- General AI chat tools need uploads to the cloud. Many users, especially students and anyone handling sensitive material, don't want their notes or documents sent to third parties.
- Most "chat with your docs" tools are cloud-first, and the local ones rarely publish measured retrieval quality.

## 3. Goals

1. Answer natural-language questions over a user's local documents.
2. Show a citation (file, page, passage) for every answer.
3. Work fully offline after setup, and prove it with a network-off test.
4. Implement the retrieval layer (chunking, hybrid search, fusion, citation logic) directly, using libraries only for the model call and file parsing.
5. Measure retrieval and answer quality on a fixed eval set, and report the numbers.
6. Benchmark latency and memory on a laptop-class machine.
7. Be simple enough to set up in under 15 minutes.

## 4. Non-goals (v1)

- Multi-user accounts, RBAC, audit logging, or multi-tenant isolation.
- Cloud sync or hosted version.
- Company or team deployment.
- Web browsing or external data sources.
- Editing or writing back to the user's documents.
- Mobile apps.
- Custom inference engine (C++/Rust). Use an existing local runtime for the model.
- GraphRAG or knowledge graph construction (candidate for v2).

## 5. Target users

- **Primary:** Engineering or CS students with large sets of PDF and markdown notes.
- **Secondary:** Developers or researchers who want a private knowledge base over their own files.

## 6. User stories

- As a student, I ask "What did the lecture say about B-trees?" and get an answer with the slide and page cited.
- As a student, I add a new folder of notes and the assistant indexes it without re-indexing everything.
- As a privacy-conscious user, I turn off my internet connection and the assistant still works.
- As a user, I delete a document and its content is removed from the index and answers.
- As a user, I can see which chunks the assistant used for an answer and why each was retrieved.
- As a user, if my notes don't contain the answer, the assistant says so.

## 7. Features

### Must have (v1)

| # | Feature | Description |
|---|---------|-------------|
| F1 | Document ingestion | Import PDF, markdown, and plain text from chosen folders. Parsing may use a library. |
| F2 | Chunking | Custom chunking with configurable size and overlap. Keep source metadata (file, page, character span). Support parent-chunk retrieval: match on small chunks, return the surrounding section as context. |
| F3 | Keyword retrieval | BM25 implemented over chunks (custom implementation or a small, inspected library). |
| F4 | Vector retrieval | Embedding-based similarity search over chunks, using a local embedding model. |
| F5 | Hybrid fusion | Merge keyword and vector result lists with reciprocal rank fusion. Make the fusion weights configurable for evaluation. |
| F6 | Reranking | Optional cross-encoder reranker over the top-N fused candidates. Measure with and without it. |
| F7 | Grounded answering | Generate an answer using only retrieved context. Prompt must forbid claims outside the context. |
| F8 | Citations | Every answer lists source file, page, and span. Show the passage on click. |
| F9 | "I don't know" behavior | If retrieval scores fall below a threshold, or the model reports the context lacks the answer, say so. |
| F10 | Local model inference | Run the language model and embedding model on the user's machine. Support a quantized and a full-precision model for comparison. |
| F11 | Index management | Add, remove, and re-index folders. Detect changed files by content hash. |
| F12 | Deletion | Removing a document deletes its chunks, vectors, and index entries. |
| F13 | Retrieval inspector | A debug view showing the top retrieved chunks, their scores, and which retriever found each. |
| F14 | Simple UI | Query box, answer with citations, document list, index status. |
| F15 | Offline mode | Full functionality with no network. Verifiable by a network-off test. |

### Should have (v1.1)

- Conversation memory within a session.
- Summaries of a single document or a folder.
- Filters (by folder, file type, or date added).
- Export of a Q&A session as markdown.

### Could have (later)

- Image and scanned-PDF support (OCR).
- Obsidian, Notion, and Google Drive export import.
- GraphRAG: a local knowledge graph over entities for multi-hop questions.
- Voice input.
- Multi-device sync with end-to-end encryption.

## 8. Privacy and security requirements

- No outbound network calls during normal use. Only model download and updates may use the network, and both must be explicit.
- Verify this with a test: run the full eval with the network disabled and confirm zero outbound connections (e.g., packet capture or firewall logs).
- No telemetry or analytics by default.
- All indexed content and embeddings stored locally, in a location the user controls.
- Deleting a document removes its chunks, embeddings, and derived data. Test this.
- Clearly document what is stored, where, and in what format.

## 9. Quality and evaluation

**Eval set:** 30–50 questions with known answers, gold source passages, and a note on whether the answer is in the documents at all (including some unanswerable questions).

**Retrieval metrics:**
- Precision@k and recall@k for k = 1, 3, 5, using gold passages.
- Mean reciprocal rank.
- Compare keyword only, vector only, hybrid, and hybrid + reranker.

**Answer metrics:**
- Answer correctness (human-labeled on a sample, then automated if it's reliable enough).
- Faithfulness: is each claim supported by the retrieved text? Use RAGAS or a simple LLM-as-judge with a manual spot check.
- Hallucination rate on unanswerable questions: how often it answers instead of saying "I don't know."
- Citation accuracy: does the cited passage actually contain the claim?

**Targets for v1:**
- Recall@5 of at least 0.80 with hybrid retrieval.
- At least 0.90 faithfulness on answerable questions.
- Hallucination rate under 10% on unanswerable questions.
- 100% of answers include a citation.

Report every number, including the ones that miss target, with a short explanation.

## 10. Performance and system benchmarks

Benchmark on a stated machine (CPU, RAM, GPU if any) and report:

- Indexing time per 100 pages.
- Query latency: retrieval time and end-to-end answer time, at p50 and p95.
- Peak RAM during indexing and during querying.
- Index size on disk per 100 pages.
- Quality vs. speed for a quantized model compared to a full-precision model on the same eval set.

## 11. Requirements

- **Hardware (minimum):** set after the benchmark. Document the RAM and disk needs.
- **Platform:** pick one primary platform for v1 (Linux or macOS recommended for the first build).
- **Setup:** one documented command or script after cloning.
- **Licensing:** only use models and libraries whose licenses allow personal and portfolio use.

## 12. Success metrics

- All v1 quality targets in section 9 are met, or the misses are explained in the write-up.
- Benchmarks from section 10 are published in the README.
- Demo runs end-to-end with the network disabled.
- A new user can set up and ask a first question in under 15 minutes.

## 13. Open questions

- Which document types matter most for the first user? (Decides the ingestion order.)
- Which local model size and quantization run well on the target machine?
- Which reranker model fits the hardware budget?
- Single platform or cross-platform for v1?
- What is the name? ("Own Your Intelligence" is a working title.)
- Does the assistant store conversation memory, and if so, where and how can the user view or delete it?

## 14. Milestones

1. **M1:** Ingest one PDF folder, implement chunking and keyword retrieval, and answer one question with a citation.
2. **M2:** Vector retrieval, hybrid fusion, "I don't know" handling. Eval set of 30–50 questions created and retrieval metrics reported for each retriever.
3. **M3:** Reranker, faithfulness and citation evaluation, deletion and re-indexing, retrieval inspector.
4. **M4:** Offline test, benchmarks, basic UI, README with results, demo video, and write-up.
