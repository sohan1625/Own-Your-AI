# RAG Pipeline

This file describes each stage, its inputs and outputs, and the parameters to tune. Every stage is custom code unless noted.

## 1. Ingestion

- **In:** file paths in a watched folder
- **Out:** plain text per page, with page number and source file
- Hash each file (SHA-256). Skip files whose hash hasn't changed.
- Parsing libraries are allowed here.

## 2. Chunking

- **In:** page text
- **Out:** chunks with `file`, `page`, `start_char`, `end_char`, `text`
- Parameters: chunk size (tokens), overlap (tokens). Start with 400 and 50, then tune on EVAL.md.
- Keep a parent reference so a matched small chunk can return its surrounding section.

## 3. Indexing

- Keyword index: BM25 term statistics over chunks.
- Vector index: one embedding per chunk from the local embedding model.
- Re-indexing is incremental: changed files are re-chunked, unchanged files are skipped.
- Deleting a file removes its chunks, postings, and vectors.

## 4. Retrieval

- Keyword retriever: BM25 top-N (N = 20).
- Vector retriever: cosine similarity top-N (N = 20).
- Both run for every query so the eval can compare them.

## 5. Fusion

- Reciprocal rank fusion: score(d) = Σ 1 / (k + rank_i(d)), with k = 60 as a starting value.
- Weights per retriever are configurable for experiments.

## 6. Reranking (optional)

- Cross-encoder scores (query, chunk) pairs for the top 20 fused results.
- Keep the top 5 for generation.
- Always report the result with and without reranking.

## 7. Threshold check

- If the best fused or reranked score is below the threshold, return "not found in your documents." Threshold is set on the eval set, not guessed.

## 8. Answer generation

- Prompt includes only the selected chunks, each labeled with an ID.
- Instruction: answer only from the provided chunks; if they don't contain the answer, say so.
- Output must reference chunk IDs for each claim.

## 9. Citation

- Map chunk IDs in the answer to file, page, and span.
- Check that each cited chunk exists in the retrieved set. Drop invalid citations and flag the answer.

## Parameters to record

| Parameter | Default | Tuned on |
|-----------|---------|----------|
| Chunk size | 400 tokens | EVAL.md |
| Overlap | 50 tokens | EVAL.md |
| Top-N per retriever | 20 | EVAL.md |
| RRF k | 60 | EVAL.md |
| Rerank top-N / keep | 20 / 5 | EVAL.md |
| Not-found threshold | TBD | EVAL.md |
