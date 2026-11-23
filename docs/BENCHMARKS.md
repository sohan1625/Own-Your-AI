# Benchmarks

Run on 2026-11-20 against the real app pipeline (`POST /upload` and `POST /ask`), using an isolated PostgreSQL schema that was dropped afterwards.

## Test machine

- CPU: AMD Ryzen 5 7235HS (4 cores, 8 threads)
- RAM: 24 GB (23.7 GB usable)
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 6 GB VRAM (driver 617.14, CUDA 13.4)
- Disk: SSD
- OS: Windows 11 Home (10.0.26300), 64-bit
- Power mode: plugged in
- Software: Ollama 0.40.16, PostgreSQL 16.15 with pgvector 0.8.7 in Docker 29.8.1, Python 3.14
- Models: mistral 7B Q4_K_M (4-bit, 4.4 GB) for answers; nomic-embed-text F16 (137M, 0.3 GB) for embeddings

## Corpus

- Documents: 1 (`Core_Java_Complete_Answers.pdf`, the eval corpus)
- Total pages: 47 (44 with text after table-of-contents removal), 44 chunks
- Total words: 9,227
- Questions: the 40 questions in `eval/questions.jsonl` (30 answerable, 10 unanswerable)

## Method

- Indexing: the PDF was uploaded 5 times through `POST /upload` (parse, chunk, embed, insert), deleting it between runs. Reported as median and range.
- Querying: each of the 40 eval questions was asked once through `POST /ask` with the answer cache on (uncached pass), then all 40 again (cached pass). 40 samples per pass, so p50 and p95 come from real distributions rather than 5 repeats. One untimed warm-up question loaded the model first.
- Retrieval time covers keyword search, embedding the question, vector search, and fusion.
- Memory: sampled every 0.5 s (working set of the Ollama runner processes, the Docker VM, and the app process; GPU memory from `nvidia-smi`).
- Conditions: other heavy apps were closed before the run. At the start 3.2 GB of RAM was free; Chrome and Notion were closed partway through indexing.

## Indexing

| Metric | Result |
|--------|--------|
| Time per 100 pages | 7.6 s (median of 5 runs: 3.6 s for 47 pages, range 3.3–3.6 s; embedding was 2.2 s of that) |
| Peak RAM during indexing | 9.6 GB total: Ollama runner 8.9 GB (both models loaded), PostgreSQL VM 0.55 GB, app 0.09 GB. GPU 5.4 of 6 GB |
| Index size on disk per 100 pages | 0.78 MB (376 KB for 47 pages: chunk text, 768-dimension embeddings, and indexes, measured after a single clean index) |

## Querying

| Metric | p50 | p95 |
|--------|-----|-----|
| Retrieval time | 0.11 s | 0.13 s |
| End-to-end answer time, generated (28 questions) | 7.0 s | 14.1 s |
| End-to-end answer time, all 40 uncached (includes 12 threshold refusals, ~0.2 s each) | 6.1 s | 10.1 s |
| End-to-end answer time, cached (28 cache hits; refusals are never cached and rerun retrieval) | 0.13 s | 0.30 s |

Slowest answers: 22.6 s and 16.4 s, both long general-knowledge answers to unanswerable questions (Kubernetes deployment, PostgreSQL replication).

Peak RAM during querying: 9.6 GB total (Ollama runner 9.0 GB, PostgreSQL VM 0.54 GB, app 0.08 GB). GPU 5.5 of 6 GB. mistral does not fit entirely in 6 GB of VRAM, so about 0.9 GB of it runs from system RAM.

## Go/no-go: speed on target hardware

**Go.** On a 6 GB laptop GPU, a typical answer takes about 7 s (14 s at p95), retrieval about 0.1 s, a repeated question about 0.1 s from the cache, and indexing about 8 s per 100 pages. Minimum hardware for this setup: a GPU with 6 GB VRAM plus about 10 GB of free RAM for the models and database. CPU-only was not tested.

## Fix found while benchmarking: "localhost" added 2 s per model request

`config.toml` pointed Ollama at `http://localhost:11434`. On Windows that name tries IPv6 (`::1`) first, Ollama listens only on IPv4, and each request waited about 2.1 s before falling back. Switching to `http://127.0.0.1:11434`:

| Measurement | With `localhost` | With `127.0.0.1` |
|-------------|------------------|------------------|
| One embedding request | ~2,150 ms | ~25 ms |
| Indexing the 47-page PDF | 98.7 s | 3.6 s |
| Retrieval per question | 2.17 s | 0.11 s |

## Quality vs. speed

| Model | Quantization | Recall@5 | Faithfulness | Answer time (p50) | Peak RAM |
|-------|-------------|----------|--------------|-------------------|----------|
| mistral 7B | Full precision (F16) | Not run | Not run | Not run | Not run |
| mistral 7B | Quantized (Q4_K_M, 4-bit) | 1.00 (hybrid) | Not measured | 7.0 s | 9.6 GB RAM + 5.5 GB VRAM |

The full-precision model (about 14.5 GB) does not fit in 6 GB of VRAM, so it would run mostly on the CPU; that comparison is still to do.

## Experiments (2026-11-23)

Each experiment is measured with `python scripts/eval.py --retriever hybrid --with-answers` on the frozen `eval/questions.jsonl` (40 questions: 30 answerable, 10 unanswerable) and the eval corpus `data/Core_Java_Complete_Answers.pdf`. Results JSON went to a scratch folder, not to `eval/results_*.json`.

Keep rule, fixed before any results were seen: a change is kept only if, against the current baseline on the same eval, Recall@3 and MRR do not drop, hallucination on unanswerable questions does not rise, and answerable questions wrongly refused do not rise.

Checks after each experiment: `pytest -q --basetemp=.pytest_tmp`, `python -m scripts.offline_check --with-network` (the strict offline mode refuses to run while the internet is reachable, and the network was left on), and `python -m scripts.deletion_check`.

### Exp 0: temperature 0 (kept; new baseline)

- Branch: `exp/temperature-zero` (from `main`)
- Settings: `[generation] temperature = 0`, sent to Ollama as `"options": {"temperature": 0}` (no `options` key when unset). Everything else as on `main`: fixed chunking (400 tokens, 50 overlap), hybrid BM25 + vector with RRF k = 60, top 20 per retriever, top 4 chunks to mistral, not-found threshold 0.605.
- Eval harness changes: (1) new summary field `answerable_wrongly_refused` (answerable questions whose final answer is the refusal); (2) eval chunks get stable IDs (`uuid5` of file, page and character span) instead of random `uuid4`. The app still uses random IDs.

Why (2): chunk IDs are part of the prompt. With random IDs the first pair of temperature-0 runs sent different prompts, and 12 of 40 answers differed in text (ignoring the IDs themselves), including q37 flipping between a hallucination and a refusal. With stable IDs the second pair of runs had identical per-question results, but 7 of 40 answers (q04, q17, q21, q22, q27, q29, q32) still differed in wording. A direct probe (the q17 prompt sent 4 times in a row, and 3 times interleaved with other prompts, with and without `seed`) gave identical output every time, so the leftover drift was not reproduced in isolation and its cause is not identified. **Determinism check: per-question outcomes identical across the two runs; answer text not byte-identical.**

| Metric | `main` (1 run, default temperature) | Exp 0 run 1 | Exp 0 run 2 |
|--------|------|------|------|
| Recall@1 | 0.6333 | 0.6333 | 0.6333 |
| Recall@3 | 0.9667 | 0.9667 | 0.9667 |
| Recall@5 | 1.0000 | 1.0000 | 1.0000 |
| MRR | 0.8028 | 0.8028 | 0.8028 |
| Hallucination (of 10) | 0.20 (q32, q37) | 0.10 (q32) | 0.10 (q32) |
| Answerable wrongly refused (of 30) | 4 (q03, q11, q13, q30) | 4 (same) | 4 (same) |

The `main` hallucination figure is one sample at the default temperature; in the first (random-ID) temperature-0 pair, q37 hallucinated in one run and not in the other, so the 0.20 → 0.10 change is within run-to-run noise and is not claimed as an improvement.

Decision: **kept**. No metric got worse. Exp 0 run 1 is the baseline for Exp 1 to 4.

```
{"total_questions": 40, "answerable_questions": 30, "unanswerable_questions": 10,
 "recall_at_1": 0.6333, "recall_at_3": 0.9667, "recall_at_5": 1.0, "mrr": 0.8028,
 "hallucination_rate": 0.1, "answerable_wrongly_refused": 4}
```

Checks: pytest 80 passed; offline check (online audit) 8 passed, 0 failed; deletion check 15 passed, 0 failed.

### Exp 3: faithfulness check (discarded by the keep rule; see note)

- Branch: `exp/faithfulness` (from `exp/temperature-zero`)
- Settings: Exp 0 baseline unchanged. New `src/faithfulness.py`, called from `scripts/eval.py` on every non-refusal answer; it never changes the answer text. Each answer sentence is scored against the chunk it cites (its own `[id]`; if it has none, the next citation in the answer, else the previous one). Score = share of the sentence's content words (lower-cased, stopwords removed, trailing plural "s" stripped) found in the cited chunk text. Supported when score ≥ 0.5, a threshold fixed before the run and not tuned.

| Metric | Baseline (Exp 0) | Exp 3 |
|--------|------|------|
| Recall@1 | 0.6333 | 0.6333 |
| Recall@3 | 0.9667 | 0.9667 |
| Recall@5 | 1.0000 | 1.0000 |
| MRR | 0.8028 | 0.8028 |
| Hallucination (of 10) | 0.10 (q32) | **0.20 (q32, q37)** |
| Answerable wrongly refused (of 30) | 4 | 4 (q03, q11, q13, q30) |
| Faithfulness, all non-refusal answers | not measured | 62 / 95 sentences supported (0.6526); 21 of 28 answers fully supported |
| Faithfulness, answerable questions only | not measured | 47 / 56 sentences (0.8393); 21 of 26 answers fully supported |
| Faithfulness, the 2 hallucinated answers | not measured | 15 / 39 sentences (0.3846); 0 of 2 fully supported |

Unsupported answerable sentences: q12, q16, q25 (2), q27, q29 (4). q25 shows the check's main use: the text matches the PDF, but the citation points at the page-41 chunk (daemon code example) while the definition is in another chunk, so the citation is wrong even though the answer is right. q29 shows a false negative: a correct paraphrase of the Statement vs PreparedStatement table scores 0.29–0.43.

Limits of the check (lexical overlap, not entailment): paraphrases score low (false "unsupported"); a flipped negation, swapped number, or wrong relation between copied words still scores high (false "supported"; unit-tested as a known limit); when mistral writes no citation the app attaches the top chunk's ID, so the check scores against that fallback chunk; very short sentences swing between 0 and 1; markdown/code lines in long answers (q37) are each counted as sentences.

Decision: **discarded by the keep rule** because hallucination rose 0.10 → 0.20. **Note:** this experiment does not touch retrieval, the prompt, or generation; the answer path is identical to the baseline. q37 changed from a refusal to a hallucination on identical code and settings, so the rise is run-to-run generation noise, not an effect of this change. Across the runs of the baseline configuration in this session (temperature 0), q37 hallucinated in 2 of 5 (once in the random-ID pair, and here), and q37 is also the only question that moved hallucination in Exp 1 and Exp 2. The faithfulness metric itself is measurement-only and could be adopted separately.

```
{"total_questions": 40, "answerable_questions": 30, "unanswerable_questions": 10,
 "recall_at_1": 0.6333, "recall_at_3": 0.9667, "recall_at_5": 1.0, "mrr": 0.8028,
 "hallucination_rate": 0.2, "answerable_wrongly_refused": 4,
 "faithfulness_answers_checked": 28, "faithfulness_answers_fully_supported": 21,
 "faithfulness_sentences_scored": 95, "faithfulness_sentences_supported": 62,
 "faithfulness_sentences_uncited": 0, "faithfulness_sentence_rate": 0.6526}
```

Checks: pytest 88 passed; offline check (online audit) 8 passed, 0 failed; deletion check 15 passed, 0 failed.
