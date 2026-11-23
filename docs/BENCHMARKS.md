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

### Summary

All five features are combined on `exp/combined`, each behind a config switch. Decisions follow the keep rule in each section.

| Exp | Feature | Switch (value on `exp/combined`) | Decision | Branch |
|---|---|---|---|---|
| 0 | Temperature 0 + repeatable eval (model unloaded before each run) | `[generation] temperature = 0`; eval `--warm-model` to skip the unload | kept | `exp/temperature-zero` |
| 1 | Semantic chunking | `[chunking] strategy = "semantic"` (default since 2026-11-24) | discarded*, then turned on | `exp/semantic-chunking` |
| 2 | LangChain retrieval (BM25Retriever + pgvector, EnsembleRetriever) | `[retrieval] engine = "langchain"` (default since 2026-11-24) | discarded*, then turned on | `exp/langchain-retrieval` |
| 3 | Faithfulness metric (token overlap per sentence) | always reported by `scripts/eval.py --with-answers` | discarded* (measurement only) | `exp/faithfulness` |
| 4 | Semantic answer cache | `[cache] semantic = true`, `semantic_threshold = 0.972` | kept | `exp/semantic-cache` |

\* Exp 1–3 were discarded only because hallucination rose from 1 to 2 of 10, in each case through q37. The Exp 0 follow-up shows q37 flips between runs on unchanged code because of Ollama's prompt cache, so these decisions rest on noise and have not been re-checked with the eval fix.

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

### Exp 0 follow-up: why q37 flipped, and an eval fix (2026-11-23)

Five runs of the unchanged Exp 0 eval (instrumented, no code changed) were identical in every metric and every answer: Recall@1 0.6333, Recall@3 0.9667, Recall@5 1.0000, MRR 0.8028, hallucination 0.20 (q32, q37), 4 answerable wrongly refused. q37 hallucinated in all five with the same uncited answer (the fallback citation is chunk `48f89474`).

Cause: **Ollama's prompt cache**, not sampling, retrieval or the threshold.
- Sampling: the runner log shows `temp = 0.000` and a `greedy` sampler for every request, so no seed is involved.
- Retrieval: chunk embeddings, query embeddings, BM25 and vector top-5 scores, and the 4 chunks sent to mistral were identical in all runs; no top-5 near-ties (< 1e-6).
- Threshold: q37's top vector score is 0.6853, 0.080 above 0.605, identical in every run.
- Prompt cache: the llama-server runner reuses the longest cached prefix of a prompt, from the previous request or from a RAM prompt cache that keeps earlier prompts (including those from earlier eval runs), and recomputes the rest. q37's identical 1996-token prompt was served with 130, 1540 and 1995 tokens reused in different runs, and produced 528-, 530-, 512- and 391-token answers. Batch boundaries change the floating-point result enough to change greedy output. None of the long q37 answers cites a chunk, so the outcome depends on whether the answer happens to contain "not found in your documents" (→ refusal) or not (→ fallback citation → hallucination). The five identical runs above were identical only because every prompt was already fully cached.
- `cache_prompt: false` (top-level or in `options`) is not honored by Ollama: 1995 tokens were still reused.
- With the model unloaded first (`keep_alive: 0`), q37's prompt was computed from token 0 in 512-token batches and gave the same 528-token answer twice.

Fix (eval only, on `exp/temperature-zero`): `scripts/eval.py` unloads the answer model before the first question (`src.answer.unload_model`, a `keep_alive: 0` request to the local Ollama), so each run starts from an empty prompt cache and sends the same sequence of requests. `--warm-model` skips it. The app is unchanged. A five-run confirmation of the fixed eval was started and stopped before any run finished, so reproducibility of full runs with the fix is not yet measured. Exp 1–3 decisions were not re-checked.

### Exp 1: semantic chunking (discarded)

- Branch: `exp/semantic-chunking` (from `exp/temperature-zero`)
- Settings: `[chunking] strategy = "semantic"` for the run (the committed default stays `"fixed"`), `semantic_percentile = 10`. Pages are split into sentences (`.`/`!`/`?` + whitespace, or a blank line); each sentence is embedded with nomic-embed-text (`search_document:` prefix); a new chunk starts where the cosine similarity of two consecutive sentences is below the 10th percentile of all consecutive-sentence similarities in the corpus. Chunks never cross pages and are capped at `chunk_size` (533 words). `start_char`/`end_char` are exact offsets into the page text. 520 sentences → 92 chunks (fixed: 44). Everything else as the Exp 0 baseline, including the not-found threshold 0.605 (not recalibrated).
- Percentile chosen on `eval/calibration.jsonl` only (`python -m scripts.calibrate_chunking`, 15 answerable questions, gold = page). Selection rule: highest Recall@3, then highest MRR.

| Calibration | Chunks | R@1 | R@3 | R@5 | MRR | Answerable below 0.605 | Unanswerable at/above 0.605 |
|---|---|---|---|---|---|---|---|
| fixed (reference) | 44 | 0.8000 | 1.0000 | 1.0000 | 0.8889 | 2 | 1 |
| semantic p5 | 68 | 0.6667 | 0.9333 | 0.9333 | 0.8095 | 2 | 2 |
| **semantic p10** | 92 | 0.8667 | 0.9333 | 0.9333 | 0.9048 | 1 | 2 |
| semantic p15 | 116 | 0.8000 | 0.9333 | 0.9333 | 0.8597 | 1 | 2 |
| semantic p25 | 163 | 0.8667 | 0.8667 | 0.8667 | 0.8767 | 1 | 2 |
| semantic p35 | 211 | 0.8667 | 0.9333 | 0.9333 | 0.9000 | 0 | 2 |
| semantic p50 | 282 | 0.7333 | 0.9333 | 1.0000 | 0.8467 | 0 | 2 |
| semantic p65 | 353 | 0.8000 | 0.9333 | 0.9333 | 0.8595 | 0 | 3 |
| semantic p80 | 424 | 0.7333 | 0.8667 | 0.8667 | 0.8044 | 0 | 4 |

No semantic setting reached the fixed strategy's calibration Recall@3.

| Metric (eval/questions.jsonl) | Baseline (Exp 0) | Exp 1 |
|--------|------|------|
| Recall@1 | 0.6333 | 0.7667 |
| Recall@3 | 0.9667 | 0.9667 |
| Recall@5 | 1.0000 | 1.0000 |
| MRR | 0.8028 | 0.8694 |
| Hallucination (of 10) | 0.10 (q32) | **0.20 (q32, q37)** |
| Answerable wrongly refused (of 30) | 4 (q03, q11, q13, q30) | 2 (q03, q13) |

Decision: **discarded**: hallucination rose (q37, PostgreSQL replication, answered from general knowledge with a citation). Retrieval improved (Recall@1 +0.13, MRR +0.07) and two fewer answerable questions were refused, but the keep rule does not allow a hallucination increase. q37 also flipped between runs in Exp 0 before chunk IDs were made stable, so this is one question near the threshold, not a clear effect. The threshold was not recalibrated for the smaller chunks, which shift top vector scores (calibration: unanswerable at/above threshold 1 → 2).

```
{"total_questions": 40, "answerable_questions": 30, "unanswerable_questions": 10,
 "recall_at_1": 0.7667, "recall_at_3": 0.9667, "recall_at_5": 1.0, "mrr": 0.8694,
 "hallucination_rate": 0.2, "answerable_wrongly_refused": 2}
```

Checks: pytest 88 passed; offline check (online audit) 8 passed, 0 failed; deletion check 15 passed, 0 failed.

### Exp 2: LangChain retrieval (discarded)

- Branch: `exp/langchain-retrieval` (from `exp/temperature-zero`)
- Settings: `[retrieval] engine = "langchain"` for the run (committed default `"custom"`). `src/lc_retrieve.py`: `BM25Retriever` (langchain-community 0.4.2, rank-bm25 0.2.2) with the project tokenizer (`src.index._tokenize`), k = 20; `PgVectorRetriever`, a `BaseRetriever` that embeds the query with nomic-embed-text and calls `db.vector_search`, k = 20; fused by `EnsembleRetriever` (langchain-classic 1.1.0, langchain-core 1.8.4) with weights [0.5, 0.5], c = 60, `id_key = "chunk_id"`. `/ask` uses the same switch.
- Eval: `python scripts/eval.py --retriever langchain --with-answers` loads the eval chunks and their embeddings into a throwaway PostgreSQL schema (dropped afterwards; verified no `eval_*` schema was left), so `db.vector_search` serves the vectors. Both engines ran on the same chunks (fixed chunking with stable IDs, 44 chunks) in this branch.
- Fusion difference: `EnsembleRetriever` uses **weighted** RRF, score(d) = Σ wᵢ / (c + rankᵢ(d)). With equal weights and c = 60 that is half our RRF (k = 60) and gives the same order for the same input lists (unit-tested). The lists themselves differ: `rank_bm25.BM25Okapi` uses a different IDF (epsilon floor) and returns the top k even at score 0, while our BM25 drops zero-score chunks.

| Metric | Baseline (Exp 0) | Custom engine (rerun here) | LangChain engine |
|--------|------|------|------|
| Recall@1 | 0.6333 | 0.6333 | 0.6333 |
| Recall@3 | 0.9667 | 0.9667 | 1.0000 |
| Recall@5 | 1.0000 | 1.0000 | 1.0000 |
| MRR | 0.8028 | 0.8028 | 0.8056 |
| Hallucination (of 10) | 0.10 (q32) | 0.10 (q32) | **0.20 (q32, q37)** |
| Answerable wrongly refused (of 30) | 4 | 4 (q03, q11, q13, q30) | 4 (same) |

The only retrieval difference is q14 (gold rank 4 → 3). q37 (PostgreSQL replication) has the same top vector score in both engines and passes the 0.605 threshold either way; with the LangChain ordering of the top 4 chunks mistral answered from general knowledge instead of refusing.

Decision: **discarded**: hallucination rose (0.10 → 0.20). Retrieval gained slightly (Recall@3 +0.033, MRR +0.003), on one question.

```
{"total_questions": 40, "answerable_questions": 30, "unanswerable_questions": 10,
 "recall_at_1": 0.6333, "recall_at_3": 1.0, "recall_at_5": 1.0, "mrr": 0.8056,
 "hallucination_rate": 0.2, "answerable_wrongly_refused": 4}
```

Checks: pytest 86 passed; offline check (online audit) 8 passed, 0 failed; deletion check 15 passed, 0 failed.

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

### Exp 4: semantic cache (kept)

- Branch: `exp/semantic-cache` (from `exp/temperature-zero` before the follow-up eval fix)
- Settings: `[cache] semantic = true`, `semantic_threshold = 0.972`. Exact match stays the first check. Only on an exact miss, the question is embedded (nomic-embed-text, `search_query:` prefix) and compared by cosine with cached question embeddings for the **same document set** (new `doc_set_key` column: hash of the sorted document IDs and their sha256 hashes). The closest entry is served only if its similarity is ≥ 0.972. Unchanged rules: refusals are never stored, follow-ups with conversation history never use the cache, and deleting a document deletes every entry that used it (semantic entries included; unit-tested). `/ask` now also returns `cache_match` (`"exact"`, `"semantic"` or null).
- Test pairs: `eval/cache_pairs.jsonl`, 20 paraphrase pairs and 20 near-miss pairs written from the eval PDF.
- Threshold chosen on those pairs only (`python -m scripts.calibrate_cache`), with a rule fixed in advance: simulate a cache holding every `question_a`; for each `question_b` take the closest cached question; the threshold is the lowest value with zero near-miss false hits (a near-miss counts as a false hit if it matches *any* cached question) plus a 0.005 margin, rounded up. Closest near-miss similarity 0.9661 (n08: "throw" vs "throws") → threshold 0.972. Three near-misses were closest to a *different* pair's question (n11 → p05 at 0.9055, n07 → p07, n14 → p04).

The two groups overlap: paraphrase similarities range 0.7895–0.9855 and near-miss similarities 0.7346–0.9661. 15 of 20 paraphrases score below the most similar near-miss, so a threshold with zero near-miss false hits can only catch the closest paraphrases.

| Measurement | Result |
|---|---|
| Calibration (similarities only): paraphrase hits at 0.972 | 4 / 20 (p20, p09, p03, p12) |
| Calibration: near-miss false hits at 0.972 | 0 / 20 |
| End to end via `POST /ask` (`python -m scripts.measure_cache`): paraphrase hit rate | **3 / 20** (p09, p12, p20); 3 / 17 counting only pairs whose `question_a` was answered (p03, p06, p08 were refused, so not cached) |
| End to end: near-miss false-hit rate | **0 / 20** (18 of the 20 `question_a` were cached; n10, n13 refused) |
| Semantic-hit latency (3 questions × 5 asks) | p50 0.246 s, p95 0.27 s |
| Exact-hit latency (35 asks) | p50 0.111 s, p95 0.15 s |
| Cache miss, answer generated (37 asks) | p50 5.3 s |

Caveats: the threshold was chosen on the same 40 pairs it is reported on, so zero false hits on new near-misses is not guaranteed, and the margin above the closest near-miss is only 0.006. The semantic hit p95 comes from 15 samples on 3 questions.

Keep rule: the eval (`scripts/eval.py`) does not use the answer cache, so Recall@3, MRR, hallucination and wrongly refused are unchanged by construction (the eval was not re-run). Near-miss false-hit rate is 0 / 20 and the paraphrase hit rate is reported. Decision: **kept**.

Checks: pytest 89 passed; offline check (online audit) 8 passed, 0 failed (exact repeat still served from cache, 0.14 s); deletion check 15 passed, 0 failed.

### Combined branch (`exp/combined`, 2026-11-24)

All five branches merged, with the settings in the summary table plus `[generation] max_tokens = 256` (Ollama `num_predict`). The cap was added after a combined-branch eval hung: with greedy decoding, mistral looped on q37 for 14 minutes (over 10,500 tokens) and Ollama kept going via context shift, because no limit was sent. The longest normal answer in the eval is 112 tokens (median 65).

One run of `python scripts/eval.py --retriever hybrid --with-answers` (cold start, 3 min 16 s):

| Metric | Result |
|---|---|
| Recall@1 / @3 / @5 | 0.6333 / 0.9667 / 1.0000 |
| MRR | 0.8028 |
| Hallucination (of 10) | 0.20 (q32, q37) |
| Answerable wrongly refused (of 30) | 4 (q03, q11, q13, q30) |
| Faithfulness (token overlap) | 61 / 82 sentences supported (0.7439); 21 of 28 answers fully supported |

Single run; repeatability of full runs with the cold start was not re-measured. Retrieval-only checks with the switches on reproduce the branch results: `--retriever langchain` R@3 1.0000, MRR 0.8056; `[chunking] strategy = "semantic"` R@3 0.9667, MRR 0.8694.

```
{"total_questions": 40, "answerable_questions": 30, "unanswerable_questions": 10,
 "recall_at_1": 0.6333, "recall_at_3": 0.9667, "recall_at_5": 1.0, "mrr": 0.8028,
 "hallucination_rate": 0.2, "answerable_wrongly_refused": 4,
 "faithfulness_answers_checked": 28, "faithfulness_answers_fully_supported": 21,
 "faithfulness_sentences_scored": 82, "faithfulness_sentences_supported": 61,
 "faithfulness_sentences_uncited": 0, "faithfulness_sentence_rate": 0.7439}
```

Checks: pytest 116 passed; offline check (online audit) 8 passed, 0 failed; deletion check 15 passed, 0 failed (run before the cap; the cap only adds `num_predict` to the local request).

### Semantic chunking and LangChain retrieval on by default (2026-11-24)

Both switched on: `[chunking] strategy = "semantic"`, `[retrieval] engine = "langchain"`. Fixes made while turning them on:

- Both features called the embedding model through their own imports, so the tests' fake embeddings did not reach them and an upload with semantic chunking failed outright when Ollama was down. Uploads and LangChain queries now embed through the app's embedding function (`src.api`), and semantic chunking falls back to fixed windows if embedding fails (same degradation as before: keyword search still works).
- Tests that check the from-scratch engine or fixed chunking now pin those settings instead of reading `config.toml`. With every real Ollama call blocked, all 121 tests pass, so they still need no model server.
- New `python -m scripts.rechunk` rebuilds the chunks and embeddings of already-uploaded documents with the current settings (chat history kept, their cached answers cleared). The stored eval PDF was re-chunked: 44 → 92 chunks, all embedded.

One run of `python scripts/eval.py --retriever langchain --with-answers` with semantic chunks (cold start, 4 min 8 s), next to the default-config run of the combined branch:

| Metric | Fixed + custom engine | Semantic + LangChain |
|---|---|---|
| Recall@1 / @3 / @5 | 0.6333 / 0.9667 / 1.0000 | **0.8333** / 0.9667 / 1.0000 |
| MRR | 0.8028 | **0.9011** |
| Hallucination (of 10) | 0.20 (q32, q37) | 0.20 (q32, q37) |
| Answerable wrongly refused (of 30) | 4 (q03, q11, q13, q30) | **2** (q03, q13) |
| Faithfulness, all non-refusal answers | 61 / 82 (0.7439) | 35 / 59 (0.5932) |
| Faithfulness, answerable questions | 47 / 54 (0.87) | 35 / 47 (0.745) |
| Answers fully supported | 21 / 28 | 21 / 30 |

Against the keep rule (Recall@3 and MRR not lower, hallucination and wrongly refused not higher) the combination passes. Faithfulness drops on answerable questions: with 92 smaller chunks an answer often draws on neighbouring chunks but cites one (q24 deadlock cites a chunk without the definition; q17, q20 similar), so citations are less specific. Single run each.

Checks: pytest 121 passed (also with real Ollama calls blocked); offline check (online audit) 8 passed, 0 failed, no outbound connections with LangChain loaded (in one run the follow-up was answered "not found"; rerun answered it, and a direct comparison gave the same retrieval score and answer with both engines, so that was answer variance); deletion check 15 passed, 0 failed.
