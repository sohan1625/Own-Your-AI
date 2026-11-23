# Own Your Intelligence

A private, offline "chat with your PDFs" assistant. Upload documents, ask questions, and get answers that come only from your documents, each with a citation to the file and page it came from. Everything runs on your own machine: the language model, the embeddings, the database, and the web UI. Nothing you upload or ask leaves the computer.

The retrieval pipeline (chunking, BM25, vector search, fusion, citation checking) is written from scratch rather than assembled from a framework, so every stage can be explained, tested, and measured. A LangChain version of the retrieval step is included as a switch, so the two can be compared on the same chunks.

## Features

- **Upload PDFs** and choose which documents each question searches.
- **Grounded answers with citations** (file and page). If the documents don't contain the answer, the app says "not found in your documents" instead of guessing.
- **Hybrid search**: keyword search (BM25) and vector search (nomic-embed-text in pgvector), merged with reciprocal rank fusion.
- **Chat history** saved in PostgreSQL, with follow-up questions ("Explain those.") resolved from the previous question.
- **Two-level answer cache**: an exact repeat of a question returns the stored answer in about 0.1 s. If there's no exact match, a **semantic cache** looks for a near-identical earlier question about the same documents (cosine similarity ≥ 0.972) and returns its answer in about 0.25 s instead of about 5–7 s. Refusals are never cached, follow-ups never use the cache, and deleting a document clears every answer that used it.
- **Deterministic answers**: temperature 0 (greedy decoding), and answers are capped at 256 tokens so a looping model can't hang a request.
- **Delete a document** and its chunks, embeddings, cached answers, and stored file are removed.
- **Fully local**: the code refuses to call any model server other than one on 127.0.0.1.

Optional, off by default (one setting in `config.toml` each):

- **Semantic chunking**: split pages where the topic shifts between consecutive sentences, instead of fixed-size windows. Page boundaries and character spans are kept, so citations still work.
- **LangChain retrieval engine**: LangChain's `BM25Retriever` plus a pgvector retriever over the same database, fused by `EnsembleRetriever`.

Evaluation tooling:

- **Repeatable eval runs**: the eval unloads the model first, which clears Ollama's prompt cache. Without that, identical temperature-0 runs gave different answers.
- **Faithfulness metric**: each answer sentence is checked against the chunk it cites (word overlap).

## How it works

```
Upload PDF ─► extract text per page ─► split into overlapping chunks ─► embed chunks ─► PostgreSQL + pgvector

Question ─► keyword search (BM25) ──┐
         └► vector search (pgvector) ┴► fuse rankings (RRF, k = 60) ─► top 4 chunks ─► local LLM (mistral)
                                                                                         │
            answer + citations ◄── keep only citations to chunks that were provided ◄────┘
```

1. **Ingest**: text is extracted page by page with `pypdf`; repeated headers, footers, and table-of-contents pages are removed.
2. **Chunk**: about 400 tokens per chunk with a 50-token overlap; each chunk keeps its file, page, and character span. With `strategy = "semantic"`, each sentence is embedded instead, and a new chunk starts where similarity between consecutive sentences drops below the 10th percentile.
3. **Index**: chunks are stored in PostgreSQL. Each chunk gets a 768-dimension embedding from `nomic-embed-text`, stored with pgvector. BM25 statistics are computed in Python.
4. **Retrieve**: the question runs through BM25 and vector search over the selected documents; the two rankings are fused with RRF. For follow-up questions, the previous question is added to the search.
5. **Not-found check**: if the best vector score is below a calibrated threshold (0.605), the app answers "not found in your documents" without calling the model.
6. **Answer**: the top 4 chunks, labelled with IDs, go to `mistral` (temperature 0, at most 256 tokens) with instructions to answer only from them and cite chunk IDs.
7. **Cite**: chunk IDs in the answer are mapped back to file and page. Citations to chunks that were not provided are dropped.
8. **Cache**: before step 4, the question is looked up in the answer cache, first by exact text, then by embedding similarity within the same set of documents.

More detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/RAG_PIPELINE.md](docs/RAG_PIPELINE.md).

## Results

Measured on a 40-question eval set (30 answerable, 10 unanswerable) over a 47-page Java PDF. Full details, including every miss and its cause, are in [docs/EVAL.md](docs/EVAL.md) and [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

**Retrieval** (is the right passage in the top k?)

| Search | Recall@1 | Recall@3 | Recall@5 | MRR |
|--------|----------|----------|----------|-----|
| Keyword (BM25) | 0.63 | 0.87 | 1.00 | 0.77 |
| Vector | 0.67 | 0.90 | 1.00 | 0.79 |
| **Hybrid (RRF)** | 0.63 | **0.97** | **1.00** | **0.80** |

**Answers** (hybrid, with the not-found threshold, temperature 0)

| Metric | Target | Result |
|--------|--------|--------|
| Recall@5 | ≥ 0.80 | 1.00 |
| Hallucination on unanswerable questions | < 10% | 20% (2 of 10) — misses target |
| Answerable questions wrongly refused | — | 4 of 30 |
| Faithfulness (sentences supported by their cited chunk, word-overlap check) | ≥ 0.90 | 0.74 (61 of 82); 21 of 28 answers fully supported |

The not-found threshold cannot fully separate in-scope from out-of-scope questions on this set; the analysis and the experiments behind the current settings are in [docs/EVAL.md](docs/EVAL.md). The faithfulness check is a word-overlap proxy, not entailment: it misses correct paraphrases and can't catch a flipped negation (limits listed in [docs/BENCHMARKS.md](docs/BENCHMARKS.md)).

**Experiments** (same eval set; each compared with the temperature-0 baseline, Recall@3 0.97 and MRR 0.80)

| Experiment | Result | Status |
|------------|--------|--------|
| Temperature 0 | Same retrieval and answer metrics; answers still varied between runs until the eval cleared Ollama's prompt cache (a full multi-run check of that fix is not done yet) | On |
| Semantic chunking (92 chunks instead of 44) | Recall@1 0.63 → 0.77, MRR 0.80 → 0.87, wrongly refused 4 → 2 | Off by default |
| LangChain retrieval | Recall@3 0.97 → 1.00, MRR 0.80 → 0.81 | Off by default |
| Semantic cache | 3 of 20 reworded questions served from cache, 0 of 20 look-alike questions wrongly matched | On |

Semantic chunking and LangChain retrieval were set aside under a keep rule fixed in advance, because hallucination went from 1 to 2 of 10. In every case the extra one was the same question, which later turned out to flip between runs on unchanged code because of Ollama's prompt cache. Those decisions have not been re-measured with the repeatable eval. The semantic-cache threshold was chosen on [eval/cache_pairs.jsonl](eval/cache_pairs.jsonl) (20 reworded and 20 look-alike question pairs); its hit rate is low because the embedding model scores some different questions ("throw" vs "throws", 0.966) above most true rewordings. Details for every experiment are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

**Speed** (Ryzen 5 7235HS, 24 GB RAM, RTX 3050 Laptop 6 GB, mistral 7B 4-bit)

| Metric | p50 | p95 |
|--------|-----|-----|
| Retrieval | 0.11 s | 0.13 s |
| Answer (generated) | 7.0 s | 14.1 s |
| Answer (exact cache hit) | 0.11–0.13 s | 0.15–0.30 s |
| Answer (semantic cache hit) | 0.25 s | 0.27 s |
| Indexing | 7.6 s per 100 pages | |

Peak memory: about 9.6 GB RAM and 5.5 GB of GPU memory, almost all of it the two models.

## Tech stack

| Part | Choice |
|------|--------|
| Backend | Python 3.11+, FastAPI, Uvicorn |
| Database | PostgreSQL 16 with pgvector, in Docker |
| Language model | `mistral` 7B (4-bit) via [Ollama](https://ollama.com) |
| Embeddings | `nomic-embed-text` via Ollama |
| PDF parsing | `pypdf` |
| UI | A single HTML page (`static/index.html`), no build step |
| Tests | pytest |

Why each was chosen: [docs/DECISIONS.md](docs/DECISIONS.md).

## Setup

Developed and tested on Windows 11. Linux and macOS should work but have not been tested.

### Prerequisites

- **Python 3.11+**
- **Docker Desktop** (runs PostgreSQL with pgvector)
- **Ollama** ([ollama.com](https://ollama.com))
- An **NVIDIA GPU with 6 GB VRAM** is recommended. About 10 GB of free RAM is needed for the models and database. CPU-only has not been tested.

### 1. Install

```powershell
git clone <this-repo-url>
cd <repo-folder>

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item .env.example .env
```

`.env` holds the database connection string. The defaults match `docker-compose.yml`.

### 2. Start the database

```powershell
docker compose up -d
docker compose ps    # the rag-pgvector container should be running on port 5432
```

The app creates its tables and the pgvector extension on startup.

### 3. Download the models (one time)

```powershell
ollama pull mistral
ollama pull nomic-embed-text
```

This is the only step that uses the network. After it, the app runs fully offline.

### 4. Run the app

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000, upload a PDF with the **+** button, and ask a question. Use **New chat** in the sidebar to start a fresh conversation; earlier chats are listed under **Chats**.

### Configuration

All tunable settings are in [config.toml](config.toml):

| Setting | Default | What it does |
|---------|---------|--------------|
| `[retrieval] method` | `"hybrid"` | `"hybrid"` or `"keyword"` (BM25 only) |
| `[retrieval] engine` | `"custom"` | `"custom"` (this repo's code) or `"langchain"` (LangChain retrievers, hybrid only) |
| `[chunking] chunk_size`, `overlap` | 400, 50 | Chunk size and overlap in tokens |
| `[chunking] strategy` | `"fixed"` | `"fixed"` windows or `"semantic"` (split at topic shifts) |
| `[chunking] semantic_percentile` | 10 | Semantic only: lower gives fewer, larger chunks (chosen on `eval/calibration.jsonl`) |
| `[fusion] rrf_k` | 60 | RRF constant |
| `[generation] top_k` | 4 | Chunks sent to the model |
| `[generation] temperature` | 0 | Sampling temperature (0 = greedy, repeatable) |
| `[generation] max_tokens` | 256 | Longest answer, in tokens |
| `[thresholds] not_found_threshold` | 0.605 | Below this vector score, answer "not found" |
| `[cache] enabled` | `true` | Answer cache on or off |
| `[cache] semantic` | `true` | Second, similarity-based lookup after an exact miss |
| `[cache] semantic_threshold` | 0.972 | Minimum cosine similarity for a semantic hit |
| `[ollama] model` / `[embedding] model` | `mistral` / `nomic-embed-text` | Local models |

Changing `[chunking] strategy` applies to documents uploaded afterwards; delete and re-upload a document to re-chunk it.

Keep the Ollama host as `http://127.0.0.1:11434`. On Windows, `localhost` adds about 2 seconds to every model request (see [docs/BENCHMARKS.md](docs/BENCHMARKS.md)).

## Testing

```powershell
python -m pytest -q
```

There are 116 tests. They need the database running (`docker compose up -d`) but **not** Ollama: model calls are mocked, and embeddings use a small deterministic fake (`tests/conftest.py`). Tests that touch the database create a temporary schema and drop it afterwards, so your documents and chats are never touched.

| Test file | Covers |
|-----------|--------|
| `test_ingest.py` | PDF text extraction, header/footer and table-of-contents removal, file hashing, skipping unchanged files |
| `test_chunk.py` | Chunk overlap, short and empty pages, chunks never crossing a page boundary |
| `test_semantic_chunk.py` | Sentence spans, topic-shift boundaries, page boundaries, size cap, exact character spans |
| `test_index.py`, `test_retrieve.py` | BM25 ranking and keyword retrieval |
| `test_vector.py` | Cosine similarity and ranking, embedding prefixes, localhost-only guard |
| `test_lc_retrieve.py` | LangChain engine: pgvector wrapper, fusion order matches RRF, `/ask` engine switch |
| `test_answer.py` | Prompt building, temperature and answer-length options, model unload, not-found threshold, answer normalization, citation fallback, reworded refusals |
| `test_cite.py` | Mapping chunk IDs to file, page, and span; dropping invalid citations |
| `test_faithfulness.py` | Sentence splitting, citation inheritance, overlap scoring, a documented negation blind spot |
| `test_api.py` | Upload, ask, delete, persistence across restarts, follow-up search |
| `test_cache.py` | Exact and semantic hits, misses below the threshold, same-document-set rule, refusals not cached, follow-ups skip the cache, deletion clears cached answers |
| `test_hybrid.py` | pgvector storage and search, RRF fusion, fallback to keyword search, embedding backfill |
| `test_eval.py` | Eval harness metrics, wrongly-refused count, stable chunk IDs, model unload |

If pytest crashes at the end with `PermissionError` on its temp folder (a Windows issue with a stale `pytest-current` link), add `--basetemp=.pytest_tmp`.

### Running the evaluation

The eval set is in `eval/questions.jsonl`. It was written for a private PDF that is not included in this repository, so to reproduce the numbers you need your own documents and questions in the same format.

1. Put the PDFs in `data/` (the folder is git-ignored). The app never reads `data/`; its own uploads go to `storage/uploads/`.
2. Run:

```powershell
python -m scripts.eval --retriever keyword
python -m scripts.eval --retriever vector
python -m scripts.eval --retriever hybrid --with-answers   # also generates answers: hallucination, wrongly refused, faithfulness
python -m scripts.eval --retriever langchain               # the LangChain engine on the same chunks
python -m scripts.calibrate                                # raw scores for choosing the not-found threshold
python -m scripts.calibrate_chunking                       # choose the semantic-chunking percentile (calibration set only)
python -m scripts.calibrate_cache                          # choose the semantic-cache threshold (eval/cache_pairs.jsonl)
python -m scripts.measure_cache                            # semantic-cache hit rates and latency through /ask
python -m scripts.demo "Your question here"                # one question end to end
```

Results are written to `eval/results_<retriever>.json` (use `--output` to write elsewhere). `measure_cache` uploads `data/Core_Java_Complete_Answers.pdf`, and `eval/cache_pairs.jsonl` is written about that PDF, so with your own documents you need your own pairs. To try semantic chunking, pass a copy of `config.toml` with `strategy = "semantic"` via `--config`.

With `--with-answers`, the eval first unloads the answer model, which clears Ollama's prompt cache. That cache reuses parts of earlier prompts, and how much it reuses slightly changes the result, so without the unload two identical runs can give different answers even at temperature 0. `--warm-model` skips the unload. The eval also gives chunks stable IDs instead of random ones, because the IDs appear in the prompt.

Thresholds are never tuned on `eval/questions.jsonl`: the not-found threshold and chunking percentile use `eval/calibration.jsonl`, and the cache threshold uses `eval/cache_pairs.jsonl`.

## Project structure

```
src/
  api.py        FastAPI app: upload, ask, documents, chats
  ingest.py     PDF text extraction and cleanup
  chunk.py      Fixed and semantic chunking with source spans
  index.py      BM25 index and embedding calls to Ollama
  retrieve.py   Keyword and vector retrieval
  fuse.py       Reciprocal rank fusion
  lc_retrieve.py  Optional LangChain retrieval engine
  answer.py     Prompt, not-found threshold, local model call
  cite.py       Citation resolution
  faithfulness.py  Sentence-level citation support check (eval metric)
  cache.py      Exact and semantic answer cache
  db.py         PostgreSQL and pgvector storage
static/
  index.html    The web UI
scripts/        eval.py, calibrate*.py, measure_cache.py, demo.py, deletion_check.py, offline_check.py
eval/           Eval, calibration, and cache-pair question sets, results
storage/        Uploaded PDFs and check results (created at runtime, git-ignored)
tests/          pytest suite
docs/           Requirements, design, decisions, eval, benchmarks, privacy, tasks
config.toml     All tunable settings
docker-compose.yml
```

## Privacy

- Documents, chunks, embeddings, questions, answers, and chat history are stored only in the local PostgreSQL database (a Docker volume) and `storage/uploads/`.
- The only network use at runtime is HTTP to Ollama on 127.0.0.1; the code rejects any other model host. (Ollama's own tray program checks ollama.com for updates; it never sees your documents.)
- Deleting a document removes its file, chunks, embeddings, and cached answers. Chat history is kept.

Both claims are checked by scripts that run the real app and models in a temporary database schema:

```powershell
python -m scripts.deletion_check                  # 15 checks that a deleted document is gone everywhere
python -m scripts.offline_check                   # run with the network off: upload, ask, follow-up, cache, delete
python -m scripts.offline_check --with-network    # online audit: no outbound connections from the app or Ollama server
```

See [docs/PRIVACY.md](docs/PRIVACY.md).

## Status and limitations

Working end to end, with results measured. Not done yet:

- No way to delete a chat yet; chats keep answers from deleted documents
- Hallucination on unanswerable questions is 20%, above the 10% target
- Faithfulness is a word-overlap proxy; an entailment-based check and a citation-accuracy metric are not built
- The semantic cache serves few rewordings (3 of 20) because its threshold must stay above look-alike questions, and the threshold was chosen and measured on the same 40 pairs
- The semantic-chunking and LangChain decisions have not been re-measured with the repeatable eval
- Reranker experiment
- Full-precision vs. quantized model comparison
- Re-uploading a changed PDF adds a second document instead of replacing the first
- Scanned (image-only) PDFs are not supported: there is no OCR

The plan and progress are in [docs/TASKS.md](docs/TASKS.md).

## Documentation

- [docs/PRD.md](docs/PRD.md): goals, scope, requirements
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): system design
- [docs/RAG_PIPELINE.md](docs/RAG_PIPELINE.md): each pipeline stage and its parameters
- [docs/DECISIONS.md](docs/DECISIONS.md): technical choices and why
- [docs/STACK_OPTIONS.md](docs/STACK_OPTIONS.md): options considered for the stack
- [docs/EVAL.md](docs/EVAL.md): eval set, metrics, results, misses
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md): speed, memory, and the experiments (temperature 0, semantic chunking, LangChain, faithfulness, semantic cache)
- [docs/PRIVACY.md](docs/PRIVACY.md): what is stored and how it is deleted
- [docs/TASKS.md](docs/TASKS.md): milestones and progress
- [docs/AI_CONTEXT.md](docs/AI_CONTEXT.md): rules for AI coding assistants working on this repo
