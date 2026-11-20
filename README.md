# Own Your Intelligence

A private, offline "chat with your PDFs" assistant. Upload documents, ask questions, and get answers that come only from your documents, each with a citation to the file and page it came from. Everything runs on your own machine: the language model, the embeddings, the database, and the web UI. Nothing you upload or ask leaves the computer.

The retrieval pipeline (chunking, BM25, vector search, fusion, citation checking) is written from scratch rather than assembled from a framework, so every stage can be explained, tested, and measured.

## Features

- **Upload PDFs** and choose which documents each question searches.
- **Grounded answers with citations** (file and page). If the documents don't contain the answer, the app says "not found in your documents" instead of guessing.
- **Hybrid search**: keyword search (BM25) and vector search (nomic-embed-text in pgvector), merged with reciprocal rank fusion.
- **Chat history** saved in PostgreSQL, with follow-up questions ("Explain those.") resolved from the previous question.
- **Answer cache**: repeating a question returns the stored answer in about 0.1 s instead of about 7 s.
- **Delete a document** and its chunks, embeddings, cached answers, and stored file are removed.
- **Fully local**: the code refuses to call any model server other than one on 127.0.0.1.

## How it works

```
Upload PDF ─► extract text per page ─► split into overlapping chunks ─► embed chunks ─► PostgreSQL + pgvector

Question ─► keyword search (BM25) ──┐
         └► vector search (pgvector) ┴► fuse rankings (RRF, k = 60) ─► top 4 chunks ─► local LLM (mistral)
                                                                                         │
            answer + citations ◄── keep only citations to chunks that were provided ◄────┘
```

1. **Ingest**: text is extracted page by page with `pypdf`; repeated headers, footers, and table-of-contents pages are removed.
2. **Chunk**: about 400 tokens per chunk with a 50-token overlap; each chunk keeps its file, page, and character span.
3. **Index**: chunks are stored in PostgreSQL. Each chunk gets a 768-dimension embedding from `nomic-embed-text`, stored with pgvector. BM25 statistics are computed in Python.
4. **Retrieve**: the question runs through BM25 and vector search over the selected documents; the two rankings are fused with RRF. For follow-up questions, the previous question is added to the search.
5. **Not-found check**: if the best vector score is below a calibrated threshold (0.605), the app answers "not found in your documents" without calling the model.
6. **Answer**: the top 4 chunks, labelled with IDs, go to `mistral` with instructions to answer only from them and cite chunk IDs.
7. **Cite**: chunk IDs in the answer are mapped back to file and page. Citations to chunks that were not provided are dropped.

More detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/RAG_PIPELINE.md](docs/RAG_PIPELINE.md).

## Results

Measured on a 40-question eval set (30 answerable, 10 unanswerable) over a 47-page Java PDF. Full details, including every miss and its cause, are in [docs/EVAL.md](docs/EVAL.md) and [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

**Retrieval** (is the right passage in the top k?)

| Search | Recall@1 | Recall@3 | Recall@5 | MRR |
|--------|----------|----------|----------|-----|
| Keyword (BM25) | 0.63 | 0.87 | 1.00 | 0.77 |
| Vector | 0.67 | 0.90 | 1.00 | 0.79 |
| **Hybrid (RRF)** | 0.63 | **0.97** | **1.00** | **0.80** |

**Answers** (hybrid, with the not-found threshold)

| Metric | Target | Result |
|--------|--------|--------|
| Recall@5 | ≥ 0.80 | 1.00 |
| Hallucination on unanswerable questions | < 10% | 20% (2 of 10) — misses target |
| Answerable questions wrongly refused | — | 4 of 30 |
| Faithfulness | ≥ 0.90 | Not measured yet |

The not-found threshold cannot fully separate in-scope from out-of-scope questions on this set; the analysis and the experiments behind the current settings are in [docs/EVAL.md](docs/EVAL.md).

**Speed** (Ryzen 5 7235HS, 24 GB RAM, RTX 3050 Laptop 6 GB, mistral 7B 4-bit)

| Metric | p50 | p95 |
|--------|-----|-----|
| Retrieval | 0.11 s | 0.13 s |
| Answer (generated) | 7.0 s | 14.1 s |
| Answer (from cache) | 0.13 s | 0.30 s |
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
| `[chunking] chunk_size`, `overlap` | 400, 50 | Chunk size and overlap in tokens |
| `[fusion] rrf_k` | 60 | RRF constant |
| `[generation] top_k` | 4 | Chunks sent to the model |
| `[thresholds] not_found_threshold` | 0.605 | Below this vector score, answer "not found" |
| `[cache] enabled` | `true` | Answer cache on or off |
| `[ollama] model` / `[embedding] model` | `mistral` / `nomic-embed-text` | Local models |

Keep the Ollama host as `http://127.0.0.1:11434`. On Windows, `localhost` adds about 2 seconds to every model request (see [docs/BENCHMARKS.md](docs/BENCHMARKS.md)).

## Testing

```powershell
python -m pytest -q
```

There are 76 tests. They need the database running (`docker compose up -d`) but **not** Ollama: model calls are mocked, and embeddings use a small deterministic fake (`tests/conftest.py`). Tests that touch the database create a temporary schema and drop it afterwards, so your documents and chats are never touched.

| Test file | Covers |
|-----------|--------|
| `test_ingest.py` | PDF text extraction, header/footer and table-of-contents removal, file hashing, skipping unchanged files |
| `test_chunk.py` | Chunk overlap, short and empty pages, chunks never crossing a page boundary |
| `test_index.py`, `test_retrieve.py` | BM25 ranking and keyword retrieval |
| `test_vector.py` | Cosine similarity and ranking, embedding prefixes, localhost-only guard |
| `test_answer.py` | Prompt building, not-found threshold, answer normalization, citation fallback, reworded refusals |
| `test_cite.py` | Mapping chunk IDs to file, page, and span; dropping invalid citations |
| `test_api.py` | Upload, ask, delete, persistence across restarts, follow-up search |
| `test_cache.py` | Cache hits, misses, refusals not cached, deletion clears cached answers |
| `test_hybrid.py` | pgvector storage and search, RRF fusion, fallback to keyword search, embedding backfill |
| `test_eval.py` | Eval harness metrics |

If pytest crashes at the end with `PermissionError` on its temp folder (a Windows issue with a stale `pytest-current` link), add `--basetemp=.pytest_tmp`.

### Running the evaluation

The eval set is in `eval/questions.jsonl`. It was written for a private PDF that is not included in this repository, so to reproduce the numbers you need your own documents and questions in the same format.

1. Put the PDFs in `data/` (the folder is git-ignored). The app never reads `data/`; its own uploads go to `storage/uploads/`.
2. Run:

```powershell
python -m scripts.eval --retriever keyword
python -m scripts.eval --retriever vector
python -m scripts.eval --retriever hybrid --with-answers   # also generates answers and measures hallucination
python -m scripts.calibrate                                # raw scores for choosing the not-found threshold
python -m scripts.demo "Your question here"                # one question end to end
```

Results are written to `eval/results_<retriever>.json`.

## Project structure

```
src/
  api.py        FastAPI app: upload, ask, documents, chats
  ingest.py     PDF text extraction and cleanup
  chunk.py      Overlapping chunks with source spans
  index.py      BM25 index and embedding calls to Ollama
  retrieve.py   Keyword and vector retrieval
  fuse.py       Reciprocal rank fusion
  answer.py     Prompt, not-found threshold, local model call
  cite.py       Citation resolution
  cache.py      Answer cache
  db.py         PostgreSQL and pgvector storage
static/
  index.html    The web UI
scripts/        eval.py, calibrate.py, demo.py, deletion_check.py, offline_check.py
eval/           Eval and calibration question sets, results
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
- Reranker experiment, and faithfulness and citation-accuracy metrics
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
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md): speed and memory
- [docs/PRIVACY.md](docs/PRIVACY.md): what is stored and how it is deleted
- [docs/TASKS.md](docs/TASKS.md): milestones and progress
- [docs/AI_CONTEXT.md](docs/AI_CONTEXT.md): rules for AI coding assistants working on this repo
