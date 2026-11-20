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
