# Stack Options & Trade-Off Analysis

This document provides a comparative analysis for each pending technical stack decision identified in the project roadmap (`TASKS.md` and `DECISIONS.md`). For each area, 2 to 3 viable options are evaluated with specific pros and cons for this local, offline, verifiable document Q&A project, along with offline support verification, licenses, and actionable recommendations.

> **Note:** These are evaluated options and recommendations only. Final choices remain pending user selection and will be formally recorded in `DECISIONS.md`.

---

## 1. Database & Index Store

The system must store document metadata, extracted chunks (with source file, page, character spans, and parent context), inverted index statistics for custom BM25 keyword search, and dense vector embeddings.

### Options Considered

#### Option A: SQLite + `sqlite-vec` (or SQLite with NumPy vector search)
- **Overview:** Embedded, serverless, single-file relational database (`sqlite3` in standard library), with vector search provided either by the lightweight `sqlite-vec` extension or in-memory cosine similarity via NumPy for laptop corpus scale (<50,000 chunks).
- **Pros for this project:**
  - **Zero setup overhead:** No separate background service, Docker daemon, or port configuration required. Ensures the project achieves the target of `< 15 minutes` setup from clone.
  - **Minimal resource footprint:** Zero idle RAM consumption, leaving maximum RAM (24 GB) and VRAM (6 GB) for the local LLM and embedding models.
  - **Verifiable privacy & deletion:** All data resides in a single database file. The deletion test (confirming a document and all related chunks/postings are erased) and offline test are trivial to verify and demonstrate.
  - **Native Python integration:** Full transactional ACID support built directly into Python.
- **Cons for this project:**
  - Vector extensions on Windows require pre-built binaries or wheel packaging.
  - Not designed for high concurrent writes (not an issue for single-user local v1).
- **Offline Support:** 100% local and file-based. Zero network dependency.
- **License:** Public Domain (SQLite), Apache 2.0 / MIT (`sqlite-vec`).

#### Option B: PostgreSQL + `pgvector` (via Docker)
- **Overview:** Full relational database with the `pgvector` extension running in a local Docker container.
- **Pros for this project:**
  - Standard enterprise stack combining relational metadata, custom schema tables, and vector similarity indexes (HNSW / IVFFlat).
  - Strong resume signal for backend-oriented roles.
  - Readily scales to multi-user or client-server architecture in future versions.
- **Cons for this project:**
  - **Heavy setup friction on Windows:** Requires Docker Desktop and WSL2. Adds substantial setup friction for new users, risking the 15-minute setup goal.
  - **High idle RAM overhead:** Docker Desktop and WSL2 on Windows typically consume 2–4 GB of system RAM just idling, competing with the LLM and embedding processes.
  - Postgres built-in full-text search (`ts_rank`) is not BM25; custom BM25 logic must still be implemented on top of custom tables/postings.
- **Offline Support:** 100% local once the Docker image is pulled.
- **License:** PostgreSQL License (permissive MIT-like), pgvector (PostgreSQL License).

#### Option C: DuckDB (Embedded Vector & Columnar Store)
- **Overview:** In-process, serverless analytical database with columnar vector operations and full-text extensions.
- **Pros for this project:**
  - Embedded in-process with zero background server process (`pip install duckdb`).
  - High performance on columnar scans, vector distance calculations, and aggregation metrics.
- **Cons for this project:**
  - Point deletions and relational referential cascades are less mature than SQLite.
  - Less familiar to interviewers than SQLite or PostgreSQL.
- **Offline Support:** 100% local, embedded in-process.
- **License:** MIT License.

### Recommendation
**Option A: SQLite + `sqlite-vec` (or SQLite with NumPy vector search)**.  
For a local-first single-user assistant running on a Windows laptop, SQLite eliminates all Docker/WSL2 configuration overhead, starts in milliseconds, uses zero idle RAM, and makes the privacy/deletion verification test verifiable in a single `.db` file.

---

## 2. Backend Language & API Framework

The backend must handle PDF/markdown parsing, custom chunking, term-frequency statistics for BM25, vector search, Reciprocal Rank Fusion (RRF), citation span validation, and answer generation via the local LLM.

### Options Considered

#### Option A: Python (FastAPI + Typer CLI)
- **Overview:** Modern Python 3.11+ using FastAPI for local HTTP endpoints and Typer for CLI operations (ingest, benchmark, eval).
- **Pros for this project:**
  - **Ecosystem dominance in local AI:** Rich native libraries for parsing (PyMuPDF/fitz, pypdf), tokenization (tiktoken, tokenizers), mathematical vectors (NumPy), and evaluation metrics.
  - **Minimal glue code:** Direct access to Ollama SDK and sentence-level embeddings without external subprocess wrapping or foreign-function interfaces.
  - **Developer focus:** Allows 100% of development effort to go into the custom retrieval algorithms (chunking, BM25, fusion, citations) rather than boilerplate.
- **Cons for this project:**
  - Slower execution than compiled languages (mitigated by NumPy vectorized operations for BM25 and cosine distance).
  - Requires clean virtual environment management.
- **Offline Support:** 100% offline once virtual environment dependencies are installed.
- **License:** Python Software Foundation (PSF) License (permissive), FastAPI (MIT).

#### Option B: TypeScript / Node.js (Fastify / Express)
- **Overview:** TypeScript backend running on Node.js.
- **Pros for this project:**
  - Strong static typing and fast asynchronous I/O.
  - Direct code-sharing if the UI is built with React/TypeScript or wrapped in Tauri/Electron.
- **Cons for this project:**
  - Substantially weaker native PDF parsing, text extraction, and tokenization ecosystem compared to Python.
  - Running local cross-encoders, ONNX embeddings, or custom BM25 math requires heavier third-party bindings or calling external Python/C++ processes.
- **Offline Support:** 100% offline once `npm install` completes.
- **License:** MIT License.

#### Option C: Go or Rust
- **Overview:** Compiled systems language producing a standalone single binary.
- **Pros for this project:**
  - Extremely fast execution, low memory overhead, and straightforward single-binary distribution.
- **Cons for this project:**
  - Very limited native libraries for complex PDF text extraction and cross-encoder rerankers.
  - Slower development velocity for tuning chunk sizes, RRF constants, and evaluation metrics on `EVAL.md`.
- **Offline Support:** 100% offline once compiled.
- **License:** BSD-style (Go), MIT / Apache 2.0 (Rust).

### Recommendation
**Option A: Python (FastAPI + Typer)**.  
Python is the natural choice given the project requirements. PDF extraction, chunk tokenization, embedding generation, and LLM orchestration are vastly simpler in Python, keeping the codebase small, clean, and focused on custom retrieval logic.

---

## 3. Local LLM Runtime

The LLM runtime executes local language models for answer generation on the target machine (Windows laptop, 6 GB GPU, 24 GB RAM) strictly using retrieved chunk context.

### Options Considered

#### Option A: Ollama
- **Overview:** Standalone local inference runtime packaging `llama.cpp` behind a clean CLI and local HTTP REST API.
- **Pros for this project:**
  - **Zero-friction Windows setup:** Native Windows installer with automatic NVIDIA GPU (CUDA) offloading.
  - **Model management:** One-line commands to download, quantize, and run models (e.g., `ollama pull mistral`, `ollama pull nomic-embed-text`).
  - **Unified serving:** Can simultaneously host both the generation model (e.g., Mistral 7B) and the embedding model.
  - **Clean HTTP interface:** Well-documented local REST API, compatible with standard HTTP clients and zero outbound telemetry.
- **Cons for this project:**
  - Runs as an external local background service rather than embedded directly inside the Python process.
- **Offline Support:** 100% offline once models are downloaded.
- **License:** Apache 2.0.

#### Option B: `llama-cpp-python` (Embedded llama.cpp bindings)
- **Overview:** Python C-types bindings directly embedding the `llama.cpp` inference engine into the Python process.
- **Pros for this project:**
  - Runs in-process: no external server or daemon process needed.
  - Maximum granular control over KV-cache allocation, CPU thread binding, and GPU layer offloading.
- **Cons for this project:**
  - Compiling `llama-cpp-python` with CUDA acceleration on Windows frequently fails without complex Visual Studio C++ toolchain configurations, conflicting with the 15-minute setup goal.
  - Managing raw GGUF files and quantization manually adds unnecessary project maintenance.
- **Offline Support:** 100% offline.
- **License:** MIT License.

#### Option C: LM Studio / vLLM
- **Overview:** LM Studio provides a desktop GUI and local server; vLLM is an enterprise throughput engine.
- **Pros for this project:**
  - LM Studio has a user-friendly GUI. vLLM has high batch throughput.
- **Cons for this project:**
  - LM Studio is closed-source and proprietary, incompatible with an open, verifiable portfolio codebase.
  - vLLM lacks native Windows support and requires large VRAM configurations (>16 GB VRAM) unsuitable for a 6 GB laptop GPU.
- **Offline Support:** LM Studio is offline; vLLM is offline on Linux.
- **License:** LM Studio is Proprietary; vLLM is Apache 2.0.

### Recommendation
**Option A: Ollama**.  
Ollama provides out-of-the-box Windows CUDA GPU acceleration, runs 100% offline with an Apache 2.0 license, and can serve both Mistral 7B (4-bit) for generation and `nomic-embed-text` for vector search with zero compilation hassle.

---

## 4. Embedding Model

Turns document text chunks and search queries into dense vector representations for nearest-neighbor semantic search.

### Options Considered

#### Option A: `nomic-embed-text` (v1.5)
- **Overview:** 137M parameter model developed by Nomic AI with an 8,192 token context window, outputting 768-dimensional embeddings.
- **Pros for this project:**
  - **Large context window (8,192 tokens):** Comfortably handles 400-token chunks + 50-token overlap + headers/metadata without any risk of truncation.
  - **Ollama native:** Supported directly within Ollama (`ollama pull nomic-embed-text`), avoiding duplicate model loaders or PyTorch overhead.
  - **High retrieval quality:** Outperforms many larger models on MTEB retrieval benchmarks.
  - **Lightweight:** Only ~275 MB on disk, with negligible RAM/VRAM footprint.
- **Cons for this project:**
  - Recommends task prefixes (`search_document:` for chunks, `search_query:` for user queries) for optimal retrieval accuracy.
- **Offline Support:** 100% offline once downloaded.
- **License:** Apache 2.0.

#### Option B: `all-MiniLM-L6-v2`
- **Overview:** Standard 22.7M parameter Sentence-Transformers model outputting 384-dimensional embeddings.
- **Pros for this project:**
  - Extremely small (~80 MB) and fast CPU inference.
  - Widely cited as a classic baseline in retrieval literature.
- **Cons for this project:**
  - **Severe context limitation:** Hard limit of 256 tokens (up to 512 with degradation). Because this project's default chunk size is 400 tokens + 50 overlap, passages will frequently be truncated and information lost.
  - Significantly lower semantic retrieval quality compared to modern 2024–2025 embedding models.
- **Offline Support:** 100% offline.
- **License:** Apache 2.0.

#### Option C: `BAAI/bge-small-en-v1.5` or `bge-m3`
- **Overview:** Flagship embedding models from Beijing Academy of Artificial Intelligence (BAAI). `bge-small-en-v1.5` is 33M params (512 context); `bge-m3` is 560M params (8,192 context).
- **Pros for this project:**
  - Top-tier retrieval accuracy across dense and sparse benchmarks.
- **Cons for this project:**
  - `bge-small-en-v1.5` has a tight 512-token context limit.
  - `bge-m3` is large (~2.2 GB) and resource-heavy for a laptop running a simultaneous generation LLM on 24 GB RAM / 6 GB VRAM.
- **Offline Support:** 100% offline.
- **License:** MIT License.

### Recommendation
**Option A: `nomic-embed-text` (v1.5)**.  
Its 8,192-token context window eliminates chunk truncation issues, it integrates cleanly into Ollama, runs with low latency on laptop hardware, and has an Apache 2.0 license.

---

## 5. Reranker Model

A cross-encoder model that scores `(query, passage)` candidate pairs from the top fused results (e.g., top 20) to output the best candidates (e.g., top 5) for answer generation.

### Options Considered

#### Option A: No Reranker in v1 (Establish pure BM25 + Vector + RRF baseline; test in M3)
- **Overview:** Rely on Reciprocal Rank Fusion (RRF) between BM25 keyword and dense vector retrieval for v1, and introduce the cross-encoder reranker in Milestone 3 as an explicitly measured experiment.
- **Pros for this project:**
  - **Follows PRD & TASKS.md specifications:** PRD section 7, section 9, and TASKS.md mandate measuring retrieval quality with and without reranking to demonstrate the delta.
  - **Faster milestone completion:** M1 and M2 can be completed and evaluated rapidly without extra latency or dependency bloat.
  - **Lower query latency:** Bypasses cross-encoder inference time, keeping initial p50 latency fast.
- **Cons for this project:**
  - Complex semantic questions with lexical overlap may rank slightly lower before M3.
- **Offline Support:** 100% offline (no external model).
- **License:** N/A.

#### Option B: `cross-encoder/ms-marco-MiniLM-L-6-v2` (or FlashRank)
- **Overview:** 22M parameter cross-encoder trained on MS MARCO. Can be run via PyTorch `sentence-transformers` or through `FlashRank` (quantized ONNX, ultra-lightweight).
- **Pros for this project:**
  - Small disk size (~80 MB) and fast CPU inference (<15 ms for 20 candidates).
  - Measurably boosts Recall@k and Precision@k on nuanced queries.
  - FlashRank option requires zero PyTorch or CUDA dependencies.
- **Cons for this project:**
  - Adds another moving part and model file to download before basic pipeline evaluation is established.
- **Offline Support:** 100% offline once downloaded.
- **License:** Apache 2.0.

#### Option C: `BAAI/bge-reranker-base` / `bge-reranker-v2-m3`
- **Overview:** State-of-the-art cross-encoders from BAAI (278M–560M parameters).
- **Pros for this project:**
  - Highest discriminative accuracy on multi-document cross-referencing questions.
- **Cons for this project:**
  - Computationally heavy on CPU (100–300 ms for 20 candidates), noticeably impacting query p95 latency.
  - Requires more VRAM or system RAM, competing with Mistral 7B.
- **Offline Support:** 100% offline.
- **License:** MIT License.

### Recommendation
**Option A: No Reranker for v1 initial baseline (add Option B `ms-marco-MiniLM-L-6-v2` / FlashRank as measured M3 experiment)**.  
Shipping the initial pipeline with pure BM25 + Vector + RRF provides a clear, unpolluted baseline. Milestone 3 can then measure and document the exact quality and latency impact of introducing `ms-marco-MiniLM-L-6-v2` or FlashRank.

---

## 6. Target Platform

The deployment and execution environment for local development, evaluation runs, benchmarks, and offline verification.

### Options Considered

#### Option A: Windows Native (PowerShell / Windows 11)
- **Overview:** Application runs directly on the Windows host OS using native Python 3.11+, PowerShell setup/execution scripts, and native Windows Ollama.
- **Pros for this project:**
  - **Direct hardware match:** Matches the developer's laptop (`LAPTOP-2MKE14EU\Sohan`) without any virtualization layer.
  - **Zero file-system I/O overhead:** Reads user lecture notes and PDFs directly from native Windows folders (`Desktop`, `Documents`) without WSL2 9P cross-filesystem performance degradation.
  - **Native GPU access:** Native Windows Ollama automatically detects the NVIDIA GPU (6 GB VRAM) with zero virtualization bridging.
- **Cons for this project:**
  - Shell scripts must use PowerShell or cross-platform Python scripts (`python run.py`) rather than Bash.
  - Running Docker requires Docker Desktop (avoided if SQLite is selected).
- **Offline Support:** 100% offline.
- **License:** N/A (Standard OS).

#### Option B: Windows via WSL2 (Ubuntu Linux inside Windows)
- **Overview:** All code, Python environments, and database services run inside a WSL2 Linux distribution.
- **Pros for this project:**
  - Standard Linux environment: Bash scripting, standard package managers (`apt`), and native Linux networking tools (`tcpdump` for offline packet-capture proof).
- **Cons for this project:**
  - Accessing files on the Windows filesystem (e.g. `/mnt/c/Users/...`) introduces noticeable I/O latency during PDF parsing and ingestion.
  - Extra setup layer that prospective evaluators on Windows would have to configure.
- **Offline Support:** 100% offline.
- **License:** N/A.

#### Option C: Cross-Platform Python Core (Windows Native First)
- **Overview:** Build all core application logic using path-agnostic standard libraries (`pathlib.Path`, OS-agnostic subprocesses), with primary execution and benchmarks validated natively on Windows.
- **Pros for this project:**
  - Runs natively on Windows today with zero friction, while remaining 100% portable to macOS and Linux without code rewrites.
  - Setup scripts can be standard Python CLI commands (`python setup.py` or Typer CLI), eliminating shell-specific dependencies.
- **Cons for this project:**
  - Requires maintaining path discipline (`pathlib`) across all module code.
- **Offline Support:** 100% offline.
- **License:** N/A.

### Recommendation
**Option C: Cross-Platform Python Core (tested natively on Windows)**.  
Developing directly on the Windows host ensures native GPU acceleration and rapid ingestion of desktop notes, while adhering to cross-platform Python standards guarantees broad portfolio appeal and portability across Windows, Linux, and macOS.

---

## Summary of Evaluated Options & Recommendations

| Stack Decision | Evaluated Options | Recommended Option | Reason for Recommendation |
|----------------|-------------------|--------------------|---------------------------|
| **Database / Index Store** | 1. SQLite + `sqlite-vec`<br>2. PostgreSQL + `pgvector`<br>3. DuckDB | **SQLite + `sqlite-vec` (or NumPy)** | Zero Docker/daemon overhead, zero idle RAM, single-file simplicity, instant <15-minute setup. |
| **Backend Language** | 1. Python (FastAPI + Typer)<br>2. TypeScript / Node.js<br>3. Go / Rust | **Python (FastAPI + Typer)** | Rich ecosystem for PDF parsing, embeddings, and token math; minimal glue code. |
| **Local LLM Runtime** | 1. Ollama<br>2. `llama-cpp-python`<br>3. LM Studio / vLLM | **Ollama** | Native Windows CUDA GPU offload, Apache 2.0 license, clean HTTP REST API, zero compilation friction. |
| **Embedding Model** | 1. `nomic-embed-text` (v1.5)<br>2. `all-MiniLM-L6-v2`<br>3. `bge-small-en-v1.5` / `bge-m3` | **`nomic-embed-text` (v1.5)** | 8,192 context window (no chunk truncation), Apache 2.0 license, runs inside Ollama with ~275 MB footprint. |
| **Reranker Model** | 1. None for v1 baseline<br>2. `ms-marco-MiniLM-L-6-v2` / FlashRank<br>3. `bge-reranker-base` | **None for v1 baseline (test Option 2 in M3)** | Establishes a pure BM25+Vector+RRF baseline first, enabling exact measurement of reranker impact in M3. |
| **Target Platform** | 1. Windows Native<br>2. Windows via WSL2<br>3. Cross-Platform Core (Windows first) | **Cross-Platform Core (Windows Native first)** | Native GPU and file access on current Windows laptop; path-agnostic Python code preserves Linux/macOS portability. |

