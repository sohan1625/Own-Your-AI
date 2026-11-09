# Own Your Intelligence (working title)

A private, offline assistant that answers questions over your own documents and cites the exact source for every answer. Nothing you upload leaves your machine.

> Status: in development. Stack: Python + FastAPI, PostgreSQL with pgvector (Docker), Ollama (Mistral 7B + nomic-embed-text), Windows target.

## Why

Study notes and personal documents are sensitive, and most "chat with your docs" tools are cloud-first. This project keeps the whole pipeline local and measures how well it works.

## Features (v1)

- Import PDF, markdown, and text files from chosen folders
- Answers grounded only in retrieved passages
- Citations with file, page, and passage for every answer
- Says "I don't know" when the documents don't contain the answer
- Works with the network turned off (see [PRIVACY.md](MD%20files/PRIVACY.md))

## Setup

Target: under 15 minutes from clone to first query.

### Prerequisites

- **OS:** Windows 10/11 (x64)
- **Python:** 3.11+
- **Docker Desktop:** Installed and running (for PostgreSQL with `pgvector`)
- **Ollama:** Installed on Windows (from [ollama.com](https://ollama.com))

### 1. Clone & Environment Setup

Open PowerShell and navigate to the project root:

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install initial dependencies
pip install -r requirements.txt

# Configure environment variables
Copy-Item .env.example .env
```

### 2. Start PostgreSQL with pgvector

Ensure Docker Desktop is running, then start the database container:

```powershell
docker compose up -d

# Verify container is running on port 5432
docker compose ps
```

### 3. Pull Local AI Models via Ollama

Download the required models (one-time explicit download; runtime is strictly offline):

```powershell
# Generation model (Mistral 7B Instruct 4-bit, Apache 2.0)
ollama pull mistral

# Embedding model (nomic-embed-text, 8192 context, Apache 2.0)
ollama pull nomic-embed-text
```

### 4. Run Verification Tests

Verify the project scaffolding and module imports:

```powershell
pytest
```

## Results

Headline numbers from [EVAL.md](MD%20files/EVAL.md) and [BENCHMARKS.md](MD%20files/BENCHMARKS.md) go here once measured.

| Metric | Target | Result |
|--------|--------|--------|
| Recall@5 (hybrid) | ≥ 0.80 | TBD |
| Faithfulness | ≥ 0.90 | TBD |
| Hallucination on unanswerable questions | < 10% | TBD |
| Answers with valid citation | 100% | TBD |

## Documentation

- [PRD.md](MD%20files/PRD.md): scope, goals, non-goals, requirements
- [ARCHITECTURE.md](MD%20files/ARCHITECTURE.md): system design
- [RAG_PIPELINE.md](MD%20files/RAG_PIPELINE.md): chunking, retrieval, fusion, reranking, answering, citing
- [EVAL.md](MD%20files/EVAL.md): eval set, metrics, results
- [PRIVACY.md](MD%20files/PRIVACY.md): what is stored and the offline test
- [DECISIONS.md](MD%20files/DECISIONS.md): tech choices and why
- [TASKS.md](MD%20files/TASKS.md): what's next
- [AI_CONTEXT.md](MD%20files/AI_CONTEXT.md): rules for AI coding assistants
- [STACK_OPTIONS.md](STACK_OPTIONS.md): evaluation of stack options and trade-offs

## Known similar tools

AnythingLLM, PrivateGPT, Open WebUI, GPT4All LocalDocs, and Khoj solve related problems. This project's focus is measured retrieval quality and a verifiable offline guarantee, not feature breadth.

## License

TBD

