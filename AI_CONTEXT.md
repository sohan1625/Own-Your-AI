# AI Context

Rules for AI coding assistants working on this project. Read this before making changes.

## What this project is

A local, offline document Q&A assistant. Answers come only from the user's documents, with citations. Full spec: PRD.md.

## Hard rules

1. **No network calls** in runtime code. Model downloads are only in the explicit download command.
2. **No telemetry or analytics.** Do not add any logging that sends data out.
3. **Retrieval is custom.** Do not replace chunking, BM25, vector search, fusion, or citation logic with framework abstractions. Libraries are allowed for parsing and for calling the local model.
4. **Answers use only retrieved context.** Never let the generator answer from its own knowledge.
5. **Every answer cites its sources** by chunk ID, mapped to file, page, and span.
6. **Don't change the eval set** in `eval/questions.jsonl` without asking. It's frozen.
7. **Record decisions** in DECISIONS.md when a technical choice is made.

## Conventions

- Keep modules small and single-purpose: ingest, chunk, index, retrieve, fuse, rerank, answer, cite.
- Every stage has unit tests with small fixtures.
- Parameters (chunk size, overlap, k, thresholds) live in one config file, not hard-coded.
- Log retrieval scores and chunk IDs for debugging, locally only.
- Prefer plain functions and data over deep class hierarchies.

## Before writing code

1. Check TASKS.md for the current milestone.
2. Check RAG_PIPELINE.md for the stage's inputs and outputs.
3. Ask before adding a dependency, and note its license.

## Out of scope for v1

Multi-user support, auth, cloud sync, RBAC, audit logs, web browsing, a custom inference engine, GraphRAG. Do not add these unless asked.
