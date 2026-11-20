# Privacy

## Principle

Your documents, their text, their embeddings, your questions, and the answers stay on your machine. The only network use the app needs is the one-time model download (`ollama pull`).

## What is stored, and where

| Data | Location | Format | Deleted when |
|------|----------|--------|--------------|
| Uploaded PDFs | `storage/uploads/<document-id>.pdf` (git-ignored) | Original file | The document is deleted |
| Document metadata (name, hash, page count) | PostgreSQL `documents` table | Rows | The document is deleted |
| Extracted text, chunks, and spans | PostgreSQL `chunks` table | Rows | The document is deleted |
| Embeddings | PostgreSQL `chunks.embedding` (pgvector) | 768-float vectors | The document is deleted |
| Keyword (BM25) index | Process memory only, rebuilt from `chunks` | In memory | Rebuilt after every upload or delete |
| Cached answers | PostgreSQL `answer_cache` table | Rows | Any document they used is deleted |
| Chat history (questions and answers) | PostgreSQL `conversations` and `messages` tables | Rows | Not deleted by the app (no delete-chat feature yet) |
| Models | Ollama's model folder (`%USERPROFILE%\.ollama\models` on Windows) | Model files | The user removes them (`ollama rm`) |

PostgreSQL runs in Docker (`docker-compose.yml`) and keeps its files in the `pgdata` Docker volume. Removing everything the app stored: `docker compose down -v` (deletes the volume) and delete `storage/`.

The eval corpus is separate: PDFs placed in `data/` are read by the eval scripts and never by the app.

## What leaves the machine

- **The app:** nothing. It talks only to PostgreSQL on localhost and to Ollama on 127.0.0.1. The code refuses any model host that is not 127.0.0.1 or localhost (`_validate_host` in `src/answer.py` and `src/index.py`).
- **Model download:** only when you run `ollama pull`.
- **Ollama's system-tray program** (`ollama app.exe` on Windows) connects to ollama.com to check for updates. It never receives documents or questions: those go to the model server (`ollama.exe`) on 127.0.0.1. To avoid the update checks, quit the tray program and run `ollama serve` instead, or work offline.
- No telemetry, crash reporting, or analytics in the app.

## Offline test

Run with the network disconnected:

```powershell
python -m scripts.offline_check
```

It refuses to start while the internet is reachable (DNS lookup and TCP to public IPs). It then uploads a test PDF, asks a question, a follow-up, a repeat (cache), and an out-of-scope question, and deletes the PDF, all through the real app and models in a temporary database schema. While it runs it samples every TCP connection owned by the app (`python`), the Ollama server (`ollama.exe`), and the model runner (`llama-server`). `--with-network` runs the same flow online, as an audit that nothing connects out even when it could. Results are saved to `storage/privacy_checks/`.

| Date | Method | Outbound connections | Result |
|------|--------|---------------------|--------|
| 2026-11-20 | `offline_check --with-network` (online audit), connection sampling every 0.5 s | None from the app, Ollama server, or model runner. In one run the Ollama tray app had a connection to ollama.com (34.36.133.15:443), its update check | 8 of 8 passed |
| 2026-11-20 | `offline_check` with Wi-Fi off (DNS and TCP to 1.1.1.1 and 8.8.8.8 confirmed failing), connection sampling every 0.5 s | None | 9 of 9 passed: upload and embedding (0.3 s), answer with citation (1.9 s), follow-up (3.2 s), cache hit (0.13 s), out-of-scope refusal, chat history, deletion |

## Deletion test

```powershell
python -m scripts.deletion_check
```

Uploads a PDF containing two unique strings (one that the question mentions, and one "document-only" string that no question mentions), confirms a query returns and caches it, deletes it, and checks every place the app stores data. Runs the real app and models in a temporary database schema and upload folder.

**Result, 2026-11-20: 15 of 15 checks passed.**

| After deleting the document | Result |
|-----------------------------|--------|
| Uploaded PDF file | Deleted |
| Document row, chunks, and embeddings | Deleted |
| Any remaining chunk containing the document's text | None |
| Cached answers that used the document | Deleted |
| Keyword (BM25) index | No longer contains it |
| Vector search | Returns only the remaining document |
| Asking with the deleted document selected | Rejected (404) |
| Asking the same question again | "not found in your documents", no citation to it |

**What is retained, by design or by PostgreSQL:**

- **Chat history.** Questions and answers already in a chat stay there after the document is deleted, including answers that quoted it (for example "encrypts every packet with a rotating quartz key"). The app has no delete-chat feature yet.
- **PostgreSQL's write-ahead log.** The raw bytes of the deleted text were found only in one write-ahead log file (`pg_wal/…`), not in any table file, and were still there after `VACUUM FULL` and `CHECKPOINT`. PostgreSQL recycles these files over time and does not zero them. For a guaranteed wipe, remove the database volume (`docker compose down -v`).
