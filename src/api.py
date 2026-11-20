"""Local FastAPI endpoints for document upload and grounded conversations."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pypdf import PdfReader

from src import cache, db
from src.answer import generate_answer, get_generation_top_k
from src.chunk import chunk_page_records
from src.cite import resolve_citations
from src.index import bm25_search, build_bm25_index
from src.ingest import parse_pdf_file
from src.retrieve import get_retrieval_params

ROOT = Path(__file__).resolve().parents[1]
STATIC_INDEX = ROOT / "static" / "index.html"
UPLOAD_DIR = ROOT / "data" / "uploads"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
documents: dict[str, dict[str, Any]] = {}
_bm25_index: dict[str, Any] | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    _invalidate_document_cache()
    db.initialize_database()
    cache.initialize_cache()
    yield


app = FastAPI(title="AI Chat", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    doc_ids: list[str] | None = None
    conversation_id: str | None = None


def _invalidate_document_cache(document_id: str | None = None) -> None:
    global _bm25_index
    if document_id is None:
        documents.clear()
    else:
        documents.pop(document_id, None)
    _bm25_index = None


def _get_bm25_index() -> dict[str, Any]:
    global _bm25_index
    if _bm25_index is None:
        chunks = db.get_chunks()
        documents.clear()
        for chunk in chunks:
            documents.setdefault(chunk["doc_id"], {"chunks": []})["chunks"].append(chunk)
        _bm25_index = build_bm25_index(chunks)
    return _bm25_index


def _upload_path(document_id: str) -> Path:
    normalized_id = str(uuid.UUID(document_id))
    return UPLOAD_DIR / f"{normalized_id}.pdf"


async def _index_upload(file: UploadFile) -> dict[str, Any]:
    saved_path: Path | None = None
    persisted = False
    try:
        filename = Path(file.filename or "").name
        if not filename or not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Choose a PDF file.")

        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="PDF must be 25 MB or smaller.")

        file_hash = sha256(content).hexdigest()
        existing = db.get_document_by_sha256(file_hash)
        if existing is not None:
            return {**existing, "already_indexed": True}

        document_id = str(uuid.uuid4())
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        saved_path = _upload_path(document_id)
        saved_path.write_bytes(content)
        try:
            pages = len(PdfReader(BytesIO(content)).pages)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read PDF: {exc}") from exc

        records = parse_pdf_file(saved_path, file_hash=file_hash)
        for record in records:
            record["file"] = filename
        if not any(record.get("text", "").strip() for record in records):
            raise HTTPException(status_code=400, detail="The PDF contains no readable text.")

        chunks = chunk_page_records(records, config_path=ROOT / "config.toml")
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Could not split the PDF text into chunks.",
            )

        db.insert_document(
            {
                "id": document_id,
                "filename": filename,
                "sha256": file_hash,
                "pages": pages,
            },
            chunks,
        )
        persisted = True
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"Could not index PDF: {exc}") from exc
    finally:
        if saved_path is not None and not persisted:
            saved_path.unlink(missing_ok=True)
        await file.close()

    _invalidate_document_cache()
    return {
        "id": document_id,
        "filename": filename,
        "sha256": file_hash,
        "pages": pages,
        "chunk_count": len(chunks),
        "already_indexed": False,
    }


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_INDEX, media_type="text/html")


@app.post("/upload")
async def upload_pdfs(files: Annotated[list[UploadFile], File()]) -> dict[str, Any]:
    return {"documents": [await _index_upload(file) for file in files]}


@app.get("/documents")
def list_documents() -> list[dict[str, Any]]:
    return db.list_documents()


@app.delete("/documents/{document_id}")
def remove_document(document_id: str) -> dict[str, bool]:
    try:
        upload_path = _upload_path(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Document not found.") from exc

    deleted = db.delete_document(document_id)
    if deleted is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    cache.delete_for_document(document_id)
    _invalidate_document_cache(document_id)
    upload_path.unlink(missing_ok=True)
    return {"deleted": True}


@app.post("/ask")
def ask_question(request: AskRequest) -> dict[str, Any]:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    available_documents = db.list_documents()
    if request.doc_ids is None:
        selected_ids = [document["id"] for document in available_documents]
    else:
        selected_ids = list(dict.fromkeys(request.doc_ids))
        existing_ids = {document["id"] for document in available_documents}
        missing_ids = [document_id for document_id in selected_ids if document_id not in existing_ids]
        if missing_ids:
            raise HTTPException(status_code=404, detail="One or more documents were not found.")
    if not selected_ids:
        raise HTTPException(status_code=400, detail="Select at least one document.")

    conversation_id = request.conversation_id
    conversation_history: list[dict[str, str]] = []
    if conversation_id and not db.conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    if conversation_id:
        conversation_history = [
            {"role": message["role"], "content": message["content"]}
            for message in db.get_conversation_messages(conversation_id)[-2:]
            if message["role"] in {"user", "assistant"}
        ]

    # The cache key covers only the question and documents, so answers that depend
    # on earlier turns (follow-ups) are neither served from nor written to the cache.
    cache_key: str | None = None
    if not conversation_history and cache.is_enabled(ROOT / "config.toml"):
        hashes_by_id = {document["id"]: document["sha256"] for document in available_documents}
        doc_hashes = {document_id: hashes_by_id[document_id] for document_id in selected_ids}
        cache_key = cache.make_key(question, doc_hashes)
        hit = cache.get(cache_key)
        if hit is not None:
            if conversation_id is None:
                conversation_id = db.create_conversation(question[:120])
            db.save_exchange(conversation_id, question, hit["answer"], hit["citations"])
            return {
                "answer": hit["answer"],
                "citations": hit["citations"],
                "conversation_id": conversation_id,
                "cached": True,
            }

    index = _get_bm25_index()
    top_n = get_retrieval_params(ROOT / "config.toml")["top_n"]
    selected_id_set = set(selected_ids)
    retrieval_query = "\n".join(
        [message["content"] for message in conversation_history] + [question]
    )
    all_results = bm25_search(retrieval_query, index, top_n=len(index["chunks"]))
    ranked_chunks = [
        chunk
        for chunk, _ in all_results
        if chunk["doc_id"] in selected_id_set
    ][:top_n]
    generation_chunks = ranked_chunks[:get_generation_top_k(ROOT / "config.toml")]
    answer = generate_answer(
        question,
        generation_chunks,
        conversation_history=conversation_history,
        config_path=ROOT / "config.toml",
    )
    citation_result = resolve_citations(answer, generation_chunks)
    citations = [
        {
            "file": citation["file"],
            "page": citation["page"],
            "start_char": citation["start_char"],
            "end_char": citation["end_char"],
        }
        for citation in citation_result.valid_citations
    ]

    if conversation_id is None:
        conversation_id = db.create_conversation(question[:120])
    db.save_exchange(conversation_id, question, answer, citations)
    if cache_key is not None:
        cache.store(cache_key, question, doc_hashes, answer, citations)
    return {
        "answer": answer,
        "citations": citations,
        "conversation_id": conversation_id,
        "cached": False,
    }


@app.get("/conversations")
def list_conversations() -> list[dict[str, Any]]:
    return db.list_conversations()


@app.get("/conversations/{conversation_id}/messages")
def get_conversation_messages(conversation_id: str) -> list[dict[str, Any]]:
    if not db.conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return db.get_conversation_messages(conversation_id)
