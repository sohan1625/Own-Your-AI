"""Local FastAPI endpoints for uploading a PDF and asking grounded questions."""

from __future__ import annotations

from collections import Counter, defaultdict
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pypdf import PdfReader

from src.answer import generate_answer, get_generation_top_k
from src.chunk import chunk_page_records
from src.cite import resolve_citations
from src.index import bm25_search, build_bm25_index, build_chunk_embeddings
from src.ingest import parse_pdf_file
from src.retrieve import get_retrieval_params, vector_retrieve


ROOT = Path(__file__).resolve().parents[1]
STATIC_INDEX = ROOT / "static" / "index.html"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
DOCUMENT_INDEX_DIR = ROOT / "data" / "index"
documents: dict[str, dict[str, Any]] = {}

app = FastAPI(title="AI Chat")


class AskRequest(BaseModel):
    doc_id: str
    question: str


def _document_index_path(doc_id: str) -> Path | None:
    try:
        normalized_id = str(uuid.UUID(doc_id))
    except (ValueError, AttributeError):
        return None
    return DOCUMENT_INDEX_DIR / f"{normalized_id}.json"


def _load_document(doc_id: str) -> dict[str, Any] | None:
    index_path = _document_index_path(doc_id)
    if index_path is None or not index_path.is_file():
        return None

    try:
        with index_path.open("r", encoding="utf-8") as index_file:
            stored = json.load(index_file)
        bm25_index = stored["bm25_index"]
        bm25_index["term_freqs"] = [
            Counter(term_freqs) for term_freqs in bm25_index["term_freqs"]
        ]
        bm25_index["doc_freq"] = Counter(bm25_index["doc_freq"])
        return {
            "chunks": stored["chunks"],
            "bm25_index": bm25_index,
            "vectors": stored["vectors"],
        }
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Stored document index could not be loaded.",
        ) from exc


def _save_document(doc_id: str, document: dict[str, Any]) -> None:
    index_path = _document_index_path(doc_id)
    if index_path is None:
        raise ValueError("Invalid document ID.")

    DOCUMENT_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=DOCUMENT_INDEX_DIR,
            suffix=".tmp",
            delete=False,
        ) as index_file:
            temp_path = index_file.name
            json.dump(document, index_file)
        Path(temp_path).replace(index_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_INDEX, media_type="text/html")


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = Path(file.filename or "").name
    if not filename or not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Choose a PDF file.")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF must be 25 MB or smaller.")

    try:
        pages = len(PdfReader(BytesIO(content)).pages)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {exc}") from exc

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        records = parse_pdf_file(temp_path)
        for record in records:
            record["file"] = filename
        if not any(record.get("text", "").strip() for record in records):
            raise HTTPException(status_code=400, detail="The PDF contains no readable text.")

        chunks = chunk_page_records(records, config_path=ROOT / "config.toml")
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not split the PDF text into chunks.")

        bm25_index = build_bm25_index(chunks)
        vectors = build_chunk_embeddings(chunks, config_path=ROOT / "config.toml")
        if not vectors or len(vectors) != len(chunks):
            raise HTTPException(status_code=503, detail="Could not create embeddings for this PDF.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not index PDF: {exc}") from exc
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        await file.close()

    doc_id = str(uuid.uuid4())
    document = {
        "chunks": chunks,
        "bm25_index": bm25_index,
        "vectors": vectors,
    }
    try:
        _save_document(doc_id, document)
    except OSError as exc:
        raise HTTPException(status_code=503, detail=f"Could not save document index: {exc}") from exc
    documents[doc_id] = document
    return {"doc_id": doc_id, "pages": pages}


@app.post("/ask")
def ask_question(request: AskRequest) -> dict[str, Any]:
    document = documents.get(request.doc_id)
    if document is None:
        document = _load_document(request.doc_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found. Re-upload the PDF.")
        documents[request.doc_id] = document
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    chunks = document["chunks"]
    top_n = get_retrieval_params(ROOT / "config.toml")["top_n"]
    bm25_results = bm25_search(question, document["bm25_index"], top_n=top_n)
    vector_results = vector_retrieve(
        question,
        chunks,
        document["vectors"],
        top_n=top_n,
        config_path=ROOT / "config.toml",
    )

    scores: dict[str, float] = defaultdict(float)
    chunk_by_id: dict[str, dict[str, Any]] = {}
    for ranked_list in (
        [chunk for chunk, _ in bm25_results],
        [chunk for chunk, _ in vector_results],
    ):
        for rank, chunk in enumerate(ranked_list, start=1):
            chunk_id = chunk["chunk_id"]
            scores[chunk_id] += 1.0 / (60 + rank)
            chunk_by_id[chunk_id] = chunk

    ranked_chunks = sorted(
        chunk_by_id.values(),
        key=lambda chunk: scores[chunk["chunk_id"]],
        reverse=True,
    )
    generation_chunks = ranked_chunks[:get_generation_top_k(ROOT / "config.toml")]
    top_vector_score = vector_results[0][1] if vector_results else None
    answer = generate_answer(
        question,
        generation_chunks,
        top_score=top_vector_score,
        config_path=ROOT / "config.toml",
    )
    citation_result = resolve_citations(answer, generation_chunks)
    citations = [
        {"file": citation["file"], "page": citation["page"]}
        for citation in citation_result.valid_citations
    ]
    return {"answer": answer, "citations": citations}
