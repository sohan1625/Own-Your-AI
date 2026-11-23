"""Re-chunk stored documents with the current [chunking] settings.

Documents keep the chunks they were uploaded with. After changing
``[chunking] strategy`` (or chunk size), run this to rebuild every document's
chunks and embeddings in place. Chat history is kept; cached answers that used a
re-chunked document are deleted.

The source PDF is the stored upload (storage/uploads/<id>.pdf). Documents uploaded
before uploads were stored there can be matched by sha256 against PDFs in a folder:

    python -m scripts.rechunk                      # documents with a stored upload
    python -m scripts.rechunk --source-dir data    # also match PDFs in data/ by hash
    python -m scripts.rechunk --dry-run            # show what would change
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src import api, db
from src.chunk import chunk_page_records, get_chunking_strategy
from src.ingest import compute_file_sha256, parse_pdf_file


def find_sources(documents: list[dict], source_dir: Path | None) -> dict[str, Path]:
    by_hash: dict[str, Path] = {}
    if source_dir is not None:
        for pdf in sorted(source_dir.rglob("*.pdf")):
            by_hash.setdefault(compute_file_sha256(pdf), pdf)
    sources = {}
    for document in documents:
        stored = api.UPLOAD_DIR / f"{document['id']}.pdf"
        if stored.is_file():
            sources[document["id"]] = stored
        elif document["sha256"] in by_hash:
            sources[document["id"]] = by_hash[document["sha256"]]
    return sources


def rechunk_document(document: dict, source: Path) -> tuple[int, int]:
    """Returns (old chunk count, new chunk count)."""
    if compute_file_sha256(source) != document["sha256"]:
        raise ValueError(f"{source} does not match the stored document's sha256")
    records = parse_pdf_file(source, file_hash=document["sha256"])
    for record in records:
        record["file"] = document["filename"]
    chunks = chunk_page_records(records, config_path=api.CONFIG_PATH, embed=api._embed_document_text)
    if not chunks:
        raise ValueError(f"no chunks produced for {document['filename']}")
    embeddings = api._embed_chunks(chunks)
    removed = db.replace_chunks(document["id"], chunks, embeddings)
    return removed, len(chunks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, help="folder of PDFs to match by sha256")
    parser.add_argument("--dry-run", action="store_true", help="list documents and sources only")
    args = parser.parse_args(argv)

    db.initialize_database()
    from src import cache

    cache.initialize_cache()
    documents = db.list_documents()
    sources = find_sources(documents, args.source_dir)
    print(f"Chunking strategy: {get_chunking_strategy(api.CONFIG_PATH)}; {len(documents)} document(s)")
    failures = 0
    for document in documents:
        source = sources.get(document["id"])
        if source is None:
            print(f"  SKIP {document['filename']}: no stored upload and no PDF with the same sha256")
            continue
        if args.dry_run:
            print(f"  would re-chunk {document['filename']} ({document['chunk_count']} chunks) from {source}")
            continue
        try:
            old, new = rechunk_document(document, source)
        except Exception as exc:  # report and continue with the next document
            failures += 1
            print(f"  FAIL {document['filename']}: {exc}")
            continue
        print(f"  {document['filename']}: {old} -> {new} chunks")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
