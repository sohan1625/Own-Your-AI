"""Chunking module.

Responsible for:
- Splitting extracted page text into overlapping chunks of configurable token size and overlap.
- Retaining full metadata for each chunk: source file, page number, start character, end character, text.
- Providing a ``parent_id`` that identifies the section (page) the chunk belongs to.

Chunk size and overlap are read from ``config.toml`` (section ``[chunking]``). The defaults are
``chunk_size = 400`` tokens and ``overlap = 50`` tokens. For the initial version we use a simple
word‑based approximation for token count – about **0.75 words per token** – to avoid pulling in a
tokeniser library. This is an approximation and will be refined later.

Chunks never cross page boundaries; each page is processed independently. If a page is empty, no
chunks are produced.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Iterable, List, Mapping

import tomllib

# ---------------------------------------------------------------------------
# Configuration helpers (mirrored from src.ingest to avoid circular imports)
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path = "config.toml") -> dict[str, Any]:
    """Load the TOML configuration file and return the full dictionary.

    Raises ``FileNotFoundError`` if the file does not exist.
    """
    cfg_file = Path(config_path)
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(cfg_file, "rb") as f:
        return tomllib.load(f)


def get_chunking_params(config_path: str | Path = "config.toml") -> Mapping[str, int]:
    """Extract ``chunk_size`` and ``overlap`` token counts from the config.

    Returns a mapping with keys ``chunk_size`` and ``overlap``. Defaults follow
    ``RAG_PIPELINE.md`` (400 tokens, 50 token overlap).
    """
    cfg = load_config(config_path)
    chunk_cfg = cfg.get("chunking", {})
    chunk_size = int(chunk_cfg.get("chunk_size", 400))
    overlap = int(chunk_cfg.get("overlap", 50))
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if overlap < 0:
        raise ValueError("overlap must be non‑negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    return {"chunk_size": chunk_size, "overlap": overlap}

# ---------------------------------------------------------------------------
# Helper: map word index to character offset in the original text
# ---------------------------------------------------------------------------

def _word_offsets(text: str) -> List[int]:
    """Return a list where each entry is the character offset of the *start* of a word.

    Words are defined by a simple ``str.split()`` on whitespace. This function walks the
    string once, recording the index of the first non‑space character after a whitespace run.
    The result length equals the number of words returned by ``text.split()``.
    """
    offsets: List[int] = []
    in_word = False
    for i, ch in enumerate(text):
        if not ch.isspace():
            if not in_word:
                offsets.append(i)
                in_word = True
        else:
            in_word = False
    return offsets

# ---------------------------------------------------------------------------
# Core chunking logic
# ---------------------------------------------------------------------------

def chunk_page_records(
    page_records: Iterable[Mapping[str, Any]],
    *,
    config_path: str | Path = "config.toml",
) -> List[dict[str, Any]]:
    """Chunk a sequence of page records produced by ``src.ingest.parse_folder``.

    Each input record must contain the keys ``file``, ``page``, ``text``, and optionally ``file_hash``.
    The function returns a list of chunk dictionaries with the following keys:

    - ``file`` (str): source PDF path
    - ``page`` (int): 1‑based page number (inherited from the source record)
    - ``start_char`` (int): character offset of the first word of the chunk within the original page text
    - ``end_char`` (int): character offset *after* the last word of the chunk
    - ``text`` (str): whitespace‑joined words forming the chunk
    - ``parent_id`` (str): identifier of the page/section this chunk belongs to. For v1 we use a deterministic
      identifier based on the file path and page number: ``f"{file}#page{page}"``.
    - ``file_hash`` (str, optional): carried over from the source record for later change detection.

    The chunk size and overlap are expressed in **tokens** in the config, but we approximate tokens as
    0.75 words. Therefore ``chunk_words = int(chunk_size / 0.75)`` and ``overlap_words = int(overlap / 0.75)``.
    This is a coarse estimate suitable for the early prototype.
    """
    params = get_chunking_params(config_path)
    chunk_size_tokens = params["chunk_size"]
    overlap_tokens = params["overlap"]

    # Approximate conversion: 1 token ≈ 0.75 words → words = tokens / 0.75
    # Use round to the nearest integer for a stable size.
    chunk_words = max(1, int(round(chunk_size_tokens / 0.75)))
    overlap_words = max(0, int(round(overlap_tokens / 0.75)))

    step = max(1, chunk_words - overlap_words)

    chunks: List[dict[str, Any]] = []

    for rec in page_records:
        text: str = rec.get("text", "") or ""
        if not text:
            # Empty page → no chunks produced.
            continue
        words = text.split()
        if not words:
            continue
        # Pre‑compute character offsets for each word to be able to report start/end chars.
        char_offsets = _word_offsets(text)

        # Parent identifier – stable per page/section.
        parent_id = f"{rec['file']}#page{rec['page']}"

        for start_idx in range(0, len(words), step):
            end_idx = min(start_idx + chunk_words, len(words))
            if start_idx >= len(words):
                break
            slice_words = words[start_idx:end_idx]
            chunk_text = " ".join(slice_words)
            start_char = char_offsets[start_idx]
            # ``end_idx`` points to the first word after the slice (or len(words)).
            if end_idx < len(char_offsets):
                end_char = char_offsets[end_idx]
            else:
                end_char = len(text)
            chunks.append({
                "file": rec["file"],
                "page": rec["page"],
                "start_char": start_char,
                "end_char": end_char,
                "text": chunk_text,
                "parent_id": parent_id,
                "file_hash": rec.get("file_hash"),
                # ``chunk_id`` is optional for now; can be added later if needed.
                "chunk_id": str(uuid.uuid4()),
            })
            if end_idx == len(words):
                break
    return chunks

# Backwards‑compatible alias – earlier placeholder name.
chunk_pages = chunk_page_records
