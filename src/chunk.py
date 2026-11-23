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

import logging
import math
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping

import tomllib

logger = logging.getLogger(__name__)

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
    embed: Callable[[str], List[float]] | None = None,
) -> List[dict[str, Any]]:
    """Chunk pages with the strategy set in ``[chunking] strategy`` (default ``fixed``).

    ``embed`` (semantic only) embeds one sentence; by default the local embedding
    model is called with the document prefix. If embedding fails (for example the
    model server is down), the pages are chunked with the fixed strategy instead, so
    a document can still be indexed and searched by keyword.
    """
    page_records = list(page_records)
    if get_chunking_strategy(config_path) == "semantic":
        try:
            return semantic_chunk_page_records(page_records, config_path=config_path, embed=embed)
        except OSError as exc:
            logger.warning("Semantic chunking unavailable (%s); using fixed chunks.", exc)
    return _fixed_chunks(page_records, config_path=config_path)


def _fixed_chunks(
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

# ---------------------------------------------------------------------------
# Semantic chunking ([chunking] strategy = "semantic")
# ---------------------------------------------------------------------------

# A sentence ends at ., ! or ? followed by whitespace, or at a blank line.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n\s*\n")


def get_chunking_strategy(config_path: str | Path = "config.toml") -> str:
    cfg = load_config(config_path)
    strategy = cfg.get("chunking", {}).get("strategy", "fixed")
    if strategy not in ("fixed", "semantic"):
        raise ValueError(f"[chunking] strategy must be 'fixed' or 'semantic', not {strategy!r}")
    return strategy


def get_semantic_percentile(config_path: str | Path = "config.toml") -> float:
    cfg = load_config(config_path)
    value = float(cfg.get("chunking", {}).get("semantic_percentile", 25))
    if not 0 <= value <= 100:
        raise ValueError("semantic_percentile must be between 0 and 100")
    return value


def split_sentences(text: str) -> List[tuple[int, int]]:
    """Return ``(start, end)`` character spans of the sentences in *text*.

    Spans exclude surrounding whitespace, so ``text[start:end]`` is the sentence.
    """
    spans: List[tuple[int, int]] = []
    pos = 0
    for match in _SENTENCE_END.finditer(text):
        _append_span(text, pos, match.start(), spans)
        pos = match.end()
    _append_span(text, pos, len(text), spans)
    return spans


def _append_span(text: str, start: int, end: int, spans: List[tuple[int, int]]) -> None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start < end:
        spans.append((start, end))


def _cosine(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    return dot / (n1 * n2) if n1 and n2 else 0.0


def percentile(values: List[float], pct: float) -> float:
    """Linear-interpolated percentile (same method as numpy's default)."""
    if not values:
        raise ValueError("percentile of an empty list")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)


def semantic_chunk_page_records(
    page_records: Iterable[Mapping[str, Any]],
    *,
    config_path: str | Path = "config.toml",
    embed: Callable[[str], List[float]] | None = None,
    percentile_value: float | None = None,
) -> List[dict[str, Any]]:
    """Chunk pages at topic shifts between consecutive sentences.

    Each page is split into sentences and every sentence is embedded. A new chunk
    starts where the cosine similarity between two consecutive sentences falls
    below the ``percentile_value``-th percentile of all consecutive-sentence
    similarities in the input (so a lower percentile gives fewer, larger chunks).
    Chunks never cross page boundaries and never exceed ``chunk_size`` (same
    word approximation as the fixed strategy); an over-long run of sentences is
    cut at a sentence boundary, or inside a sentence only if one sentence alone is
    too long. ``start_char``/``end_char`` are exact offsets into the page text.
    """
    if embed is None:
        from src.index import embed_text, get_embedding_prefix

        prefix = get_embedding_prefix("document", config_path=config_path)

        def embed(text: str) -> List[float]:
            return embed_text(text, prefix=prefix, config_path=config_path)

    if percentile_value is None:
        percentile_value = get_semantic_percentile(config_path)
    max_words = max(1, int(round(get_chunking_params(config_path)["chunk_size"] / 0.75)))

    pages: List[tuple[Mapping[str, Any], str, List[tuple[int, int]], List[float]]] = []
    all_sims: List[float] = []
    for rec in page_records:
        text: str = rec.get("text", "") or ""
        spans = split_sentences(text)
        if not spans:
            continue
        vectors = [embed(text[s:e]) for s, e in spans]
        sims = [_cosine(vectors[i], vectors[i + 1]) for i in range(len(spans) - 1)]
        all_sims.extend(sims)
        pages.append((rec, text, spans, sims))

    cutoff = percentile(all_sims, percentile_value) if all_sims else 0.0

    chunks: List[dict[str, Any]] = []
    for rec, text, spans, sims in pages:
        groups: List[List[tuple[int, int]]] = [[spans[0]]]
        for i, sim in enumerate(sims):
            if sim < cutoff:
                groups.append([spans[i + 1]])
            else:
                groups[-1].append(spans[i + 1])
        for group in groups:
            for start, end in _limit_size(text, group, max_words):
                chunks.append(_make_chunk(rec, text, start, end))
    return chunks


def _limit_size(
    text: str, group: List[tuple[int, int]], max_words: int
) -> List[tuple[int, int]]:
    """Split a sentence group into spans of at most ``max_words`` words."""
    out: List[tuple[int, int]] = []
    current: tuple[int, int] | None = None
    current_words = 0
    for start, end in group:
        words = len(text[start:end].split())
        if words > max_words:
            if current is not None:
                out.append(current)
                current, current_words = None, 0
            offsets = [start + o for o in _word_offsets(text[start:end])]
            for i in range(0, len(offsets), max_words):
                piece_end = offsets[i + max_words] if i + max_words < len(offsets) else end
                _append_span(text, offsets[i], piece_end, out)
            continue
        if current is not None and current_words + words > max_words:
            out.append(current)
            current, current_words = None, 0
        current = (start, end) if current is None else (current[0], end)
        current_words += words
    if current is not None:
        out.append(current)
    return out


def _make_chunk(rec: Mapping[str, Any], text: str, start: int, end: int) -> dict[str, Any]:
    return {
        "file": rec["file"],
        "page": rec["page"],
        "start_char": start,
        "end_char": end,
        "text": " ".join(text[start:end].split()),
        "parent_id": f"{rec['file']}#page{rec['page']}",
        "file_hash": rec.get("file_hash"),
        "chunk_id": str(uuid.uuid4()),
    }


# Backwards‑compatible alias – earlier placeholder name.
chunk_pages = chunk_page_records
