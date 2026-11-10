"""Document ingestion module.

Responsible for:
- Reading PDF documents from a watched folder path (or configured path from config.toml).
- Computing SHA-256 hash per file for integrity and change detection.
- Extracting plain text page-by-page, recording file path, 1-based page number, text, and file hash.
- Skipping unchanged files when previous hashes are provided (per RAG_PIPELINE.md).
"""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import re
import tomllib
from typing import Any, Mapping, Sequence

# Attempt to import the PDF parser. If it is unavailable we set a sentinel
# value so that the rest of the module can be imported without error.
try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None  # type: ignore


def load_config(config_path: str | Path = "config.toml") -> dict[str, Any]:
    """Load configuration from a TOML file.

    Raises ``FileNotFoundError`` if the file does not exist.
    """
    cfg_file = Path(config_path)
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(cfg_file, "rb") as f:
        return tomllib.load(f)


def get_data_dir_from_config(config_path: str | Path = "config.toml") -> Path:
    """Read the data/documents directory path from config.toml.

    The config must contain a ``[paths]`` section with a ``data_dir`` entry.
    """
    config = load_config(config_path)
    data_dir = config.get("paths", {}).get("data_dir")
    if not data_dir:
        raise ValueError("Missing 'paths.data_dir' in configuration file.")
    return Path(data_dir)


def compute_file_sha256(file_path: str | Path) -> str:
    """Compute the SHA‑256 hex digest of a file.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


PAGE_NUMBER_PATTERN = re.compile(
    r"^(?:page\s+\d+(?:\s*(?:/|of)\s*\d+)?|\d+|[—\-]\s*\d+\s*[—\-])$",
    re.IGNORECASE,
)


def is_table_of_contents_page(text: str, min_dotted_lines: int = 4) -> bool:
    """Detect if a page is a table of contents based on dotted leader lines or dots."""
    dotted_lines = [
        line for line in text.splitlines()
        if re.search(r"(?:\.\s*){4,}", line)
    ]
    return len(dotted_lines) >= min_dotted_lines


def remove_headers_and_footers(
    pages: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    """Remove repeated page headers and footers (such as titles and 'Page N').

    A line is considered a repeated header if it appears in the top 2 lines of
    multiple pages (or matches a page-number pattern).
    A line is considered a repeated footer if it appears in the bottom 2 lines of
    multiple pages (or matches a page-number pattern).
    """
    if not pages:
        return []

    top_counts: Counter[str] = Counter()
    bottom_counts: Counter[str] = Counter()

    for _, text in pages:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines[:2]:
            if not PAGE_NUMBER_PATTERN.match(line):
                top_counts[line] += 1
        for line in lines[-2:]:
            if (
                not PAGE_NUMBER_PATTERN.match(line)
                and len(line) > 2
                and line not in ("}", "{", ";", "};", ")")
            ):
                bottom_counts[line] += 1

    total_pages = len(pages)
    min_repeat = 2 if total_pages >= 2 else 999
    repeated_headers = {line for line, count in top_counts.items() if count >= min_repeat}
    repeated_footers = {line for line, count in bottom_counts.items() if count >= min_repeat}

    cleaned_pages: list[tuple[int, str]] = []
    for page_num, text in pages:
        lines = text.splitlines()

        # Clean top
        start_idx = 0
        while start_idx < len(lines):
            line_s = lines[start_idx].strip()
            if not line_s:
                start_idx += 1
                continue
            if PAGE_NUMBER_PATTERN.match(line_s) or line_s in repeated_headers:
                start_idx += 1
                continue
            break

        # Clean bottom
        end_idx = len(lines)
        while end_idx > start_idx:
            line_s = lines[end_idx - 1].strip()
            if not line_s:
                end_idx -= 1
                continue
            if PAGE_NUMBER_PATTERN.match(line_s) or line_s in repeated_footers:
                end_idx -= 1
                continue
            break

        cleaned_text = "\n".join(lines[start_idx:end_idx]).strip()
        cleaned_pages.append((page_num, cleaned_text))

    return cleaned_pages


def parse_pdf_file(
    file_path: str | Path,
    file_hash: str | None = None,
) -> list[dict[str, Any]]:
    """Parse a single PDF file into page records.

    Args:
        file_path: Path to the PDF file.
        file_hash: Optional precomputed SHA‑256 hash of the file.

    Returns:
        List of dicts with keys: ``file``, ``page``, ``text``, ``file_hash``.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if file_hash is None:
        file_hash = compute_file_sha256(path)

    if PdfReader is None:
        raise ImportError("pypdf library is required to parse PDFs.")
    reader = PdfReader(str(path))
    raw_pages: list[tuple[int, str]] = []

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1  # 1-indexed page number
        page_text = page.extract_text() or ""
        # Skip table of contents pages
        if is_table_of_contents_page(page_text):
            continue
        raw_pages.append((page_num, page_text))

    cleaned_pages = remove_headers_and_footers(raw_pages)

    records: list[dict[str, Any]] = []
    for page_num, text in cleaned_pages:
        records.append({
            "file": str(path),
            "page": page_num,
            "text": text,
            "file_hash": file_hash,
        })

    return records


def parse_folder(
    folder_path: str | Path | None = None,
    config_path: str | Path = "config.toml",
    known_hashes: Mapping[str, str] | Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Parse all PDF files in a folder into page records.

    Args:
        folder_path: Path to the folder containing PDFs. If ``None``, reads from ``config.toml``.
        config_path: Path to configuration file (used if ``folder_path`` is ``None``).
        known_hashes: Optional mapping of file path → hash or collection of known hashes.
                      Files whose computed hash matches will be skipped (per RAG_PIPELINE.md).

    Returns:
        List of records, each containing:
        - ``file`` (str): path to the PDF file
        - ``page`` (int): 1‑indexed page number
        - ``text`` (str): plain text content of the page
        - ``file_hash`` (str): SHA‑256 hash of the file
    """
    if folder_path is None:
        target_dir = get_data_dir_from_config(config_path)
    else:
        target_dir = Path(folder_path)

    if not target_dir.exists():
        raise FileNotFoundError(f"Target folder does not exist: {target_dir}")
    if not target_dir.is_dir():
        raise NotADirectoryError(f"Target path is not a directory: {target_dir}")

    # Discover all PDF files recursively (case‑insensitive extension match)
    pdf_files = sorted(
        p for p in target_dir.rglob("*")
        if p.is_file() and p.suffix.lower() == ".pdf"
    )

    records: list[dict[str, Any]] = []

    for pdf_path in pdf_files:
        current_hash = compute_file_sha256(pdf_path)

        # Skip unchanged files if known hashes are provided
        if known_hashes is not None:
            if isinstance(known_hashes, Mapping):
                if known_hashes.get(str(pdf_path)) == current_hash:
                    continue
            elif current_hash in known_hashes:
                continue

        file_records = parse_pdf_file(pdf_path, file_hash=current_hash)
        records.extend(file_records)

    return records

# Alias to match alternate naming conventions
ingest_folder = parse_folder
