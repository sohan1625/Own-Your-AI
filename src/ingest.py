"""Document ingestion module.

Responsible for:
- Reading PDF documents from a watched folder path (or configured path from config.toml).
- Computing SHA-256 hash per file for integrity and change detection.
- Extracting plain text page-by-page, recording file path, 1-based page number, text, and file hash.
- Skipping unchanged files when previous hashes are provided (per RAG_PIPELINE.md).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import tomllib
from typing import Any, Mapping, Sequence

try:\n    from pypdf import PdfReader\nexcept ImportError:  # pragma: no cover\n    PdfReader = None  # type: ignore


def load_config(config_path: str | Path = "config.toml") -> dict[str, Any]:
    """Load configuration from a TOML file."""
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(cfg_file, "rb") as f:
        return tomllib.load(f)


def get_data_dir_from_config(config_path: str | Path = "config.toml") -> Path:
    """Read the data/documents directory path from config.toml."""
    config = load_config(config_path)
    data_dir = config.get("paths", {}).get("data_dir")
    if not data_dir:
        raise ValueError("Missing 'paths.data_dir' in configuration file.")
    return Path(data_dir)


def compute_file_sha256(file_path: str | Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_pdf_file(
    file_path: str | Path,
    file_hash: str | None = None,
) -> list[dict[str, Any]]:
    """Parse a single PDF file into page records.

    Args:
        file_path: Path to the PDF file.
        file_hash: Optional precomputed SHA-256 hash of the file.

    Returns:
        List of dicts with keys: file, page, text, file_hash.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if file_hash is None:
        file_hash = compute_file_sha256(path)

    reader = PdfReader(str(path))
    records: list[dict[str, Any]] = []

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1  # 1-indexed page number
        page_text = page.extract_text() or ""
        records.append({
            "file": str(path),
            "page": page_num,
            "text": page_text,
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
        folder_path: Path to the folder containing PDFs. If None, reads from config.toml.
        config_path: Path to configuration file (used if folder_path is None).
        known_hashes: Optional mapping of file path -> hash or collection of known hashes.
                      Files whose computed hash matches will be skipped (per RAG_PIPELINE.md).

    Returns:
        List of records, each containing:
        - file (str): path to the PDF file
        - page (int): 1-indexed page number
        - text (str): plain text content of the page
        - file_hash (str): SHA-256 hash of the file
    """
    if folder_path is None:
        target_dir = get_data_dir_from_config(config_path)
    else:
        target_dir = Path(folder_path)

    if not target_dir.exists():
        raise FileNotFoundError(f"Target folder does not exist: {target_dir}")
    if not target_dir.is_dir():
        raise NotADirectoryError(f"Target path is not a directory: {target_dir}")

    # Discover all PDF files recursively (case-insensitive extension match)
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
