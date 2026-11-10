"""Tests for document ingestion (src/ingest.py).

Uses generated synthetic PDF fixtures without using any real documents or network calls.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from src.ingest import (
    compute_file_sha256,
    get_data_dir_from_config,
    ingest_folder,
    is_table_of_contents_page,
    parse_folder,
    parse_pdf_file,
    remove_headers_and_footers,
)


def create_minimal_pdf_bytes(pages_text: list[str]) -> bytes:
    """Generate a clean, specification-compliant multi-page PDF in memory."""
    total_pages = len(pages_text)
    font_id = 3 + total_pages * 2
    page_ids = [3 + i * 2 for i in range(total_pages)]
    content_ids = [4 + i * 2 for i in range(total_pages)]

    kids_str = " ".join(f"{pid} 0 R" for pid in page_ids)
    catalog = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    pages_obj = (
        f"2 0 obj\n<< /Type /Pages /Kids [{kids_str}] /Count {total_pages} >>\nendobj\n"
    ).encode("latin1")
    objects = [catalog, pages_obj]

    for i, text in enumerate(pages_text):
        # Escape parentheses in stream text if needed
        safe_text = text.replace("(", r"\(").replace(")", r"\)")
        stream_content = f"BT /F1 12 Tf 50 700 Td ({safe_text}) Tj ET".encode("latin1")
        page_obj = (
            f"{page_ids[i]} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_ids[i]} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>\nendobj\n"
        ).encode("latin1")
        content_obj = (
            f"{content_ids[i]} 0 obj\n<< /Length {len(stream_content)} >>\nstream\n".encode("latin1")
            + stream_content
            + b"\nendstream\nendobj\n"
        )
        objects.extend([page_obj, content_obj])

    font_obj = (
        f"{font_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    ).encode("latin1")
    objects.append(font_obj)

    header = b"%PDF-1.4\n"
    body = b""
    offsets = [0]
    curr = len(header)
    for obj in objects:
        offsets.append(curr)
        body += obj
        curr += len(obj)

    xref_offset = curr
    xref = f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode("latin1")
    for off in offsets[1:]:
        xref += f"{off:010d} 00000 n \n".encode("latin1")
    trailer = (
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    ).encode("latin1")

    return header + body + xref + trailer


def write_test_pdf(filepath: Path, pages_text: list[str]) -> Path:
    """Write generated synthetic PDF bytes to disk."""
    pdf_bytes = create_minimal_pdf_bytes(pages_text)
    filepath.write_bytes(pdf_bytes)
    return filepath


class TestIngest(unittest.TestCase):
    """Test suite for PDF ingestion functionality."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_compute_file_sha256(self):
        """Verify that SHA-256 calculation matches hashlib standard."""
        test_file = self.dir_path / "sample.pdf"
        write_test_pdf(test_file, ["Sample Content"])

        computed_hash = compute_file_sha256(test_file)
        expected_hash = hashlib.sha256(test_file.read_bytes()).hexdigest()
        self.assertEqual(computed_hash, expected_hash)

    def test_parse_single_pdf(self):
        """Verify that a multi-page PDF extracts into structured records with 1-based page numbers."""
        test_file = self.dir_path / "lecture.pdf"
        pages_content = [
            "Introduction to B-trees and balanced search structures.",
            "B-tree insertion and node splitting algorithms.",
            "Deletion and borrowing from sibling nodes.",
        ]
        write_test_pdf(test_file, pages_content)

        records = parse_pdf_file(test_file)
        self.assertEqual(len(records), 3)

        expected_hash = hashlib.sha256(test_file.read_bytes()).hexdigest()
        for idx, rec in enumerate(records):
            self.assertEqual(rec["file"], str(test_file))
            self.assertEqual(rec["page"], idx + 1)
            self.assertEqual(rec["file_hash"], expected_hash)
            self.assertIn(pages_content[idx], rec["text"])

    def test_parse_folder(self):
        """Verify that all PDFs in a folder (including subdirectories) are parsed."""
        sub_dir = self.dir_path / "lectures"
        sub_dir.mkdir()

        pdf1 = write_test_pdf(self.dir_path / "doc1.pdf", ["Page 1 of doc 1"])
        pdf2 = write_test_pdf(sub_dir / "doc2.pdf", ["Page 1 of doc 2", "Page 2 of doc 2"])

        # Create a non-PDF file to ensure it is ignored
        (self.dir_path / "notes.txt").write_text("Plain text notes should be ignored by PDF parser")

        records = parse_folder(self.dir_path)
        self.assertEqual(len(records), 3)

        files_in_records = {r["file"] for r in records}
        self.assertIn(str(pdf1), files_in_records)
        self.assertIn(str(pdf2), files_in_records)

    def test_read_paths_from_config(self):
        """Verify that ingest reads the data directory path from config.toml when no path is passed."""
        # Create a dedicated folder for config-driven ingestion
        custom_data_dir = self.dir_path / "my_documents"
        custom_data_dir.mkdir()
        write_test_pdf(custom_data_dir / "test.pdf", ["Content from config path"])

        # Create a temporary config.toml pointing to this directory
        custom_config = self.dir_path / "config.toml"
        # Escape backslashes for TOML on Windows
        escaped_path = str(custom_data_dir).replace("\\", "\\\\")
        custom_config.write_text(f'[paths]\ndata_dir = "{escaped_path}"\n')

        # Test helper function
        loaded_dir = get_data_dir_from_config(custom_config)
        self.assertEqual(loaded_dir, custom_data_dir)

        # Test ingest_folder defaulting to config path
        records = ingest_folder(folder_path=None, config_path=custom_config)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["page"], 1)
        self.assertIn("Content from config path", records[0]["text"])

    def test_skip_unchanged_files(self):
        """Verify that unchanged files are skipped when known hashes are provided."""
        pdf1 = write_test_pdf(self.dir_path / "doc1.pdf", ["Doc 1 text"])
        pdf2 = write_test_pdf(self.dir_path / "doc2.pdf", ["Doc 2 text"])

        hash1 = compute_file_sha256(pdf1)
        hash2 = compute_file_sha256(pdf2)

        # Provide hash for doc1 so only doc2 is parsed
        known_hashes = {str(pdf1): hash1}
        records = parse_folder(self.dir_path, known_hashes=known_hashes)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["file"], str(pdf2))
        self.assertEqual(records[0]["file_hash"], hash2)

    def test_nonexistent_folder_raises(self):
        """Verify that passing a nonexistent folder raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            parse_folder(self.dir_path / "nonexistent_dir")

    def test_empty_folder_returns_empty_list(self):
        """Verify that an empty folder returns an empty list without error."""
        records = parse_folder(self.dir_path)
        self.assertEqual(records, [])

    def test_remove_repeated_headers_and_footers(self):
        """Verify that repeated headers (e.g. document title) and 'Page N' lines are stripped."""
        test_file = self.dir_path / "headers_footers.pdf"
        pages_content = [
            "Core Java (J2SE) - Complete Answers\nPage 1\nIntroduction to Java language.\nConfidential Material",
            "Core Java (J2SE) - Complete Answers\nPage 2\nFeatures and OOP concepts in Java.\nConfidential Material",
        ]
        write_test_pdf(test_file, pages_content)

        records = parse_pdf_file(test_file)
        self.assertEqual(len(records), 2)

        # Verify repeated headers, page numbers, and repeated footers are removed
        self.assertNotIn("Core Java (J2SE) - Complete Answers", records[0]["text"])
        self.assertNotIn("Page 1", records[0]["text"])
        self.assertNotIn("Confidential Material", records[0]["text"])
        self.assertIn("Introduction to Java language.", records[0]["text"])

        self.assertNotIn("Core Java (J2SE) - Complete Answers", records[1]["text"])
        self.assertNotIn("Page 2", records[1]["text"])
        self.assertNotIn("Confidential Material", records[1]["text"])
        self.assertIn("Features and OOP concepts in Java.", records[1]["text"])

    def test_skip_table_of_contents_pages(self):
        """Verify that table of contents pages with dotted leader lines are skipped."""
        test_file = self.dir_path / "toc_doc.pdf"
        pages_content = [
            "Title and overview of the course.",
            "Contents\n1. Overview . . . . . . . . . 1\n2. Fundamentals . . . . . . . . 5\n3. Collections . . . . . . . . 12\n4. Multithreading . . . . . . . 20\n5. Advanced . . . . . . . . . . 30",
            "Chapter 1: The real substance starts here.",
        ]
        write_test_pdf(test_file, pages_content)

        records = parse_pdf_file(test_file)
        # Page 2 (TOC) must be skipped, leaving 2 records with pages 1 and 3
        self.assertEqual(len(records), 2)
        self.assertEqual([r["page"] for r in records], [1, 3])
        self.assertIn("Title and overview of the course.", records[0]["text"])
        self.assertIn("Chapter 1: The real substance starts here.", records[1]["text"])


if __name__ == "__main__":
    unittest.main()

