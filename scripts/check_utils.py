"""Shared helpers for the privacy checks (deletion_check.py, offline_check.py).

Each check runs the real app (real PostgreSQL, real Ollama models) inside a
temporary database schema and a temporary upload folder, both removed at the
end, so the user's documents and chats are never touched.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import psycopg
from dotenv import dotenv_values
from psycopg import sql
from psycopg.conninfo import make_conninfo

ROOT = Path(__file__).resolve().parents[1]


def database_url() -> str:
    url = os.environ.get("DATABASE_URL") or dotenv_values(ROOT / ".env").get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set. Copy .env.example to .env first.")
    return url


@contextmanager
def isolated_app() -> Iterator[tuple]:
    """Yield ``(client, api, schema_url)`` for the real app in a throwaway schema."""
    base_url = database_url()
    schema = f"check_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(base_url, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    schema_url = make_conninfo(base_url, options=f"-c search_path={schema}")
    previous_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = schema_url
    upload_dir = Path(tempfile.mkdtemp(prefix="check_uploads_"))

    from fastapi.testclient import TestClient

    from src import api

    previous_upload_dir = api.UPLOAD_DIR
    api.UPLOAD_DIR = upload_dir
    try:
        with TestClient(api.app) as client:
            yield client, api, schema_url
    finally:
        api.UPLOAD_DIR = previous_upload_dir
        if previous_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous_url
        with psycopg.connect(base_url, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
        shutil.rmtree(upload_dir, ignore_errors=True)


def make_pdf(text: str) -> bytes:
    """A minimal one-page PDF containing ``text`` (ASCII)."""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 50 740 Td ({escaped}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii"))
    return bytes(pdf)


class Report:
    """Collects PASS / FAIL / NOTE lines and prints them as they happen."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def check(self, ok: bool, text: str) -> bool:
        self._add("PASS" if ok else "FAIL", text)
        return ok

    def note(self, text: str) -> None:
        self._add("NOTE", text)

    def _add(self, status: str, text: str) -> None:
        self.lines.append((status, text))
        print(f"[{status}] {text}", flush=True)

    @property
    def failed(self) -> int:
        return sum(status == "FAIL" for status, _ in self.lines)
