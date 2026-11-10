"""Answer generation module.

Provides a function that builds a prompt from retrieved chunks and calls the
local Ollama server (HTTP on localhost) to obtain a generated answer. The prompt
instructs the model to answer *only* from the supplied chunks, to cite the chunk
IDs for each claim, and to return the exact string ``not found in your documents``
when the answer cannot be derived from the context.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
import tomllib
from typing import Any, List, Dict
from urllib.parse import urlparse


def _load_config(config_path: str | Path = "config.toml") -> dict[str, Any]:
    """Load the TOML configuration file.

    Raises ``FileNotFoundError`` if the file does not exist.
    """
    cfg_file = Path(config_path)
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(cfg_file, "rb") as f:
        return tomllib.load(f)


def _validate_host(host: str) -> str:
    """Ensure the Ollama host is localhost or 127.0.0.1 (exact hostname match)."""
    host = host.rstrip("/")
    parsed = urlparse(host)
    if parsed.scheme in ("http", "https") and parsed.hostname in ("localhost", "127.0.0.1"):
        return host
    raise ValueError("Ollama host must be localhost (127.0.0.1); outbound network calls are prohibited.")

def _build_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    """Construct the prompt that will be sent to Ollama.

    The prompt includes the instruction and each chunk labelled with its ID.
    """
    # Instruction block – exactly as required by the spec
    instruction = (
        "Answer the question using ONLY the provided chunks. "
        "Write a complete sentence that answers the question. "
        "End every sentence that uses information from a chunk with that chunk's ID "
        "in square brackets, like [chunk-id]. "
        "If the chunks do not contain the answer, reply exactly "
        "'not found in your documents'."
    )
    # Add each chunk
    chunk_parts = []
    for chunk in chunks:
        # Expected keys: chunk_id (or id) and text
        cid = chunk.get("chunk_id") or chunk.get("id") or "UNKNOWN"
        text = chunk.get("text", "")
        chunk_parts.append(f"Chunk {cid}: {text}")
    chunks_section = "\n".join(chunk_parts)
    return f"{instruction}\n\nQuestion: {question}\n\nContext:\n{chunks_section}"


def get_generation_top_k(config_path: str | Path = "config.toml") -> int:
    """Read top_k from config.toml under [generation]."""
    try:
        cfg = _load_config(config_path)
        gen_cfg = cfg.get("generation", {})
        val = gen_cfg.get("top_k", gen_cfg.get("GENERATION_TOP_K", gen_cfg.get("generation_top_k", 1)))
        return int(val)
    except Exception:
        return 1


def normalize_answer(raw_text: str, chunk_ids: List[str]) -> str:
    """Normalise a raw model response before citation resolution.

    Three transformations are applied in order:

    1. **Bracket normalisation** – every occurrence of a known *chunk_id* that
       appears without square brackets (bare, or wrapped in parentheses) is
       rewritten as ``[chunk-id]``.
    2. **Label truncation** – the text is cut at the first line whose content
       starts with ``Question:`` or ``Answer:`` (case-insensitive).  Everything
       from that line onwards is discarded.
    3. **Whitespace stripping** – leading and trailing whitespace is removed
       from the result.

    Parameters
    ----------
    raw_text:
        The unmodified string returned by the language model.
    chunk_ids:
        The list of chunk ID strings that were included in the prompt.  Only
        IDs from this list are considered for bracket normalisation.

    Returns
    -------
    str
        The cleaned answer string.
    """
    text = raw_text

    # 1. Rewrite every known chunk-id to [chunk-id].
    #    Match: already-bracketed [id], parenthesised (id), or bare id.
    #    We replace all three forms with [id] so the result is idempotent.
    for cid in chunk_ids:
        # Escape the id for use inside a regex pattern.
        escaped = re.escape(cid)
        # Pattern: optional leading bracket/paren, the id, optional trailing bracket/paren.
        # Covers: [cid], (cid), cid (bare).
        pattern = r"[\[\(]?" + escaped + r"[\]\)]?"
        replacement = f"[{cid}]"
        text = re.sub(pattern, replacement, text)

    # 2. Cut at the first line starting with "Question:" or "Answer:" (case-insensitive).
    lines = text.splitlines()
    cut_lines: List[str] = []
    for line in lines:
        if re.match(r"^\s*(question|answer)\s*:", line, flags=re.IGNORECASE):
            break
        cut_lines.append(line)
    text = "\n".join(cut_lines)

    # 3. Strip leading/trailing whitespace.
    return text.strip()


def generate_answer(question: str, chunks: List[Dict[str, Any]], *, config_path: str | Path = "config.toml") -> str:
    """Generate an answer for *question* using retrieved *chunks*.

    The function:
    1. Loads the Ollama ``model`` name and ``host`` from the ``[ollama]`` section of
       ``config.toml`` (or falls back to environment variables ``OLLAMA_MODEL`` and
       ``OLLAMA_HOST``).
    2. Validates that the host points to localhost; otherwise a ``ValueError`` is
       raised.
    3. Builds a prompt that tells the model to answer only from the provided
       chunks (limited to top GENERATION_TOP_K chunks) and to cite chunk IDs, with a fallback "not found" response.
    4. Sends a POST request to ``<host>/api/generate`` using the standard library
       ``urllib.request`` and returns the generated answer text.
    """
    cfg = _load_config(config_path)
    ollama_cfg = cfg.get("ollama", {})
    model = ollama_cfg.get("model") or os.getenv("OLLAMA_MODEL")
    host = ollama_cfg.get("host") or os.getenv("OLLAMA_HOST")
    if not model or not host:
        raise ValueError("Ollama model and host must be specified in config.toml or environment.")
    host = _validate_host(host)

    top_k = get_generation_top_k(config_path)
    chunks_to_use = chunks[:top_k]

    prompt = _build_prompt(question, chunks_to_use)
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    request = urllib.request.Request(f"{host}/api/generate", data=payload, method="POST")
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request) as resp:
            resp_data = resp.read()
            resp_json = json.loads(resp_data)
            # Ollama returns the generated text under the 'response' key.
            raw_response = resp_json.get("response", "")
            # Collect the chunk IDs that were actually sent in the prompt.
            chunk_ids = [
                c.get("chunk_id") or c.get("id") or "UNKNOWN"
                for c in chunks_to_use
            ]
            return normalize_answer(raw_response, chunk_ids)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama request failed with HTTP {exc.code}: {body}"
        ) from exc

# End of answer module
