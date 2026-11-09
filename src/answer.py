"""Answer generation module.

Provides a function that builds a prompt from retrieved chunks and calls the
local Ollama server (HTTP on localhost) to obtain a generated answer. The prompt
instructs the model to answer *only* from the supplied chunks, to cite the chunk
IDs for each claim, and to return the exact string ``not found in your documents``
when the answer cannot be derived from the context.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
import tomllib
from typing import Any, List, Dict


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
    """Ensure the Ollama host points to localhost.

    Acceptable forms are ``http://localhost`` (with optional port) or ``http://127.0.0.1``.
    Returns the host string without trailing slash.
    """
    # Strip trailing slash for uniformity
    host = host.rstrip("/")
    # Accept both http and https schemes for localhost
    if host.startswith("http://localhost") or host.startswith("http://127.0.0.1"):
        return host
    raise ValueError("Ollama host must be localhost (127.0.0.1); outbound network calls are prohibited.")


def _build_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    """Construct the prompt that will be sent to Ollama.

    The prompt includes the instruction and each chunk labelled with its ID.
    """
    # Instruction block – exactly as required by the spec
    instruction = (
        "Answer the question using ONLY the provided chunks. "
        "Cite each claim with the chunk ID. "
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


def generate_answer(question: str, chunks: List[Dict[str, Any]], *, config_path: str | Path = "config.toml") -> str:
    """Generate an answer for *question* using retrieved *chunks*.

    The function:
    1. Loads the Ollama ``model`` name and ``host`` from the ``[ollama]`` section of
       ``config.toml`` (or falls back to environment variables ``OLLAMA_MODEL`` and
       ``OLLAMA_HOST``).
    2. Validates that the host points to localhost; otherwise a ``ValueError`` is
       raised.
    3. Builds a prompt that tells the model to answer only from the provided
       chunks and to cite chunk IDs, with a fallback "not found" response.
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

    prompt = _build_prompt(question, chunks)
    payload = json.dumps({"model": model, "prompt": prompt}).encode("utf-8")
    request = urllib.request.Request(f"{host}/api/generate", data=payload, method="POST")
    request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request) as resp:
        resp_data = resp.read()
        resp_json = json.loads(resp_data)
        # Ollama returns the generated text under the 'response' key.
        return resp_json.get("response", "")

# End of answer module
