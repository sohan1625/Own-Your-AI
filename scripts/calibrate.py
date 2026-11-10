"""Calibration script for the RAG pipeline.

For each question in eval/calibration.jsonl, runs BM25, vector, and hybrid
(RRF) retrieval and prints the top score and rank-1 chunk metadata for each
retriever.  No threshold decision is printed – this is raw score data for
threshold calibration.

Usage:
    python scripts/calibrate.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ingest import parse_folder
from src.chunk import chunk_page_records
from src.index import build_bm25_index, build_chunk_embeddings, bm25_search
from src.retrieve import (
    vector_retrieve,
    get_retrieval_params,
    load_config,
)
# Reuse RRF helpers from the eval harness – same codebase, no duplication.
from scripts.eval import get_rrf_k, reciprocal_rank_fusion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_calibration(path: str | Path) -> List[Dict[str, Any]]:
    """Load calibration questions from a JSON Lines file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _chunk_label(chunk: Optional[Dict[str, Any]]) -> tuple[str, int]:
    """Return (basename, page) for a chunk, or ('—', 0) if None."""
    if chunk is None:
        return "—", 0
    file_name = Path(chunk.get("file", "")).name or "—"
    page = int(chunk.get("page", 0))
    return file_name, page


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()
    top_n = get_retrieval_params()["top_n"]
    rrf_k = get_rrf_k()

    # ------------------------------------------------------------------
    # 1. Ingest + chunk (identical to demo.py and eval.py)
    # ------------------------------------------------------------------
    print("Loading documents…", flush=True)
    try:
        records = parse_folder()
    except Exception as exc:
        print(f"[Error] Ingestion failed: {exc}", file=sys.stderr)
        records = []

    chunks = chunk_page_records(records) if records else []
    print(f"  {len(chunks)} chunks loaded.", flush=True)

    # ------------------------------------------------------------------
    # 2. Build BM25 index
    # ------------------------------------------------------------------
    bm25_index = build_bm25_index(chunks)

    # ------------------------------------------------------------------
    # 3. Build vector index (embed all chunks once, cache in memory)
    # ------------------------------------------------------------------
    print("Building vector index (embedding chunks)…", flush=True)
    try:
        vectors = build_chunk_embeddings(chunks)
        print(f"  {len(vectors)} vectors built.", flush=True)
    except Exception as exc:
        print(f"[Warning] Vector embedding failed: {exc}. Vector scores will be N/A.", file=sys.stderr)
        vectors = []

    # ------------------------------------------------------------------
    # 4. Load calibration questions
    # ------------------------------------------------------------------
    cal_path = project_root / "eval" / "calibration.jsonl"
    questions = load_calibration(cal_path)

    # ------------------------------------------------------------------
    # 5. Print header
    # ------------------------------------------------------------------
    print()
    header = (
        f"{'ID':<5}  {'ANS':<5}  "
        f"{'BM25':>8}  {'VEC':>8}  {'RRF':>8}  "
        f"{'RANK-1 FILE':<40}  {'PAGE':>4}"
    )
    print(header)
    print("-" * len(header))

    # ------------------------------------------------------------------
    # 6. Run retrievers for each question
    # ------------------------------------------------------------------
    for q in questions:
        qid: str = q["id"]
        question: str = q["question"]
        answerable: bool = q["answerable"]

        # --- BM25 ---
        bm25_results = bm25_search(question, bm25_index, top_n=top_n)
        if bm25_results:
            bm25_top_chunk, bm25_top_score = bm25_results[0]
            bm25_ranked = [c for c, _ in bm25_results]
        else:
            bm25_top_chunk, bm25_top_score = None, 0.0
            bm25_ranked = []

        # --- Vector ---
        vec_top_score: float = 0.0
        vec_top_chunk: Optional[Dict[str, Any]] = None
        vec_ranked: List[Dict[str, Any]] = []
        if vectors:
            try:
                vec_results = vector_retrieve(question, chunks, vectors, top_n=top_n)
                if vec_results:
                    vec_top_chunk, vec_top_score = vec_results[0]
                    vec_ranked = [c for c, _ in vec_results]
            except Exception as exc:
                print(f"  [Warning] Vector retrieve failed for {qid}: {exc}", file=sys.stderr)

        # --- Hybrid RRF ---
        rrf_top_score: float = 0.0
        rrf_top_chunk: Optional[Dict[str, Any]] = None
        ranked_lists = [lst for lst in [bm25_ranked, vec_ranked] if lst]
        if ranked_lists:
            fused = reciprocal_rank_fusion(ranked_lists, k=rrf_k, top_n=top_n)
            if fused:
                rrf_top_chunk = fused[0]
                # Compute the actual RRF score for the top chunk.
                from collections import defaultdict
                scores: Dict[Any, float] = defaultdict(float)
                for r_list in ranked_lists:
                    for rank, chunk in enumerate(r_list, start=1):
                        key = chunk.get("chunk_id") or (
                            Path(chunk.get("file", "")).name.lower(),
                            int(chunk.get("page", -1)),
                            int(chunk.get("start_char", 0)),
                        )
                        scores[key] += 1.0 / (rrf_k + rank)
                top_key = rrf_top_chunk.get("chunk_id") or (
                    Path(rrf_top_chunk.get("file", "")).name.lower(),
                    int(rrf_top_chunk.get("page", -1)),
                    int(rrf_top_chunk.get("start_char", 0)),
                )
                rrf_top_score = scores[top_key]

        # Pick the rank-1 chunk from BM25 (most interpretable baseline file/page)
        display_chunk = bm25_top_chunk
        file_name, page = _chunk_label(display_chunk)

        ans_str = "Y" if answerable else "N"
        bm25_str = f"{bm25_top_score:.4f}" if bm25_top_score else "   0.0000"
        vec_str = f"{vec_top_score:.4f}" if vec_top_score else "   0.0000"
        rrf_str = f"{rrf_top_score:.4f}" if rrf_top_score else "   0.0000"
        file_str = file_name[:40]

        print(
            f"{qid:<5}  {ans_str:<5}  "
            f"{bm25_str:>8}  {vec_str:>8}  {rrf_str:>8}  "
            f"{file_str:<40}  {page:>4}"
        )


if __name__ == "__main__":
    main()

