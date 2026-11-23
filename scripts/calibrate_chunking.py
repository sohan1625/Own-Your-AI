"""Choose the semantic-chunking percentile on eval/calibration.jsonl only.

For each candidate percentile, chunks the corpus with the semantic strategy, runs
hybrid retrieval (BM25 + vector, RRF) for the calibration questions, and prints
Recall@1/@3/@5 and MRR on the answerable ones (gold = page), the chunk count, and
how the not-found threshold behaves on the new chunks (answerable questions whose
top vector score is below it, unanswerable ones at or above it). The fixed strategy
is printed first as the reference. eval/questions.jsonl is never read.

    python -m scripts.calibrate_chunking
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from scripts.eval import assign_stable_chunk_ids, get_rrf_k, reciprocal_rank_fusion
from src.chunk import _fixed_chunks, semantic_chunk_page_records
from src.index import build_bm25_index, embed_text, get_embedding_prefix
from src.ingest import parse_folder
from src.retrieve import cosine_similarity, get_retrieval_params, keyword_retrieve

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [5, 10, 15, 25, 35, 50, 65, 80]


def _load(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    config = ROOT / "config.toml"
    with open(config, "rb") as f:
        import tomllib

        threshold = float(tomllib.load(f)["thresholds"]["not_found_threshold"])
    questions = _load(ROOT / "eval" / "calibration.jsonl")
    records = parse_folder(config_path=config)
    doc_prefix = get_embedding_prefix("document", config_path=config)
    query_prefix = get_embedding_prefix("query", config_path=config)
    cache: Dict[str, List[float]] = {}

    def embed_doc(text: str) -> List[float]:
        if text not in cache:
            cache[text] = embed_text(text, prefix=doc_prefix, config_path=config)
        return cache[text]

    query_vectors = {q["id"]: embed_text(q["question"], prefix=query_prefix, config_path=config) for q in questions}
    top_n = get_retrieval_params(config)["top_n"]
    rrf_k = get_rrf_k(config)

    def evaluate(label: str, chunks: List[Dict[str, Any]]) -> None:
        chunks = assign_stable_chunk_ids(chunks)
        index = build_bm25_index(chunks)
        vectors = [embed_doc(chunk["text"]) for chunk in chunks]
        hits = {1: 0, 3: 0, 5: 0}
        rr_sum = 0.0
        answerable = 0
        low_answerable = 0
        high_unanswerable = 0
        for q in questions:
            bm25 = keyword_retrieve(q["question"], index, config_path=config)
            bm25 = [] if isinstance(bm25, str) else bm25
            scored = sorted(
                ((c, cosine_similarity(query_vectors[q["id"]], v)) for c, v in zip(chunks, vectors)),
                key=lambda pair: pair[1],
                reverse=True,
            )[:top_n]
            fused = reciprocal_rank_fusion([bm25, [c for c, _ in scored]], k=rrf_k, top_n=top_n)
            top_score = scored[0][1] if scored else 0.0
            if q["answerable"]:
                answerable += 1
                rank = next((i for i, c in enumerate(fused, 1) if int(c["page"]) == int(q["gold_page"])), None)
                for k in hits:
                    hits[k] += bool(rank and rank <= k)
                rr_sum += 1.0 / rank if rank else 0.0
                low_answerable += top_score < threshold
            else:
                high_unanswerable += top_score >= threshold
        print(
            f"{label:<12} chunks={len(chunks):<4} R@1={hits[1]/answerable:.4f} R@3={hits[3]/answerable:.4f} "
            f"R@5={hits[5]/answerable:.4f} MRR={rr_sum/answerable:.4f} "
            f"answerable<thr={low_answerable} unanswerable>=thr={high_unanswerable}",
            flush=True,
        )

    evaluate("fixed", _fixed_chunks(records, config_path=config))
    for pct in CANDIDATES:
        evaluate(
            f"semantic p{pct}",
            semantic_chunk_page_records(records, config_path=config, embed=embed_doc, percentile_value=pct),
        )


if __name__ == "__main__":
    main()
