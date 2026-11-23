"""Evaluation harness for the document Q&A assistant.

Reads eval/questions.jsonl, runs keyword retrieval (top-20), and evaluates:
- Recall@1, Recall@3, Recall@5 against gold passages (file and page match)
- Mean Reciprocal Rank (MRR)
- For unanswerable questions: hallucination rate (when run with --with-answers)

Saves detailed output to eval/results_keyword.json and prints summary metrics.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import uuid
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ingest import parse_folder
from src.chunk import chunk_page_records
from src.index import build_bm25_index, build_chunk_embeddings
from src.retrieve import (
    keyword_retrieve,
    vector_retrieve,
    get_retrieval_params,
    load_config,
)
from src.answer import generate_answer, get_generation_top_k
import src.answer as answer_module
from src.cite import resolve_citations
from scripts.demo import evaluate_citations_and_answer


def load_questions(jsonl_path: str | Path) -> List[Dict[str, Any]]:
    """Load evaluation questions from a JSON Lines file."""
    path = Path(jsonl_path)
    if not path.is_file():
        raise FileNotFoundError(f"Questions file not found: {jsonl_path}")
    questions: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def evaluate_retrieval_match(
    retrieved_chunks: List[Dict[str, Any]],
    gold_passages: List[Dict[str, Any]],
) -> Optional[int]:
    """Find the 1-based rank of the first retrieved chunk matching any gold passage.

    Matching is done by filename (case-insensitive) and page number.
    Returns the rank (1..N) or None if no match is found.
    """
    if not gold_passages:
        return None

    gold_set = {(Path(gp["file"]).name.lower(), int(gp["page"])) for gp in gold_passages}

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        chunk_file = Path(chunk.get("file", "")).name.lower()
        chunk_page = int(chunk.get("page", -1))
        if (chunk_file, chunk_page) in gold_set:
            return rank

    return None


def get_rrf_k(config_path: str | Path = "config.toml") -> int:
    """Read the RRF constant k from config.toml [fusion] section."""
    try:
        cfg = load_config(config_path)
        fusion_cfg = cfg.get("fusion", {})
        k = int(fusion_cfg.get("rrf_k", 60))
        if k <= 0:
            raise ValueError("rrf_k must be a positive integer")
        return k
    except Exception:
        return 60


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """Fuse multiple ranked lists of chunk dictionaries using Reciprocal Rank Fusion (RRF).

    Formula: RRF_score(d) = sum_{list} 1 / (k + rank_list(d))
    where rank_list(d) is the 1-based rank of chunk d in that list.
    """
    scores: Dict[Any, float] = defaultdict(float)
    chunk_map: Dict[Any, Dict[str, Any]] = {}

    for r_list in ranked_lists:
        for rank, chunk in enumerate(r_list, start=1):
            key = chunk.get("chunk_id")
            if not key:
                key = (
                    Path(chunk.get("file", "")).name.lower(),
                    int(chunk.get("page", -1)),
                    int(chunk.get("start_char", 0)),
                )
            scores[key] += 1.0 / (k + rank)
            if key not in chunk_map:
                chunk_map[key] = chunk

    # Sort chunks descending by RRF score; stable sort preserves initial order on tie
    sorted_keys = sorted(chunk_map.keys(), key=lambda k_val: scores[k_val], reverse=True)
    return [chunk_map[k_val] for k_val in sorted_keys[:top_n]]


_CHUNK_ID_NAMESPACE = uuid.UUID("6f1c2a3e-6b0e-4f1e-9d55-0c7e4a1b2c3d")


def assign_stable_chunk_ids(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Replace random chunk IDs with IDs derived from file, page and character span.

    Chunk IDs appear in the prompt, so random IDs make every run send a different
    prompt and the answers cannot repeat even at temperature 0.
    """
    for chunk in chunks:
        key = f"{Path(chunk.get('file', '')).name}|{chunk.get('page')}|{chunk.get('start_char')}|{chunk.get('end_char')}"
        chunk["chunk_id"] = str(uuid.uuid5(_CHUNK_ID_NAMESPACE, key))
    return chunks


def run_evaluation(
    questions_path: str | Path = "eval/questions.jsonl",
    output_path: Optional[str | Path] = None,
    with_answers: bool = False,
    config_path: str | Path = "config.toml",
    bm25_index: Optional[Dict[str, Any]] = None,
    retriever: str = "keyword",
    chunks: Optional[List[Dict[str, Any]]] = None,
    vectors: Optional[List[List[float]]] = None,
) -> Dict[str, Any]:
    """Run the evaluation harness over questions and return the results dictionary."""
    if retriever not in ("keyword", "vector", "hybrid", "langchain"):
        raise ValueError(
            f"Unknown retriever: {retriever}. Expected 'keyword', 'vector', 'hybrid', or 'langchain'."
        )
    if retriever == "langchain":
        with langchain_store(chunks, vectors, config_path=config_path) as (lc_chunks, lc_vectors, doc_ids):
            return _run_evaluation(
                questions_path, output_path, with_answers, config_path, None, retriever,
                lc_chunks, lc_vectors, doc_ids,
            )
    return _run_evaluation(
        questions_path, output_path, with_answers, config_path, bm25_index, retriever, chunks, vectors, None
    )


@contextmanager
def langchain_store(
    chunks: Optional[List[Dict[str, Any]]],
    vectors: Optional[List[List[float]]],
    *,
    config_path: str | Path = "config.toml",
):
    """Load the eval chunks and embeddings into a throwaway PostgreSQL schema.

    The LangChain engine reads vectors through ``db.vector_search``, so the chunks
    must be in the database. The schema is dropped afterwards; the main schema is
    never touched.
    """
    import os

    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    from src import db

    if chunks is None:
        chunks = assign_stable_chunk_ids(chunk_page_records(parse_folder(config_path=config_path), config_path=config_path))
    if vectors is None:
        vectors = build_chunk_embeddings(chunks, config_path=config_path)
    base_url = os.environ["DATABASE_URL"]
    schema = f"eval_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(base_url, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    os.environ["DATABASE_URL"] = make_conninfo(base_url, options=f"-c search_path={schema}")
    try:
        db.initialize_database()
        doc_ids: List[str] = []
        by_file: Dict[str, List[int]] = defaultdict(list)
        for i, chunk in enumerate(chunks):
            by_file[chunk["file"]].append(i)
        for file_name, indexes in by_file.items():
            doc_id = str(uuid.uuid5(_CHUNK_ID_NAMESPACE, f"doc|{Path(file_name).name}"))
            db.insert_document(
                {
                    "id": doc_id,
                    "filename": Path(file_name).name,
                    "sha256": chunks[indexes[0]].get("file_hash") or doc_id,
                    "pages": len({chunks[i]["page"] for i in indexes}),
                },
                [chunks[i] for i in indexes],
                [vectors[i] for i in indexes],
            )
            doc_ids.append(doc_id)
        yield chunks, vectors, doc_ids
    finally:
        os.environ["DATABASE_URL"] = base_url
        with psycopg.connect(base_url, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def _run_evaluation(
    questions_path: str | Path,
    output_path: Optional[str | Path],
    with_answers: bool,
    config_path: str | Path,
    bm25_index: Optional[Dict[str, Any]],
    retriever: str,
    chunks: Optional[List[Dict[str, Any]]],
    vectors: Optional[List[List[float]]],
    doc_ids: Optional[List[str]],
) -> Dict[str, Any]:

    if output_path is None:
        output_path = f"eval/results_{retriever}.json"

    questions = load_questions(questions_path)

    # Ingest and chunk once if chunks not supplied
    if chunks is None:
        if bm25_index is not None and "chunks" in bm25_index:
            chunks = bm25_index["chunks"]
        else:
            records = parse_folder(config_path=config_path)
            chunks = assign_stable_chunk_ids(chunk_page_records(records, config_path=config_path))

    # BM25 index built once if needed
    if bm25_index is None and retriever in ("keyword", "hybrid"):
        bm25_index = build_bm25_index(chunks)

    # Embed chunks once and cache in memory if needed
    if vectors is None and (retriever in ("vector", "hybrid") or with_answers):
        try:
            vectors = build_chunk_embeddings(chunks, config_path=config_path)
        except Exception:
            vectors = None

    rrf_k = get_rrf_k(config_path)
    retrieval_params = get_retrieval_params(config_path)
    top_n = retrieval_params["top_n"]
    lc_engine = None
    if retriever == "langchain":
        from src.lc_retrieve import LangChainHybrid

        lc_engine = LangChainHybrid(chunks, doc_ids or [], top_n=top_n, rrf_k=rrf_k, config_path=config_path)

    eval_results: List[Dict[str, Any]] = []
    answerable_count = 0
    unanswerable_count = 0
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    sum_reciprocal_rank = 0.0
    hallucination_count = 0
    wrongly_refused_count = 0

    for q in questions:
        qid = q.get("id", "unknown")
        q_text = q.get("question", "")
        answerable = bool(q.get("answerable", True))
        gold_passages = q.get("gold_passages", [])

        # Retrieval based on selected retriever strategy
        scored_chunks = []
        vector_results = []
        if retriever == "keyword":
            retrieved_raw = keyword_retrieve(q_text, bm25_index, config_path=config_path)
            retrieved = [] if isinstance(retrieved_raw, str) else retrieved_raw
        elif retriever == "vector":
            scored_chunks = vector_retrieve(q_text, chunks, vectors, top_n=top_n, config_path=config_path)
            retrieved = [chunk for chunk, _ in scored_chunks]
        elif retriever == "hybrid":
            bm25_raw = keyword_retrieve(q_text, bm25_index, config_path=config_path)
            bm25_chunks = [] if isinstance(bm25_raw, str) else bm25_raw
            vector_results = vector_retrieve(q_text, chunks, vectors, top_n=top_n, config_path=config_path)
            vector_chunks = [chunk for chunk, _ in vector_results]
            retrieved = reciprocal_rank_fusion([bm25_chunks, vector_chunks], k=rrf_k, top_n=top_n)
        elif retriever == "langchain":
            retrieved, lc_top_vector_score = lc_engine.retrieve(q_text)

        first_gold_rank = evaluate_retrieval_match(retrieved, gold_passages)

        hit_1 = bool(first_gold_rank == 1)
        hit_3 = bool(first_gold_rank is not None and first_gold_rank <= 3)
        hit_5 = bool(first_gold_rank is not None and first_gold_rank <= 5)
        rr = (1.0 / first_gold_rank) if first_gold_rank else 0.0

        generated_answer: Optional[str] = None
        hallucinated: Optional[bool] = None
        wrongly_refused: Optional[bool] = False if (with_answers and answerable) else None

        if with_answers:
            top_k = get_generation_top_k(config_path=config_path)
            gen_chunks = retrieved[:top_k]
            top_vector_score: Optional[float] = None
            if retriever == "langchain":
                top_vector_score = lc_top_vector_score
            elif retriever == "hybrid" and vector_results:
                top_vector_score = vector_results[0][1]
            elif retriever == "vector" and scored_chunks:
                top_vector_score = scored_chunks[0][1]
            elif vectors is not None and chunks is not None:
                v_res = vector_retrieve(q_text, chunks, vectors, top_n=1, config_path=config_path)
                top_vector_score = v_res[0][1] if v_res else None

            try:
                if qid in {"q04", "q05"}:
                    original_normalize_answer = answer_module.normalize_answer

                    def print_raw_response(raw_response: str, chunk_ids: List[str]) -> str:
                        print(f"[{qid} raw model response before normalize_answer]\n{raw_response}", flush=True)
                        return original_normalize_answer(raw_response, chunk_ids)

                    answer_module.normalize_answer = print_raw_response
                    try:
                        raw_ans = generate_answer(
                            q_text, gen_chunks, config_path=config_path, top_score=top_vector_score
                        )
                    finally:
                        answer_module.normalize_answer = original_normalize_answer
                else:
                    raw_ans = generate_answer(
                        q_text, gen_chunks, config_path=config_path, top_score=top_vector_score
                    )
            except Exception as exc:
                raw_ans = f"[Error: {exc}]"
            cit_res = resolve_citations(raw_ans, gen_chunks)
            ans_text, _failed = evaluate_citations_and_answer(cit_res)
            generated_answer = ans_text

            if not answerable:
                hallucinated = bool("not found in your documents" not in ans_text.lower())
                if hallucinated:
                    hallucination_count += 1
            elif ans_text.strip().rstrip(".").strip().lower() == "not found in your documents":
                wrongly_refused = True
                wrongly_refused_count += 1

        if answerable:
            answerable_count += 1
            if hit_1:
                hits_at_1 += 1
            if hit_3:
                hits_at_3 += 1
            if hit_5:
                hits_at_5 += 1
            sum_reciprocal_rank += rr
            status = "PASS" if hit_5 else "FAIL"
        else:
            unanswerable_count += 1
            if with_answers:
                status = "FAIL" if hallucinated else "PASS"
            else:
                status = "PASS"

        eval_results.append({
            "id": qid,
            "question": q_text,
            "answerable": answerable,
            "first_gold_rank": first_gold_rank,
            "hit_at_1": hit_1,
            "hit_at_3": hit_3,
            "hit_at_5": hit_5,
            "reciprocal_rank": rr,
            "status": status,
            "generated_answer": generated_answer,
            "hallucinated": hallucinated,
            "wrongly_refused": wrongly_refused,
        })

    recall_1 = (hits_at_1 / answerable_count) if answerable_count else 0.0
    recall_3 = (hits_at_3 / answerable_count) if answerable_count else 0.0
    recall_5 = (hits_at_5 / answerable_count) if answerable_count else 0.0
    mrr = (sum_reciprocal_rank / answerable_count) if answerable_count else 0.0

    hallucination_rate = None
    if with_answers and unanswerable_count:
        hallucination_rate = hallucination_count / unanswerable_count

    summary = {
        "total_questions": len(questions),
        "answerable_questions": answerable_count,
        "unanswerable_questions": unanswerable_count,
        "recall_at_1": round(recall_1, 4),
        "recall_at_3": round(recall_3, 4),
        "recall_at_5": round(recall_5, 4),
        "mrr": round(mrr, 4),
        "hallucination_rate": round(hallucination_rate, 4) if hallucination_rate is not None else None,
        # Answerable questions answered with the refusal (threshold or model).
        "answerable_wrongly_refused": wrongly_refused_count if with_answers else None,
    }

    full_output = {
        "mode": retriever,
        "with_answers": with_answers,
        "summary": summary,
        "questions": eval_results,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    return full_output


def print_results_table(eval_data: Dict[str, Any]) -> None:
    """Print a formatted results table and summary metrics."""
    questions = eval_data["questions"]
    summary = eval_data["summary"]
    mode_name = eval_data.get("mode", "keyword").capitalize()

    print("\n" + "=" * 50)
    print(f"{'ID':<6} | {'Answerable':<10} | {'Gold Rank':<10} | {'Status':<6}")
    print("-" * 50)
    for q in questions:
        rank_str = str(q["first_gold_rank"]) if q["first_gold_rank"] is not None else "N/A"
        ans_str = str(q["answerable"])
        print(f"{q['id']:<6} | {ans_str:<10} | {rank_str:<10} | {q['status']:<6}")
    print("=" * 50)

    print(f"\nSummary Metrics ({mode_name} Retrieval Top-20):")
    print(f"  Total Questions:        {summary['total_questions']}")
    print(f"  Answerable Questions:   {summary['answerable_questions']}")
    print(f"  Unanswerable Questions: {summary['unanswerable_questions']}")
    print(f"  Recall@1:               {summary['recall_at_1']:.4f}")
    print(f"  Recall@3:               {summary['recall_at_3']:.4f}")
    print(f"  Recall@5:               {summary['recall_at_5']:.4f}")
    print(f"  MRR:                    {summary['mrr']:.4f}")
    if summary['hallucination_rate'] is not None:
        print(f"  Hallucination Rate:     {summary['hallucination_rate']:.4f}")
    else:
        print("  Hallucination Rate:     N/A (retrieval-only mode; use --with-answers to evaluate)")
    if summary.get('answerable_wrongly_refused') is not None:
        print(f"  Answerable refused:     {summary['answerable_wrongly_refused']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation on questions.jsonl.")
    parser.add_argument(
        "--retriever",
        choices=["keyword", "vector", "hybrid", "langchain"],
        default="keyword",
        help="Retrieval strategy: keyword (default), vector, hybrid, or langchain (hybrid via LangChain).",
    )
    parser.add_argument(
        "--with-answers",
        action="store_true",
        help="Also generate answers via Ollama and measure hallucination on unanswerables.",
    )
    parser.add_argument(
        "--questions",
        default="eval/questions.jsonl",
        help="Path to questions JSONL file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output results JSON file (default: eval/results_<retriever>.json).",
    )
    parser.add_argument(
        "--config",
        default="config.toml",
        help="Path to configuration TOML file.",
    )
    args = parser.parse_args()

    results = run_evaluation(
        questions_path=args.questions,
        output_path=args.output,
        with_answers=args.with_answers,
        config_path=args.config,
        retriever=args.retriever,
    )
    print_results_table(results)


if __name__ == "__main__":
    main()
