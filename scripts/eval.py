"""Evaluation harness for the document Q&A assistant.

Reads eval/questions.jsonl, runs keyword retrieval (top-20), and evaluates:
- Recall@1, Recall@3, Recall@5 against gold passages (file and page match)
- Mean Reciprocal Rank (MRR)
- For unanswerable questions: hallucination rate (when run with --with-answers)

Saves detailed output to eval/results_keyword.json and prints summary metrics.
"""

from __future__ import annotations

import argparse
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
from src.index import build_bm25_index
from src.retrieve import keyword_retrieve
from src.answer import generate_answer, get_generation_top_k
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


def run_evaluation(
    questions_path: str | Path = "eval/questions.jsonl",
    output_path: str | Path = "eval/results_keyword.json",
    with_answers: bool = False,
    config_path: str | Path = "config.toml",
    bm25_index: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the evaluation harness over questions and return the results dictionary."""
    questions = load_questions(questions_path)

    # Ingest and index once if index not supplied
    if bm25_index is None:
        records = parse_folder(config_path=config_path)
        chunks = chunk_page_records(records, config_path=config_path)
        bm25_index = build_bm25_index(chunks)

    eval_results: List[Dict[str, Any]] = []
    answerable_count = 0
    unanswerable_count = 0
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    sum_reciprocal_rank = 0.0
    hallucination_count = 0

    for q in questions:
        qid = q.get("id", "unknown")
        q_text = q.get("question", "")
        answerable = bool(q.get("answerable", True))
        gold_passages = q.get("gold_passages", [])

        # Retrieval: top 20 chunks
        retrieved_raw = keyword_retrieve(q_text, bm25_index, config_path=config_path)
        retrieved = [] if isinstance(retrieved_raw, str) else retrieved_raw

        first_gold_rank = evaluate_retrieval_match(retrieved, gold_passages)

        hit_1 = bool(first_gold_rank == 1)
        hit_3 = bool(first_gold_rank is not None and first_gold_rank <= 3)
        hit_5 = bool(first_gold_rank is not None and first_gold_rank <= 5)
        rr = (1.0 / first_gold_rank) if first_gold_rank else 0.0

        generated_answer: Optional[str] = None
        hallucinated: Optional[bool] = None

        if with_answers:
            top_k = get_generation_top_k(config_path=config_path)
            gen_chunks = retrieved[:top_k]
            try:
                raw_ans = generate_answer(q_text, gen_chunks, config_path=config_path)
            except Exception as exc:
                raw_ans = f"[Error: {exc}]"
            cit_res = resolve_citations(raw_ans, gen_chunks)
            ans_text, _failed = evaluate_citations_and_answer(cit_res)
            generated_answer = ans_text

            if not answerable:
                hallucinated = bool("not found in your documents" not in ans_text.lower())
                if hallucinated:
                    hallucination_count += 1

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
    }

    full_output = {
        "mode": "keyword",
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

    print("\n" + "=" * 50)
    print(f"{'ID':<6} | {'Answerable':<10} | {'Gold Rank':<10} | {'Status':<6}")
    print("-" * 50)
    for q in questions:
        rank_str = str(q["first_gold_rank"]) if q["first_gold_rank"] is not None else "N/A"
        ans_str = str(q["answerable"])
        print(f"{q['id']:<6} | {ans_str:<10} | {rank_str:<10} | {q['status']:<6}")
    print("=" * 50)

    print("\nSummary Metrics (Keyword Retrieval Top-20):")
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
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation on questions.jsonl.")
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
        default="eval/results_keyword.json",
        help="Path to output results JSON file.",
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
    )
    print_results_table(results)


if __name__ == "__main__":
    main()

