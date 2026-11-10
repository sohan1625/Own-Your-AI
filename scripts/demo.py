"""M1 end-to-end demo script.

This script runs the full pipeline on the data folder defined in ``config.toml``:
1. Ingest PDFs (src.ingest)
2. Chunk pages (src.chunk)
3. Build a BM25 index (src.index)
4. Retrieve top chunks for a question (src.retrieve)
5. Generate an answer via the local Ollama Mistral model (src.answer)
6. Resolve citations (src.cite)

The question is taken from the command line; if omitted the default is
"Why was Oak renamed to Java?".
"""

import sys
from pathlib import Path

# Ensure the project root (parent of "scripts") is on the import path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.ingest import parse_folder
from src.chunk import chunk_page_records
from src.index import build_bm25_index, build_chunk_embeddings
from src.retrieve import keyword_retrieve, vector_retrieve
from src.answer import generate_answer, get_generation_top_k
from src.cite import resolve_citations, CitationResult


def evaluate_citations_and_answer(citation_result: CitationResult) -> tuple[str, bool]:
    """Evaluate citation result against the zero-citation rule.

    If the citation result has zero valid citations, return 'not found in your documents'
    and flag answer_failed = True.
    Otherwise, return the model's answer text and answer_failed = False.
    """
    if not citation_result.valid_citations:
        return "not found in your documents", True
    return citation_result.answer, False


def main() -> None:
    # ------------------------------------------------------------
    # 1. Get the question
    # ------------------------------------------------------------
    question = sys.argv[1] if len(sys.argv) > 1 else "Why was Oak renamed to Java?"

    # ------------------------------------------------------------
    # 2. Ingest PDF files (may raise if the data folder is missing)
    # ------------------------------------------------------------
    try:
        records = parse_folder()
    except Exception as exc:
        print(f"[Error] Ingestion failed: {exc}")
        records = []

    # ------------------------------------------------------------
    # 3. Chunk the page records
    # ------------------------------------------------------------
    chunks = chunk_page_records(records) if records else []

    # ------------------------------------------------------------
    # 4. Build BM25 index (always succeeds, even with an empty list)
    # ------------------------------------------------------------
    bm25_index = build_bm25_index(chunks)

    # ------------------------------------------------------------
    # 5. Retrieve top‑N chunks for the question
    # ------------------------------------------------------------
    retrieved_raw = keyword_retrieve(question, bm25_index)
    # ``keyword_retrieve`` returns a list of chunks or the string "not found"
    retrieved = [] if isinstance(retrieved_raw, str) else retrieved_raw

    # Pass only top GENERATION_TOP_K retrieved chunks to generate_answer
    top_k = get_generation_top_k()
    generation_chunks = retrieved[:top_k]

    # Compute top vector score for the question
    top_vector_score = None
    if chunks:
        try:
            vectors = build_chunk_embeddings(chunks)
            vec_results = vector_retrieve(question, chunks, vectors, top_n=1)
            if vec_results:
                top_vector_score = vec_results[0][1]
        except Exception as exc:
            print(f"[Warning] Vector retrieval failed: {exc}")

    # ------------------------------------------------------------
    # 6. Generate an answer (guard against missing Ollama server)
    # ------------------------------------------------------------
    try:
        answer_text = generate_answer(question, generation_chunks, top_score=top_vector_score)
    except Exception as exc:
        answer_text = f"[Error generating answer: {exc}]"

    # ------------------------------------------------------------
    # 7. Resolve citations in the answer
    # ------------------------------------------------------------
    citation_result = resolve_citations(answer_text, generation_chunks)
    answer_to_print, answer_failed = evaluate_citations_and_answer(citation_result)

    # ------------------------------------------------------------
    # 8. Print results
    # ------------------------------------------------------------
    print(f"Question: {question}\n")
    print(f"Answer: {answer_to_print}\n")
    print("Citations:")
    for cit in citation_result.valid_citations:
        print(
            f"  ID {cit['id']}: file {cit['file']}, page {cit['page']}, "
            f"chars {cit['start_char']}-{cit['end_char']}"
        )
    if citation_result.dropped_citations:
        print("\nDropped citations: " + ", ".join(citation_result.dropped_citations))
    else:
        print("\nNo dropped citations.")


if __name__ == "__main__":
    main()
