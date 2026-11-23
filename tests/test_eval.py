"""Unit tests for the evaluation harness (scripts/eval.py).

Uses a 3-question fixture in tests/fixtures without calling real models or the real eval set.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from scripts.eval import (
    load_questions,
    evaluate_retrieval_match,
    reciprocal_rank_fusion,
    run_evaluation,
)
from src.index import build_bm25_index


class TestEvalHarness(unittest.TestCase):
    def setUp(self):
        self.fixtures_dir = Path(__file__).resolve().parent / "fixtures"
        self.fixture_questions = self.fixtures_dir / "questions_fixture.jsonl"
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_json = Path(self.temp_dir.name) / "test_results.json"

        # Synthetic chunks for a controlled index
        self.synthetic_chunks = [
            {
                "chunk_id": "c1",
                "file": "sample.pdf",
                "page": 1,
                "start_char": 0,
                "end_char": 50,
                "text": "Python is a high-level general-purpose programming language.",
            },
            {
                "chunk_id": "c2",
                "file": "other.pdf",
                "page": 2,
                "start_char": 0,
                "end_char": 40,
                "text": "Some other irrelevant text in another document.",
            },
        ]
        self.bm25_index = build_bm25_index(self.synthetic_chunks)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_questions(self):
        """Verify load_questions reads all entries from JSONL."""
        questions = load_questions(self.fixture_questions)
        self.assertEqual(len(questions), 3)
        self.assertEqual(questions[0]["id"], "fx01")
        self.assertTrue(questions[0]["answerable"])
        self.assertFalse(questions[2]["answerable"])

    def test_evaluate_retrieval_match(self):
        """Verify rank detection against gold passages."""
        gold = [{"file": "sample.pdf", "page": 1}]
        retrieved = [
            {"file": "sample.pdf", "page": 2},  # rank 1 (wrong page)
            {"file": "sample.pdf", "page": 1},  # rank 2 (match!)
            {"file": "other.pdf", "page": 1},   # rank 3
        ]
        rank = evaluate_retrieval_match(retrieved, gold)
        self.assertEqual(rank, 2)

        # No match returns None
        no_match = evaluate_retrieval_match(retrieved, [{"file": "missing.pdf", "page": 1}])
        self.assertIsNone(no_match)

        # Empty gold returns None
        empty_gold = evaluate_retrieval_match(retrieved, [])
        self.assertIsNone(empty_gold)

    @patch("src.answer.urllib.request.urlopen")
    def test_run_evaluation_retrieval_only(self, mock_urlopen):
        """Retrieval-only evaluation does not call Ollama and computes metrics correctly."""
        results = run_evaluation(
            questions_path=self.fixture_questions,
            output_path=self.output_json,
            with_answers=False,
            bm25_index=self.bm25_index,
        )

        # Ollama should NOT be called in retrieval-only mode
        mock_urlopen.assert_not_called()

        summary = results["summary"]
        self.assertEqual(summary["total_questions"], 3)
        self.assertEqual(summary["answerable_questions"], 2)
        self.assertEqual(summary["unanswerable_questions"], 1)

        # fx01 ("What is Python?") matches sample.pdf page 1 at rank 1 -> Recall@1 = 1/2 = 0.5
        # fx02 ("quantum gravity") does not match -> Recall = 0.0
        self.assertAlmostEqual(summary["recall_at_1"], 0.5)
        self.assertAlmostEqual(summary["recall_at_3"], 0.5)
        self.assertAlmostEqual(summary["recall_at_5"], 0.5)
        self.assertAlmostEqual(summary["mrr"], 0.5)
        self.assertIsNone(summary["hallucination_rate"])

        # Check output JSON file was written
        self.assertTrue(self.output_json.is_file())
        with open(self.output_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["mode"], "keyword")
        self.assertFalse(data["with_answers"])
        self.assertEqual(len(data["questions"]), 3)

    @patch("scripts.eval.generate_answer")
    def test_run_evaluation_with_answers_evaluates_hallucination(self, mock_gen_answer):
        """When with_answers=True, unanswerable questions are checked for hallucination."""
        # Mock answers:
        # fx01: provides answer with citation
        # fx02: not found
        # fx03 (unanswerable): returns "not found in your documents"
        mock_gen_answer.side_effect = [
            "Python is a language. [c1]",
            "not found in your documents",
            "not found in your documents",
        ]

        results = run_evaluation(
            questions_path=self.fixture_questions,
            output_path=self.output_json,
            with_answers=True,
            bm25_index=self.bm25_index,
        )

        summary = results["summary"]
        # fx03 did not hallucinate (returned not found in your documents)
        self.assertEqual(summary["hallucination_rate"], 0.0)
        # fx02 is answerable but was refused
        self.assertEqual(summary["answerable_wrongly_refused"], 1)
        flags = {q["id"]: q["wrongly_refused"] for q in results["questions"]}
        self.assertEqual(flags, {"fx01": False, "fx02": True, "fx03": None})

    @patch("scripts.eval.unload_model")
    @patch("scripts.eval.generate_answer", return_value="not found in your documents")
    def test_cold_start_unloads_model_once_only_when_asked(self, _gen, mock_unload):
        run_evaluation(questions_path=self.fixture_questions, output_path=self.output_json,
                       with_answers=True, bm25_index=self.bm25_index)
        mock_unload.assert_not_called()
        run_evaluation(questions_path=self.fixture_questions, output_path=self.output_json,
                       with_answers=True, bm25_index=self.bm25_index, cold_start=True)
        mock_unload.assert_called_once()

    def test_assign_stable_chunk_ids_is_repeatable_and_unique(self):
        from scripts.eval import assign_stable_chunk_ids
        import copy
        first = assign_stable_chunk_ids(copy.deepcopy(self.synthetic_chunks))
        second = assign_stable_chunk_ids(copy.deepcopy(self.synthetic_chunks))
        self.assertEqual([c["chunk_id"] for c in first], [c["chunk_id"] for c in second])
        self.assertEqual(len({c["chunk_id"] for c in first}), len(first))

    def test_reciprocal_rank_fusion(self):
        """Test RRF scoring and combination across ranked lists."""
        c1 = {"chunk_id": "c1", "file": "doc1.pdf", "page": 1, "start_char": 0, "text": "text 1"}
        c2 = {"chunk_id": "c2", "file": "doc2.pdf", "page": 2, "start_char": 0, "text": "text 2"}
        c3 = {"chunk_id": "c3", "file": "doc3.pdf", "page": 3, "start_char": 0, "text": "text 3"}

        list1 = [c1, c2]  # c1 is rank 1, c2 is rank 2
        list2 = [c2, c3]  # c2 is rank 1, c3 is rank 2

        # k = 60
        # c2 is in both lists: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ≈ 0.016129 + 0.016393 = 0.032522
        # c1 is only in list 1: 1/(60+1) = 1/61 ≈ 0.016393
        # c3 is only in list 2: 1/(60+2) = 1/62 ≈ 0.016129
        # Expected order: c2, c1, c3
        fused = reciprocal_rank_fusion([list1, list2], k=60, top_n=20)
        self.assertEqual(len(fused), 3)
        self.assertEqual(fused[0]["chunk_id"], "c2")
        self.assertEqual(fused[1]["chunk_id"], "c1")
        self.assertEqual(fused[2]["chunk_id"], "c3")

        # Test top_n limits output length
        fused_top1 = reciprocal_rank_fusion([list1, list2], k=60, top_n=1)
        self.assertEqual(len(fused_top1), 1)
        self.assertEqual(fused_top1[0]["chunk_id"], "c2")

    @patch("src.retrieve.embed_text")
    def test_run_evaluation_vector_mode(self, mock_embed):
        """Vector evaluation mode embeds queries and computes metrics without hitting real Ollama."""
        # Query embedding mock
        mock_embed.return_value = [1.0, 0.0]
        # Vectors: c1 matches query [1.0, 0.0], c2 is orthogonal [0.0, 1.0]
        vectors = [[1.0, 0.0], [0.0, 1.0]]

        results = run_evaluation(
            questions_path=self.fixture_questions,
            output_path=self.output_json,
            with_answers=False,
            retriever="vector",
            chunks=self.synthetic_chunks,
            vectors=vectors,
        )

        self.assertEqual(results["mode"], "vector")
        summary = results["summary"]
        self.assertEqual(summary["total_questions"], 3)
        self.assertEqual(summary["answerable_questions"], 2)
        # Check output JSON file was written with mode vector
        self.assertTrue(self.output_json.is_file())
        with open(self.output_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["mode"], "vector")

    @patch("src.retrieve.embed_text")
    def test_run_evaluation_hybrid_mode(self, mock_embed):
        """Hybrid evaluation fuses BM25 and vector results."""
        mock_embed.return_value = [1.0, 0.0]
        vectors = [[1.0, 0.0], [0.0, 1.0]]

        results = run_evaluation(
            questions_path=self.fixture_questions,
            output_path=self.output_json,
            with_answers=False,
            retriever="hybrid",
            bm25_index=self.bm25_index,
            chunks=self.synthetic_chunks,
            vectors=vectors,
        )

        self.assertEqual(results["mode"], "hybrid")
        self.assertTrue(self.output_json.is_file())
        with open(self.output_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["mode"], "hybrid")

    def test_invalid_retriever_raises_error(self):
        """Passing an unrecognized retriever strategy raises ValueError."""
        with self.assertRaises(ValueError):
            run_evaluation(
                questions_path=self.fixture_questions,
                output_path=self.output_json,
                retriever="invalid_strategy",
            )


if __name__ == "__main__":
    unittest.main()

