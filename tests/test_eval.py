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


if __name__ == "__main__":
    unittest.main()

