"""Unit tests for vector retrieval and embedding utilities.

Tests:
- Cosine ranking order with a small hand-made vector set
- Search prefixes applied for questions (search_query:) and chunks (search_document:)
- Rejection of non-localhost Ollama hosts
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from src.index import (
    _validate_host,
    apply_prefix,
    build_chunk_embeddings,
    embed_text,
    get_embedding_prefix,
)
from src.retrieve import cosine_similarity, vector_retrieve


class TestVectorRetrieval(unittest.TestCase):
    def setUp(self):
        # Small handcrafted chunk set
        self.chunks = [
            {"chunk_id": "c1", "file": "doc1.pdf", "page": 1, "text": "Python programming language"},
            {"chunk_id": "c2", "file": "doc1.pdf", "page": 2, "text": "Java programming language"},
            {"chunk_id": "c3", "file": "doc2.pdf", "page": 5, "text": "Culinary arts and baking bread"},
        ]
        # Handcrafted 3D vectors:
        # Vector 1 aligned with [1, 0, 0] (exact match to query)
        # Vector 2 partially aligned [1, 1, 0] normalized
        # Vector 3 orthogonal [0, 0, 1]
        self.vectors = [
            [1.0, 0.0, 0.0],
            [0.70710678, 0.70710678, 0.0],
            [0.0, 0.0, 1.0],
        ]
        self.query_vector = [1.0, 0.0, 0.0]

    def test_cosine_similarity(self):
        """Test cosine similarity calculation with known mathematical vectors."""
        # Exact match
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        # Orthogonal
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)
        # Opposite
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)
        # 45 degrees
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 1.0]), 0.70710678, places=5)
        # Zero norm
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 1.0]), 0.0)
        # Dimension mismatch
        self.assertEqual(cosine_similarity([1.0], [1.0, 2.0]), 0.0)

    @patch("src.index.urllib.request.urlopen")
    def test_cosine_ranking_order(self, mock_urlopen):
        """Verify vector_retrieve ranks chunks in descending cosine similarity order."""
        # Mock embed response for the query
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"embeddings": [self.query_vector]}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        results = vector_retrieve("Python programming", self.chunks, self.vectors, top_n=3)

        self.assertEqual(len(results), 3)
        # c1 should rank 1st with cosine sim 1.0
        self.assertEqual(results[0][0]["chunk_id"], "c1")
        self.assertAlmostEqual(results[0][1], 1.0, places=4)

        # c2 should rank 2nd with cosine sim ~0.7071
        self.assertEqual(results[1][0]["chunk_id"], "c2")
        self.assertAlmostEqual(results[1][1], 0.70710678, places=4)

        # c3 should rank 3rd with cosine sim 0.0
        self.assertEqual(results[2][0]["chunk_id"], "c3")
        self.assertAlmostEqual(results[2][1], 0.0, places=4)

        # Verify top_n slicing
        results_top2 = vector_retrieve("Python programming", self.chunks, self.vectors, top_n=2)
        self.assertEqual(len(results_top2), 2)
        self.assertEqual([r[0]["chunk_id"] for r in results_top2], ["c1", "c2"])

    @patch("src.index.urllib.request.urlopen")
    def test_prefix_applied_for_query_and_documents(self, mock_urlopen):
        """Verify search_query prefix is used for queries and search_document prefix for chunks."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"embeddings": [[0.5, 0.5, 0.0]]}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        # 1. Query embedding via vector_retrieve
        vector_retrieve("What is Java?", self.chunks[:1], self.vectors[:1], top_n=1)
        args, _ = mock_urlopen.call_args
        req = args[0]
        payload = json.loads(req.data.decode("utf-8"))
        self.assertEqual(payload["model"], "nomic-embed-text")
        self.assertTrue(
            payload["input"].startswith("search_query: "),
            f"Expected search_query prefix, got: {payload['input']}",
        )
        self.assertIn("What is Java?", payload["input"])

        # 2. Chunk embedding via build_chunk_embeddings
        mock_urlopen.reset_mock()
        sample_chunks = [{"chunk_id": "test_c", "text": "JVM runs bytecode"}]
        build_chunk_embeddings(sample_chunks)
        args, _ = mock_urlopen.call_args
        req = args[0]
        payload = json.loads(req.data.decode("utf-8"))
        self.assertEqual(payload["model"], "nomic-embed-text")
        self.assertTrue(
            payload["input"].startswith("search_document: "),
            f"Expected search_document prefix, got: {payload['input']}",
        )
        self.assertIn("JVM runs bytecode", payload["input"])

    def test_non_localhost_host_refused(self):
        """Verify that non-localhost Ollama hosts are refused with a clear ValueError."""
        # Remote endpoints must be rejected
        with self.assertRaises(ValueError):
            _validate_host("http://remote-server:11434")

        with self.assertRaises(ValueError):
            _validate_host("http://example.com")

        with self.assertRaises(ValueError):
            _validate_host("https://192.168.1.100:11434")

        with self.assertRaises(ValueError):
            embed_text("test text", host="http://external-api.com:11434")

        with self.assertRaises(ValueError):
            build_chunk_embeddings(self.chunks, host="http://external-api.com:11434")

        # Localhost endpoints must be accepted
        self.assertEqual(_validate_host("http://localhost:11434"), "http://localhost:11434")
        self.assertEqual(_validate_host("http://127.0.0.1:11434"), "http://127.0.0.1:11434")
        self.assertEqual(_validate_host("http://localhost:11434/"), "http://localhost:11434")


if __name__ == "__main__":
    unittest.main()

