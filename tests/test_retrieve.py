import unittest
from src.index import build_bm25_index
from src.retrieve import keyword_retrieve

class TestKeywordRetrieve(unittest.TestCase):
    def setUp(self):
        # Hand‑crafted chunk set for deterministic BM25 scoring
        self.chunks = [
            {"file": "a.pdf", "page": 1, "text": "apple orange banana", "parent_id": "a#page1"},
            {"file": "b.pdf", "page": 2, "text": "apple apple fruit", "parent_id": "b#page2"},
            {"file": "c.pdf", "page": 3, "text": "car truck vehicle", "parent_id": "c#page3"},
        ]
        self.index = build_bm25_index(self.chunks)

    def test_match_found(self):
        # Query that matches "apple" should return chunks (at least one)
        result = keyword_retrieve("apple", self.index)
        self.assertIsInstance(result, list)
        self.assertGreaterEqual(len(result), 1)
        # The top result should be the chunk with two apples (b.pdf)
        self.assertEqual(result[0]["file"], "b.pdf")

    def test_no_match(self):
        result = keyword_retrieve("pineapple", self.index)
        self.assertEqual(result, "not found")

    def test_more_term_matches(self):
        # Query with two terms; chunk containing both should rank higher
        result = keyword_retrieve("apple fruit", self.index)
        # Expect the chunk that contains both terms (b.pdf) to be first
        self.assertEqual(result[0]["file"], "b.pdf")
        # The second result should be the chunk that contains only "apple"
        self.assertEqual(result[1]["file"], "a.pdf")

if __name__ == "__main__":
    unittest.main()

