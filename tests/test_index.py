import unittest
from src.index import build_bm25_index, bm25_search

class TestBM25Index(unittest.TestCase):
    def setUp(self):
        # Create simple chunk records
        self.chunks = [
            {"file": "doc1.pdf", "page": 1, "text": "apple orange banana", "parent_id": "doc1#page1"},
            {"file": "doc2.pdf", "page": 2, "text": "apple apple fruit", "parent_id": "doc2#page2"},
            {"file": "doc3.pdf", "page": 3, "text": "car truck vehicle", "parent_id": "doc3#page3"},
        ]
        self.index = build_bm25_index(self.chunks)

    def test_search_returns_correct_order(self):
        results = bm25_search("apple", self.index, top_n=2)
        # Expect the chunk with two 'apple' occurrences to rank higher
        self.assertEqual(len(results), 2)
        top_chunk, _ = results[0]
        self.assertEqual(top_chunk["file"], "doc2.pdf")
        # Second result should be doc1
        second_chunk, _ = results[1]
        self.assertEqual(second_chunk["file"], "doc1.pdf")

    def test_search_no_match(self):
        results = bm25_search("pineapple", self.index, top_n=5)
        self.assertEqual(results, [])

if __name__ == "__main__":
    unittest.main()

