import pathlib
import tempfile
import unittest
from src.chunk import chunk_page_records

class TestChunking(unittest.TestCase):
    """Fixed-window chunking. Uses its own config, since config.toml may select "semantic"."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.fixed_config = pathlib.Path(self._tmp.name) / "config.toml"
        self.fixed_config.write_text(
            '[chunking]\nchunk_size = 400\noverlap = 50\nstrategy = "fixed"\n', encoding="utf-8"
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_empty_page(self):
        records = [{"file": "doc.pdf", "page": 1, "text": "", "file_hash": "h"}]
        chunks = chunk_page_records(records, config_path=self.fixed_config)
        self.assertEqual(chunks, [], "Empty page should produce no chunks")

    def test_short_page_default_config(self):
        # Default config (400 tokens ≈ 533 words) -> any short text yields a single chunk
        text = "short page text"
        records = [{"file": "doc.pdf", "page": 1, "text": text, "file_hash": "h"}]
        chunks = chunk_page_records(records, config_path=self.fixed_config)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], text)
        self.assertEqual(chunks[0]["parent_id"], "doc.pdf#page1")

    def test_overlap_correctness(self):
        # Write a temporary config with small chunk size and overlap
        cfg_path = pathlib.Path(__file__).parent / "temp_config.toml"
        cfg_path.write_text("[chunking]\nchunk_size = 30\noverlap = 10\n")
        # 70 words page
        words = [f"w{i}" for i in range(70)]
        text = " ".join(words)
        records = [{"file": "doc.pdf", "page": 1, "text": text, "file_hash": "h"}]
        chunks = chunk_page_records(records, config_path=cfg_path)
        # Expect three chunks (based on config and 70 words)
        self.assertEqual(len(chunks), 3)
        # Compute expected overlap word count
        overlap_words = int(round(10 / 0.75))
        # Verify overlapping words are identical between each consecutive chunk
        for i in range(len(chunks) - 1):
            cur_words = chunks[i]["text"].split()
            next_words = chunks[i + 1]["text"].split()
            self.assertEqual(cur_words[-overlap_words:], next_words[:overlap_words])
        # Clean up
        cfg_path.unlink()

    def test_no_cross_page_boundary(self):
        # Two pages with distinct vocab; ensure chunks stay within each page
        text1 = "alpha beta gamma delta epsilon"
        text2 = "zeta eta theta iota kappa"
        records = [
            {"file": "doc.pdf", "page": 1, "text": text1, "file_hash": "h"},
            {"file": "doc.pdf", "page": 2, "text": text2, "file_hash": "h"},
        ]
        chunks = chunk_page_records(records, config_path=self.fixed_config)
        # All chunks from page 1 should have parent_id ending with #page1, similarly for page2
        page1_chunks = [c for c in chunks if c["page"] == 1]
        page2_chunks = [c for c in chunks if c["page"] == 2]
        self.assertTrue(all(c["parent_id"].endswith("#page1") for c in page1_chunks))
        self.assertTrue(all(c["parent_id"].endswith("#page2") for c in page2_chunks))
        # Ensure no chunk contains text from the other page
        for c in page1_chunks:
            self.assertNotIn("zeta", c["text"], "Chunk from page 1 should not contain page 2 words")
        for c in page2_chunks:
            self.assertNotIn("alpha", c["text"], "Chunk from page 2 should not contain page 1 words")

if __name__ == "__main__":
    unittest.main()
