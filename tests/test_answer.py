import unittest
from unittest.mock import patch, MagicMock
import os
import json

from src.answer import generate_answer, _build_prompt, _validate_host

class TestAnswerGeneration(unittest.TestCase):
    def setUp(self):
        # Ensure config points to localhost for tests
        self.original_env = os.environ.copy()
        os.environ['OLLAMA_HOST'] = 'http://localhost:11434'
        os.environ['OLLAMA_MODEL'] = 'mistral'
        self.sample_chunks = [
            {"chunk_id": "c1", "text": "First chunk text."},
            {"chunk_id": "c2", "text": "Second chunk text."},
        ]
        self.question = "What is the answer?"
        self.expected_prompt = (
            "Answer the question using ONLY the provided chunks. Cite each claim with the chunk ID. "
            "If the chunks do not contain the answer, reply exactly 'not found in your documents'."
            "\n\nQuestion: What is the answer?\n\nContext:\n"
            "Chunk c1: First chunk text.\nChunk c2: Second chunk text."
        )

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_build_prompt_contains_chunks_and_instruction(self):
        prompt = _build_prompt(self.question, self.sample_chunks)
        self.assertEqual(prompt, self.expected_prompt)
        self.assertIn("Chunk c1: First chunk text.", prompt)
        self.assertIn("Chunk c2: Second chunk text.", prompt)
        self.assertIn("Answer the question using ONLY the provided chunks", prompt)
        self.assertIn("not found in your documents", prompt)

    def test_validate_host_allows_localhost(self):
        self.assertEqual(_validate_host('http://localhost:11434'), 'http://localhost:11434')
        self.assertEqual(_validate_host('http://127.0.0.1'), 'http://127.0.0.1')
        with self.assertRaises(ValueError):
            _validate_host('http://example.com')

    @patch('urllib.request.urlopen')
    def test_generate_answer_calls_ollama_and_returns_response(self, mock_urlopen):
        # Mock response from Ollama
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"response": "Generated answer"}).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response

        answer = generate_answer(self.question, self.sample_chunks)
        # Verify the function returned the mocked response text
        self.assertEqual(answer, "Generated answer")
        # Verify that the request was built with the correct URL and payload
        args, kwargs = mock_urlopen.call_args
        request = args[0]
        self.assertTrue(request.full_url.startswith('http://localhost'))
        payload = json.loads(request.data.decode('utf-8'))
        self.assertEqual(payload['model'], 'mistral')
        self.assertEqual(payload['prompt'], self.expected_prompt)

if __name__ == '__main__':
    unittest.main()

