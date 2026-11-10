import unittest
from unittest.mock import patch, MagicMock
import os
import json

from src.answer import generate_answer, _build_prompt, _validate_host, normalize_answer

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
            "Answer the question using ONLY the provided chunks. "
            "Write a complete sentence that answers the question. "
            "End every sentence that uses information from a chunk with that chunk's ID "
            "in square brackets, like [chunk-id]. "
            "If the chunks do not contain the answer, reply exactly "
            "'not found in your documents'."
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

    def test_prompt_contains_square_bracket_citation_and_no_labels(self):
        """The prompt must use square-bracket citations."""
        prompt = _build_prompt(self.question, self.sample_chunks)
        # Must instruct square-bracket citation style
        self.assertIn("[chunk-id]", prompt)
        # Must not instruct parentheses for citations
        self.assertNotIn("(chunk-id)", prompt)

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
        from src.answer import get_generation_top_k
        top_k = get_generation_top_k()
        expected = _build_prompt(self.question, self.sample_chunks[:top_k])
        self.assertEqual(payload['prompt'], expected)

    @patch('urllib.request.urlopen')
    def test_request_body_has_stream_false(self, mock_urlopen):
        """The request body must include "stream": false so Ollama returns one JSON object."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"response": "ok"}).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response

        generate_answer(self.question, self.sample_chunks)

        args, _ = mock_urlopen.call_args
        payload = json.loads(args[0].data.decode('utf-8'))
        self.assertIn('stream', payload)
        self.assertIs(payload['stream'], False)

    @patch('urllib.request.urlopen')
    def test_prompt_contains_top_k_chunks_from_config_when_given_20(self, mock_urlopen):
        """When generate_answer is given 20 chunks, prompt should contain only top_k chunks from config."""
        from src.answer import get_generation_top_k
        top_k = get_generation_top_k()

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"response": "ok"}).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response

        twenty_chunks = [{"chunk_id": f"c{i}", "text": f"Text chunk {i}"} for i in range(20)]
        generate_answer(self.question, twenty_chunks)

        args, _ = mock_urlopen.call_args
        payload = json.loads(args[0].data.decode('utf-8'))
        prompt = payload['prompt']
        for i in range(top_k):
            self.assertIn(f"Chunk c{i}: Text chunk {i}", prompt)
        self.assertNotIn(f"Chunk c{top_k}: Text chunk {top_k}", prompt)
        self.assertEqual(prompt.count("Chunk c"), top_k)

    def test_zero_citation_rule_on_demo_helper(self):
        """The demo helper returns 'not found in your documents' and answer_failed=True if zero valid citations."""
        from scripts.demo import evaluate_citations_and_answer
        from src.cite import CitationResult

        # Case 1: Zero valid citations
        res_zero = CitationResult(
            answer="Model returned answer without citing anything.",
            valid_citations=[],
            dropped_citations=[],
            has_invalid=False,
        )
        answer_text, answer_failed = evaluate_citations_and_answer(res_zero)
        self.assertEqual(answer_text, "not found in your documents")
        self.assertTrue(answer_failed)

        # Case 2: At least one valid citation
        res_valid = CitationResult(
            answer="Model answer citing [c1].",
            valid_citations=[{"id": "c1", "file": "doc.pdf", "page": 1, "start_char": 0, "end_char": 10}],
            dropped_citations=[],
            has_invalid=False,
        )
        answer_text, answer_failed = evaluate_citations_and_answer(res_valid)
        self.assertEqual(answer_text, "Model answer citing [c1].")
        self.assertFalse(answer_failed)

if __name__ == '__main__':
    unittest.main()


class TestNormalizeAnswer(unittest.TestCase):
    """Unit tests for normalize_answer."""

    UUID = "a6cc4097-9e64-4b1b-9d9d-ba6257e4ec9e"

    def test_parenthesised_id_becomes_bracketed(self):
        """(uuid) in the raw text must be rewritten as [uuid]."""
        raw = f"The JVM executes bytecode. ({self.UUID})"
        result = normalize_answer(raw, [self.UUID])
        self.assertIn(f"[{self.UUID}]", result)
        self.assertNotIn(f"({self.UUID})", result)

    def test_trailing_question_block_is_cut(self):
        """A trailing 'Question: ...' line and everything after it must be removed."""
        raw = (
            f"The JVM is a virtual machine. [{self.UUID}]\n"
            "Question: What is the JVM?\n"
            "Answer: not found in your documents"
        )
        result = normalize_answer(raw, [self.UUID])
        self.assertIn(f"[{self.UUID}]", result)
        self.assertNotIn("Question:", result)
        self.assertNotIn("Answer:", result)

    def test_trailing_answer_block_is_cut(self):
        """A trailing 'Answer: ...' line must also be cut off."""
        raw = (
            f"The JVM loads class files. [{self.UUID}]\n"
            "Answer: not found in your documents"
        )
        result = normalize_answer(raw, [self.UUID])
        self.assertIn(f"[{self.UUID}]", result)
        self.assertNotIn("Answer:", result)

    def test_already_bracketed_id_is_unchanged(self):
        """An answer that already uses [uuid] must not be double-bracketed."""
        raw = f"The JVM executes bytecode. [{self.UUID}]"
        result = normalize_answer(raw, [self.UUID])
        self.assertEqual(result, raw.strip())
        # Must not produce [[uuid]] or [[[uuid]]]
        self.assertNotIn(f"[[{self.UUID}]]", result)


if __name__ == '__main__':
    unittest.main()
