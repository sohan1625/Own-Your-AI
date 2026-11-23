import unittest
from unittest.mock import patch, MagicMock
import os
import json
import tempfile
from pathlib import Path

from src.answer import generate_answer, _build_prompt, _validate_host, normalize_answer, is_below_threshold

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
            "If the chunks do not contain the answer, reply with only the exact text "
            "'not found in your documents' and nothing else. Never include that text "
            "inside a longer answer. "
            "Answer in at most 3 sentences. Answer only what was asked. "
            "Do not mention the context, the chunks, or the document. "
            "Do not describe related concepts unless the question asks for them."
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
        self.assertIn("Never include that text inside a longer answer.", prompt)

    def test_prompt_limits_answer_length_and_scope(self):
        prompt = _build_prompt(self.question, self.sample_chunks)
        self.assertIn("Answer in at most 3 sentences.", prompt)
        self.assertIn("Answer only what was asked.", prompt)
        self.assertIn("Do not mention the context, the chunks, or the document.", prompt)
        self.assertIn(
            "Do not describe related concepts unless the question asks for them.",
            prompt,
        )

    def test_prompt_uses_history_only_to_resolve_references(self):
        history = [
            {"role": "user", "content": "What are the different types of delays?"},
            {
                "role": "assistant",
                "content": "Transmission Delay, Propagation Delay, Queuing Delay, and Processing Delay.",
            },
        ]

        prompt = _build_prompt("Explain those.", self.sample_chunks, history)

        self.assertIn("Transmission Delay", prompt)
        self.assertIn(
            "reference resolution only; not a source of facts",
            prompt,
        )
        self.assertIn("Use the provided chunks for factual answers", prompt)
        self.assertIn("Question: Explain those.", prompt)
        self.assertIn("It may include an earlier answer: do not repeat it.", prompt)

    def test_needs_conversation_history(self):
        from src.answer import needs_conversation_history
        for question in ("Explain those.", "explain it", "When is it used?", "What types are there?",
                         "why?", "and 2NF?", "Tell me more about them"):
            self.assertTrue(needs_conversation_history(question), question)
        for question in ("explain all the normalizations", "What is a primary key in SQL?",
                         "What are all the different normalizations?"):
            self.assertFalse(needs_conversation_history(question), question)

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
        self.assertEqual(answer, "Generated answer [c1]")
        # Verify that the request was built with the correct URL and payload
        args, kwargs = mock_urlopen.call_args
        request = args[0]
        self.assertTrue(request.full_url.startswith(('http://localhost', 'http://127.0.0.1')))
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

    def test_is_below_threshold_score_0_60(self):
        """A score of 0.60 is below the 0.605 threshold."""
        self.assertTrue(is_below_threshold(0.60))
        self.assertFalse(is_below_threshold(0.61))

    def test_is_below_threshold_score_0_61(self):
        """A score of 0.61 is not blocked while the threshold is disabled."""
        self.assertFalse(is_below_threshold(0.61))

    @patch('urllib.request.urlopen')
    def test_generate_answer_ignores_low_score_when_threshold_disabled(self, mock_urlopen):
        """A low score still generates an answer while threshold checks are disabled."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"response": "Generated answer"}).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            config_path.write_text(
                '[ollama]\n'
                'host = "http://localhost:11434"\n'
                'model = "mistral"\n'
                '\n[generation]\n'
                'top_k = 1\n'
                '\n[thresholds]\n',
                encoding="utf-8",
            )
            result = generate_answer(
                self.question,
                self.sample_chunks,
                top_score=0.60,
                config_path=config_path,
            )
        self.assertEqual(result, "Generated answer [c1]")
        mock_urlopen.assert_called_once()

    def _write_config(self, temp_dir, generation_extra=""):
        config_path = Path(temp_dir) / "config.toml"
        config_path.write_text(
            '[ollama]\n'
            'host = "http://127.0.0.1:11434"\n'
            'model = "mistral"\n'
            '\n[generation]\n'
            'top_k = 1\n'
            + generation_extra,
            encoding="utf-8",
        )
        return config_path

    @patch('urllib.request.urlopen')
    def test_temperature_from_config_is_sent_in_options(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"response": "Answer [c1]"}).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(temp_dir, 'temperature = 0\n')
            generate_answer(self.question, self.sample_chunks, config_path=config_path)
        payload = json.loads(mock_urlopen.call_args[0][0].data.decode('utf-8'))
        self.assertEqual(payload["options"], {"temperature": 0})

    @patch('urllib.request.urlopen')
    def test_no_options_when_temperature_unset(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"response": "Answer [c1]"}).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(temp_dir)
            generate_answer(self.question, self.sample_chunks, config_path=config_path)
        payload = json.loads(mock_urlopen.call_args[0][0].data.decode('utf-8'))
        self.assertNotIn("options", payload)

    @patch('urllib.request.urlopen')
    def test_max_tokens_from_config_is_sent_as_num_predict(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"response": "Answer [c1]"}).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(temp_dir, 'temperature = 0\nmax_tokens = 256\n')
            generate_answer(self.question, self.sample_chunks, config_path=config_path)
        payload = json.loads(mock_urlopen.call_args[0][0].data.decode('utf-8'))
        self.assertEqual(payload["options"], {"temperature": 0, "num_predict": 256})

    def test_project_config_caps_answer_length(self):
        from src.answer import get_generation_max_tokens
        self.assertEqual(get_generation_max_tokens("config.toml"), 256)

    def test_project_config_sets_temperature_zero(self):
        from src.answer import get_generation_temperature
        self.assertEqual(get_generation_temperature("config.toml"), 0.0)

    @patch('urllib.request.urlopen')
    def test_generate_answer_appends_top_chunk_id_when_citation_is_missing(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"response": "Generated answer"}).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response

        answer = generate_answer(self.question, self.sample_chunks)

        self.assertEqual(answer, "Generated answer [c1]")

    @patch('urllib.request.urlopen')
    def test_generate_answer_preserves_existing_valid_citation(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"response": "Generated answer [c2]"}).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response

        answer = generate_answer(self.question, self.sample_chunks)

        self.assertEqual(answer, "Generated answer [c2]")

    @patch('urllib.request.urlopen')
    def test_generate_answer_does_not_append_citation_to_not_found(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "response": "not found in your documents"
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response

        answer = generate_answer(self.question, self.sample_chunks)

        self.assertEqual(answer, "not found in your documents")

    @patch('urllib.request.urlopen')
    def test_reworded_refusal_becomes_exact_refusal_without_citation(self, mock_urlopen):
        # Real mistral outputs from the eval set (q36, q35).
        for response in (
            "The capital of France is not found in your documents.",
            "Not found in your documents. The provided chunks do not contain information "
            "about the features in Python 3.12.",
        ):
            with self.subTest(response=response):
                mock_response = MagicMock()
                mock_response.read.return_value = json.dumps({"response": response}).encode('utf-8')
                mock_urlopen.return_value.__enter__.return_value = mock_response

                answer = generate_answer(self.question, self.sample_chunks)

                self.assertEqual(answer, "not found in your documents")

    @patch('urllib.request.urlopen')
    def test_cited_partial_answer_mentioning_not_found_is_kept(self, mock_urlopen):
        response = "Part one is covered here [c2]. Part two is not found in your documents."
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"response": response}).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response

        answer = generate_answer(self.question, self.sample_chunks)

        self.assertEqual(answer, response)

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

    def test_bracketed_non_id_string_is_removed(self):
        result = normalize_answer(
            "The class is a blueprint [not found in your documents].",
            [self.UUID],
        )
        self.assertEqual(result, "The class is a blueprint.")

    def test_mixed_answer_removes_bracketed_not_found_text_but_keeps_citation(self):
        raw = (
            f"A class is a blueprint [{self.UUID}], and OOP has four pillars "
            "[not found in your documents] respectively."
        )
        result = normalize_answer(raw, [self.UUID])
        self.assertEqual(
            result,
            f"A class is a blueprint [{self.UUID}], and OOP has four pillars respectively.",
        )


if __name__ == '__main__':
    unittest.main()


class TestUnloadModel(unittest.TestCase):
    @patch('urllib.request.urlopen')
    def test_unload_sends_keep_alive_zero_to_local_host(self, mock_urlopen):
        from src.answer import unload_model
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            config_path.write_text('[ollama]\nhost = "http://127.0.0.1:11434"\nmodel = "mistral"\n', encoding="utf-8")
            unload_model(config_path)
        request = mock_urlopen.call_args[0][0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/generate")
        self.assertEqual(json.loads(request.data.decode("utf-8")), {"model": "mistral", "keep_alive": 0})

    @patch('urllib.request.urlopen')
    def test_unload_refuses_non_local_host(self, mock_urlopen):
        from src.answer import unload_model
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            config_path.write_text('[ollama]\nhost = "http://example.com:11434"\nmodel = "mistral"\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                unload_model(config_path)
        mock_urlopen.assert_not_called()
