"""Token-overlap faithfulness check (src/faithfulness.py)."""

from src.faithfulness import SUPPORT_THRESHOLD, check_answer, content_tokens, overlap, split_answer_sentences

CHUNKS = [
    {"chunk_id": "c1", "text": "Oak was renamed to Java in 1995 because Oak was already a registered trademark."},
    {"chunk_id": "c2", "text": "PreparedStatement prevents SQL injection because values are bound as data."},
]


def test_supported_sentence_with_own_citation():
    result = check_answer("Oak was renamed to Java because Oak was a registered trademark. [c1]", CHUNKS)
    assert result["scored"] == 1 and result["supported"] == 1 and result["uncited"] == 0
    assert result["sentences"][0]["source"] == "own"


def test_unsupported_sentence_scores_low():
    result = check_answer("Kubernetes deploys containers using Helm charts. [c1]", CHUNKS)
    assert result["supported"] == 0
    assert result["sentences"][0]["score"] < SUPPORT_THRESHOLD


def test_uncited_sentence_uses_next_citation():
    answer = "Oak was renamed to Java in 1995. It was already a registered trademark. [c1]"
    result = check_answer(answer, CHUNKS)
    assert [s["source"] for s in result["sentences"]] == ["next", "own"]
    assert result["supported"] == 2


def test_trailing_sentence_uses_previous_citation():
    answer = "PreparedStatement prevents SQL injection. [c2] Values are bound as data."
    result = check_answer(answer, CHUNKS)
    assert [s["source"] for s in result["sentences"]] == ["own", "previous"]


def test_answer_without_any_known_citation_is_uncited():
    result = check_answer("Java is popular. [unknown-id]", CHUNKS)
    assert result["uncited"] == 1 and result["scored"] == 0 and result["supported"] == 0


def test_citation_only_fragment_joins_previous_sentence():
    assert split_answer_sentences("Oak became Java. [c1]") == ["Oak became Java. [c1]"]


def test_overlap_ignores_stopwords_and_plurals():
    assert content_tokens("The values are bound") == ["value", "bound"]
    assert overlap("Values bound", "value is bound") == 1.0


def test_known_limit_negation_is_not_detected():
    # Lexical overlap cannot see a flipped negation: documented limit.
    result = check_answer("PreparedStatement does not prevent SQL injection. [c2]", CHUNKS)
    assert result["supported"] == 1
