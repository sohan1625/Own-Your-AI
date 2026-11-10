import pytest
from src.cite import resolve_citations, CitationResult

# Helper chunk fixtures
chunks = [
    {
        "chunk_id": "C1",
        "file": "doc1.pdf",
        "page": 2,
        "start_char": 10,
        "end_char": 50,
        "text": "Chunk one text",
    },
    {
        "chunk_id": "C2",
        "file": "doc2.pdf",
        "page": 5,
        "start_char": 0,
        "end_char": 30,
        "text": "Chunk two text",
    },
]

def test_all_citations_valid():
    answer = "The answer references [C1] and also [C2]."
    result = resolve_citations(answer, chunks)
    assert isinstance(result, CitationResult)
    assert result.answer == answer
    assert result.has_invalid is False
    assert result.dropped_citations == []
    # Order should follow appearance
    assert [c["id"] for c in result.valid_citations] == ["C1", "C2"]
    # Metadata matches
    assert result.valid_citations[0]["file"] == "doc1.pdf"
    assert result.valid_citations[1]["page"] == 5

def test_one_invalid_citation_dropped_and_flagged():
    answer = "Here we cite [C1] and a missing [C3]."
    result = resolve_citations(answer, chunks)
    assert result.has_invalid is True
    assert result.dropped_citations == ["C3"]
    assert [c["id"] for c in result.valid_citations] == ["C1"]

def test_no_citations():
    answer = "This answer contains no citations."
    result = resolve_citations(answer, chunks)
    assert result.has_invalid is False
    assert result.dropped_citations == []
    assert result.valid_citations == []

def test_duplicate_citation_ids():
    answer = "Refer to [C2] twice: first [C2] then again [C2]."
    result = resolve_citations(answer, chunks)
    # All three citations should be present in valid_citations (including duplicates)
    assert result.has_invalid is False
    assert result.dropped_citations == []
    assert [c["id"] for c in result.valid_citations] == ["C2", "C2", "C2"]

