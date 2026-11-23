"""Semantic chunking: sentence spans, topic-shift boundaries, page boundaries, size cap.

Uses a fake embedding (one dimension per topic word), so no model is called.
"""

import pytest

from src.chunk import (
    chunk_page_records,
    get_chunking_strategy,
    percentile,
    semantic_chunk_page_records,
    split_sentences,
)

TOPICS = ("cat", "car", "sun")


def topic_embed(text: str) -> list[float]:
    words = text.lower()
    return [1.0 if topic in words else 0.0 for topic in TOPICS] + [0.01]


def _config(tmp_path, chunk_size=400, strategy="semantic"):
    path = tmp_path / "config.toml"
    path.write_text(
        f'[chunking]\nchunk_size = {chunk_size}\noverlap = 0\nstrategy = "{strategy}"\nsemantic_percentile = 50\n',
        encoding="utf-8",
    )
    return path


def _assert_spans_exact(chunks, records):
    texts = {(r["file"], r["page"]): r["text"] for r in records}
    for chunk in chunks:
        page_text = texts[(chunk["file"], chunk["page"])]
        span = page_text[chunk["start_char"] : chunk["end_char"]]
        assert " ".join(span.split()) == chunk["text"]
        assert span == span.strip()


def test_split_sentences_spans_point_at_each_sentence():
    text = "  First one.  Second one!\nThird one?\n\nHeading line\nFinal words  "
    spans = split_sentences(text)
    assert [text[s:e] for s, e in spans] == [
        "First one.",
        "Second one!",
        "Third one?",
        "Heading line\nFinal words",
    ]


def test_split_sentences_empty_text():
    assert split_sentences("") == []
    assert split_sentences("   \n ") == []


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 0) == 1
    assert percentile([1, 2, 3, 4], 100) == 4
    assert percentile([1, 2, 3, 4], 50) == 2.5


def test_new_chunk_starts_at_topic_shift(tmp_path):
    text = (
        "The cat sleeps. The cat eats fish. "
        "A car needs fuel. The car has wheels. "
        "The sun is hot. The sun rises."
    )
    records = [{"file": "doc.pdf", "page": 3, "text": text, "file_hash": "h"}]
    chunks = semantic_chunk_page_records(records, config_path=_config(tmp_path), embed=topic_embed)
    assert [c["text"] for c in chunks] == [
        "The cat sleeps. The cat eats fish.",
        "A car needs fuel. The car has wheels.",
        "The sun is hot. The sun rises.",
    ]
    assert all(c["page"] == 3 and c["parent_id"] == "doc.pdf#page3" for c in chunks)
    _assert_spans_exact(chunks, records)
    # Chunks are contiguous and cover the page text in order.
    assert chunks[0]["start_char"] == 0
    assert chunks[-1]["end_char"] == len(text)


def test_chunks_never_cross_pages(tmp_path):
    records = [
        {"file": "doc.pdf", "page": 1, "text": "The cat sleeps. The cat purrs.", "file_hash": "h"},
        {"file": "doc.pdf", "page": 2, "text": "The cat eats. The cat runs.", "file_hash": "h"},
    ]
    chunks = semantic_chunk_page_records(records, config_path=_config(tmp_path), embed=topic_embed)
    assert [(c["page"], c["text"]) for c in chunks] == [
        (1, "The cat sleeps. The cat purrs."),
        (2, "The cat eats. The cat runs."),
    ]
    _assert_spans_exact(chunks, records)


def test_size_cap_splits_long_groups_and_long_sentences(tmp_path):
    # chunk_size 3 tokens -> 4 words per chunk.
    long_sentence = "cat " * 9 + "end."
    text = f"The cat sat. The cat ran. {long_sentence}"
    records = [{"file": "doc.pdf", "page": 1, "text": text, "file_hash": "h"}]
    chunks = semantic_chunk_page_records(
        records, config_path=_config(tmp_path, chunk_size=3), embed=topic_embed
    )
    assert all(len(c["text"].split()) <= 4 for c in chunks)
    assert chunks[0]["text"] == "The cat sat."
    _assert_spans_exact(chunks, records)
    # No words lost: every word of the page appears once across the chunks.
    assert " ".join(c["text"] for c in chunks).split() == text.split()


def test_empty_pages_produce_no_chunks(tmp_path):
    records = [{"file": "doc.pdf", "page": 1, "text": "  ", "file_hash": "h"}]
    assert semantic_chunk_page_records(records, config_path=_config(tmp_path), embed=topic_embed) == []


def test_strategy_switch(tmp_path, monkeypatch):
    assert get_chunking_strategy("config.toml") == "fixed"
    import src.chunk as chunk_module

    called = {}

    def fake_semantic(records, *, config_path):
        called["yes"] = True
        return []

    monkeypatch.setattr(chunk_module, "semantic_chunk_page_records", fake_semantic)
    chunk_page_records([{"file": "a.pdf", "page": 1, "text": "x"}], config_path=_config(tmp_path))
    assert called == {"yes": True}

    bad = tmp_path / "bad.toml"
    bad.write_text('[chunking]\nstrategy = "magic"\n', encoding="utf-8")
    with pytest.raises(ValueError):
        get_chunking_strategy(bad)
