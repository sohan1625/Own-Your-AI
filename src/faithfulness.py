"""Deterministic faithfulness check: is each answer sentence supported by the chunk it cites?

For every sentence of a generated answer, the cited chunk is the one named in that
sentence's ``[chunk-id]``. A sentence with no citation of its own uses the next
citation after it in the answer (models often cite once at the end of a run of
sentences); if none follows, the previous one. A sentence with no citation in the
whole answer is reported as uncited.

Support is **token overlap**: the share of the sentence's content words (lower-cased,
stopwords removed, a trailing plural "s" stripped) that also appear in the cited
chunk's text. A sentence is supported when that share is at least
``SUPPORT_THRESHOLD``. The threshold is fixed in advance, not tuned on any eval set.

Known limits (this is a lexical proxy, not entailment):
- Paraphrases that use different words score low (false "unsupported").
- Copied words in a wrong relation, a flipped negation, or a swapped number can
  still score high (false "supported").
- When the answer had no model-written citation, the app appends the top chunk's
  ID, so the check scores against that fallback chunk.
- Very short sentences (one or two content words) swing between 0 and 1.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

SUPPORT_THRESHOLD = 0.5

_CITATION = re.compile(r"\[([^\[\]]+)\]")
# A sentence ends after . ! ? (or after a citation that follows one) when the next
# sentence starts; a citation right after the period stays with that sentence.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|(?<=\])\s+(?=[A-Z0-9\"'(])|\n+")
_WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    """a an and are as at be been being but by can could did do does for from had has have
    how i if in into is it its may might more most must no not of on or our shall should so
    such than that the their them then there these they this those to too was we were what
    when where which while who whom why will with would you your also only both each other
    any all about over under between within without very just using used use""".split()
)


def content_tokens(text: str) -> List[str]:
    tokens = []
    for word in _WORD.findall(text.lower()):
        if word in STOPWORDS:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        tokens.append(word)
    return tokens


def split_answer_sentences(answer: str) -> List[str]:
    parts = [part.strip() for part in _SENTENCE_END.split(answer)]
    # A sentence that is only a citation ("[id]") belongs to the one before it.
    sentences: List[str] = []
    for part in parts:
        if not part:
            continue
        if sentences and not _CITATION.sub("", part).strip(" .,;:"):
            sentences[-1] = f"{sentences[-1]} {part}"
        else:
            sentences.append(part)
    return sentences


def overlap(sentence: str, chunk_text: str) -> float:
    sentence_tokens = set(content_tokens(_CITATION.sub(" ", sentence)))
    if not sentence_tokens:
        return 1.0
    chunk_tokens = set(content_tokens(chunk_text))
    return len(sentence_tokens & chunk_tokens) / len(sentence_tokens)


def check_answer(answer: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Score each sentence of *answer* against the chunk it cites.

    Returns ``{"sentences": [...], "scored": n, "supported": n, "uncited": n}``.
    Citations to IDs not in *chunks* count as uncited.
    """
    text_by_id = {
        (chunk.get("chunk_id") or chunk.get("id")): chunk.get("text", "") for chunk in chunks
    }
    sentences = split_answer_sentences(answer)
    own = [
        [cid for cid in _CITATION.findall(sentence) if cid in text_by_id] for sentence in sentences
    ]
    results = []
    for index, sentence in enumerate(sentences):
        cited = own[index]
        source = "own"
        if not cited:
            following = next((ids for ids in own[index + 1 :] if ids), None)
            previous = next((ids for ids in reversed(own[:index]) if ids), None)
            cited = following or previous or []
            source = "next" if following else ("previous" if previous else "none")
        if not cited:
            results.append({"sentence": sentence, "cited": [], "source": "none", "score": None, "supported": False})
            continue
        # Several citations: the sentence is supported by their combined text.
        score = overlap(sentence, " ".join(text_by_id[cid] for cid in cited))
        results.append(
            {
                "sentence": sentence,
                "cited": cited,
                "source": source,
                "score": round(score, 4),
                "supported": score >= SUPPORT_THRESHOLD,
            }
        )
    return {
        "sentences": results,
        "scored": sum(1 for r in results if r["score"] is not None),
        "supported": sum(1 for r in results if r["supported"]),
        "uncited": sum(1 for r in results if r["source"] == "none"),
    }
