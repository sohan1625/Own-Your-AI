# Evaluation

## Eval set

- 30–50 questions, written from your own documents.
- Each question has:
  - `id`
  - `question`
  - `answerable` (true or false)
  - `gold_passages` (file, page, span) if answerable
  - `reference_answer` if answerable
  - `notes` (why it's hard, if it is)
- Include at least 8 unanswerable questions, so hallucination can be measured.
- Include a mix: single-fact, multi-part, paraphrased, and cross-document.
- Store the set as `eval/questions.jsonl`. Freeze it before tuning. Changing it after tuning invalidates the results.

## Metrics

**Retrieval**
- Precision@k and recall@k (k = 1, 3, 5) against gold passages
- Mean reciprocal rank (MRR)

**Answers**
- Correctness: matches the reference answer (human-labeled on a sample)
- Faithfulness: every claim is supported by the retrieved text
- Citation accuracy: the cited passage contains the claim
- Hallucination rate: answers given to unanswerable questions (should be "not found")

## Comparisons to run

1. Keyword only
2. Vector only
3. Hybrid (RRF)
4. Hybrid + reranker

Report each on the same frozen set.

## Targets (v1)

| Metric | Target |
|--------|--------|
| Recall@5, hybrid | ≥ 0.80 |
| Faithfulness | ≥ 0.90 |
| Hallucination on unanswerable | < 10% |
| Answers with valid citation | 100% |

## Results log

| Date | Config | Recall@5 | MRR | Faithfulness | Hallucination | Notes |
|------|--------|----------|-----|--------------|---------------|-------|
| | | | | | | |

Record misses too, with a short explanation of the cause.
