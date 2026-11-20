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
| 2026-11-10 | Keyword only (BM25), top 20 | 1.00 | 0.769 | Not measured | Not measured | Recall@1 0.633, @3 0.867. Retrieval only. `eval/results_keyword.json` |
| 2026-11-10 | Vector only (nomic-embed-text, cosine), top 20 | 1.00 | 0.790 | Not measured | Not measured | Recall@1 0.667, @3 0.900. Retrieval only. `eval/results_vector.json` |
| 2026-11-10 | Hybrid (BM25 + vector, RRF k = 60), top 4 to mistral, not-found threshold 0.605 | 1.00 | 0.803 | Not measured | 20% (2 of 10) | Recall@1 0.633, @3 0.967. Meets the Recall@5 target; misses the hallucination target (< 10%). 4 of 30 answerable questions wrongly refused. `eval/results_hybrid.json` |
| — | Hybrid + reranker | — | — | — | — | Not run yet |

Set: `eval/questions.jsonl`, 40 questions (30 answerable, 10 unanswerable), frozen since 2026-11-10. Corpus: the 2 PDFs in `data/` (73 chunks).

### Misses (hybrid, with answers)

**Retrieval.** All 30 answerable questions have a gold passage in the top 5. Eleven do not have it at rank 1 (q11, q13, q15, q16, q17, q18, q25, q29, q30 at rank 2; q12 at 3; q14 at 4). Fusion helps most at rank 3 (0.867 keyword → 0.967 hybrid) but not at rank 1.

**Answers given to unanswerable questions (hallucination).**

| ID | Question | Top vector score | Cause |
|----|----------|------------------|-------|
| q32 | How do you deploy a Java app to Kubernetes? | 0.622 | Score passes the 0.605 threshold (shares "Java" with the corpus); the model answered from general knowledge. The citation fallback in `src/answer.py` then attached the top chunk's citation, so the zero-citation check did not catch it. |
| q37 | How do you set up PostgreSQL replication? | 0.685 | Same pattern: database wording is close to the JDBC chunks, so the threshold passed it, and the model answered from general knowledge with a fallback citation. |

**Answerable questions wrongly refused.**

| ID | Question | Gold rank | Top vector score | Cause |
|----|----------|-----------|------------------|-------|
| q03 | Which company acquired Sun Microsystems in 2010? | 1 | 0.511 | Below the 0.605 threshold, so refused before generation even though the right passage ranked first. |
| q11 | What is the role of the finally block? | 2 | 0.544 | Below threshold. |
| q13 | What is encapsulation? | 2 | 0.542 | Below threshold. Short definition questions score low. |
| q30 | What is a ResultSet? | 2 | 0.511 | Below threshold. |

**Finding (2026-11-20).** On this set the top vector score does not separate answerable from unanswerable questions: the 4 refused answerable questions score 0.51–0.54, while the 2 hallucinated unanswerable ones score 0.62–0.69. No single threshold fixes both. Moving it lower trades false refusals for more hallucinations. Next steps to evaluate: drop the citation fallback for answers with no valid citation, add a reranker score as the not-found signal, and measure faithfulness and citation accuracy.

Scores recomputed 2026-11-20 with the same corpus, embedding model, and query prefix (embedding the 73 chunks took 166 s on this machine).

### Experiment: citation fallback x not-found threshold (2026-11-20)

One run of the hybrid pipeline (same config as above) with the threshold disabled during generation, so every question reached mistral. Each variant is scored on the same 40 answers: "fallback off" turns an answer with no model-written citation into "not found"; "threshold on" refuses when the top vector score is below 0.605.

| Citation fallback | Threshold 0.605 | Hallucination (of 10) | Answerable wrongly refused (of 30) |
|---|---|---|---|
| On | On (current app setting) | 2 (q32, q37) | 4 (q03, q11, q13, q30) |
| On | Off | 7 | 0 |
| Off | On | 0 | 22 |
| Off | Off | 0 | 19 |

**Result: keep both on.** Dropping the fallback removes hallucinations but makes the system unusable, because mistral leaves out citations on most correct answers: the fallback was needed on 19 of 30 answerable questions (and on all 10 unanswerable ones). Turning the threshold off removes the 4 false refusals but lets 7 of 10 unanswerable questions through.

**Other findings from the raw answers.**
- mistral never returned the exact refusal string. When it declined it wrote variations such as "Not found in your documents. The provided chunks do not contain…" or "The capital of France is not found in your documents." (q35, q36, q40), and the fallback then attached a citation to them.
- Several answers say the topic is "not explicitly mentioned in the provided chunks" and then answer from general knowledge anyway (q33, q38).

**Next levers** (not the fallback): get the model to cite reliably (prompt examples or a model that follows citation instructions better), then retest dropping the fallback; or use a reranker score as the not-found signal instead of the raw vector score.
