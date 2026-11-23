"""Measure the semantic cache end to end through POST /ask (real models, throwaway schema).

1. Upload the eval PDF (data/Core_Java_Complete_Answers.pdf).
2. Ask every ``question_a`` in eval/cache_pairs.jsonl once (generates and caches the
   answer unless it is a refusal).
3. Ask every ``question_b`` once and record whether it was served from the cache and how.
4. Latency: ask each cached ``question_a`` again (exact hits) and each ``question_b``
   that hit semantically 5 more times (semantic hits); report p50/p95 of /ask time.

Uses the [cache] settings in config.toml. Results are printed and saved to
storage/cache_checks/ (git-ignored).

    python -m scripts.measure_cache
"""

from __future__ import annotations

import datetime
import json
import statistics
import time
from pathlib import Path

from scripts.check_utils import ROOT, isolated_app
from src import cache

PDF = ROOT / "data" / "Core_Java_Complete_Answers.pdf"


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def main() -> None:
    pairs = [json.loads(l) for l in (ROOT / "eval" / "cache_pairs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    enabled, threshold = cache.semantic_settings(ROOT / "config.toml")
    print(f"semantic cache enabled={enabled} threshold={threshold}")
    out: dict = {"threshold": threshold, "pairs": []}
    with isolated_app() as (client, _api, _url):
        upload = client.post("/upload", files=[("files", (PDF.name, PDF.read_bytes(), "application/pdf"))])
        assert upload.status_code == 200, upload.text
        doc_id = upload.json()["documents"][0]["id"]

        def ask(question: str) -> tuple[dict, float]:
            start = time.perf_counter()
            response = client.post("/ask", json={"question": question, "doc_ids": [doc_id]})
            elapsed = time.perf_counter() - start
            assert response.status_code == 200, response.text
            return response.json(), elapsed

        for p in pairs:
            a, _ = ask(p["question_a"])
            p["a_refused"] = cache.is_refusal(a["answer"])
            p["a_answer"] = a["answer"]
        for p in pairs:
            b, seconds = ask(p["question_b"])
            p["b_cached"] = b["cached"]
            p["b_match"] = b.get("cache_match")
            p["b_seconds"] = seconds
            p["b_same_answer_as_a"] = b["answer"] == p["a_answer"]
            out["pairs"].append({k: p[k] for k in ("id", "group", "a_refused", "b_cached", "b_match", "b_seconds", "b_same_answer_as_a")})

        exact_times = [ask(p["question_a"])[1] for p in pairs if not p["a_refused"]]
        semantic_times = []
        for p in pairs:
            if p["b_match"] == "semantic":
                for _ in range(5):
                    r, seconds = ask(p["question_b"])
                    assert r["cache_match"] == "semantic"
                    semantic_times.append(seconds)

    para = [p for p in pairs if p["group"] == "paraphrase"]
    near = [p for p in pairs if p["group"] == "near_miss"]
    para_cached_a = [p for p in para if not p["a_refused"]]
    para_hits = [p for p in para if p["b_match"] == "semantic"]
    near_cached_a = [p for p in near if not p["a_refused"]]
    near_false = [p for p in near if p["b_cached"]]
    summary = {
        "threshold": threshold,
        "paraphrase_hits": f"{len(para_hits)}/{len(para)}",
        "paraphrase_hits_where_a_was_cached": f"{len(para_hits)}/{len(para_cached_a)}",
        "near_miss_false_hits": f"{len(near_false)}/{len(near)}",
        "near_miss_pairs_where_a_was_cached": len(near_cached_a),
        "refused_question_a": [p["id"] for p in pairs if p["a_refused"]],
        "semantic_hit_ids": [p["id"] for p in para_hits],
        "near_miss_false_hit_ids": [p["id"] for p in near_false],
        "semantic_hit_latency_s": {"n": len(semantic_times), "p50": round(pct(semantic_times, 50), 3), "p95": round(pct(semantic_times, 95), 3)} if semantic_times else None,
        "exact_hit_latency_s": {"n": len(exact_times), "p50": round(pct(exact_times, 50), 3), "p95": round(pct(exact_times, 95), 3)} if exact_times else None,
        "miss_latency_s_question_b": {"n": len([p for p in pairs if not p["b_cached"]]), "p50": round(statistics.median([p["b_seconds"] for p in pairs if not p["b_cached"]]), 3)},
    }
    out["summary"] = summary
    print(json.dumps(summary, indent=1))
    target = ROOT / "storage" / "cache_checks"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"cache_measure_{datetime.datetime.now():%Y-%m-%d_%H%M%S}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"Saved to {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
