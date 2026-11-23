"""Choose the semantic-cache threshold on eval/cache_pairs.jsonl only.

Embeds every question with the app's embedding function (search_query prefix) and
simulates a cache that holds all ``question_a`` entries. For each ``question_b`` it
finds the closest cached question:

- a paraphrase pair is a hit when the closest entry is its own ``question_a`` and the
  similarity is at or above the threshold;
- a near-miss pair is a false hit when *any* entry is at or above the threshold.

Rule (fixed in advance): the threshold is the lowest value with zero near-miss false
hits, plus a 0.005 margin, rounded up to 3 decimals. eval/questions.jsonl is not read.

    python -m scripts.calibrate_cache
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from src.index import embed_text, get_embedding_prefix
from src.retrieve import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
MARGIN = 0.005


def load_pairs(path: Path = ROOT / "eval" / "cache_pairs.jsonl") -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def choose_threshold(near_miss_best: list[float]) -> float:
    return math.ceil((max(near_miss_best) + MARGIN) * 1000) / 1000


def main() -> None:
    config = ROOT / "config.toml"
    prefix = get_embedding_prefix("query", config_path=config)
    pairs = load_pairs()
    cache: dict[str, list[float]] = {}

    def embed(text: str) -> list[float]:
        if text not in cache:
            cache[text] = embed_text(text, prefix=prefix, config_path=config)
        return cache[text]

    stored = {p["id"]: embed(p["question_a"]) for p in pairs}
    rows = []
    for p in pairs:
        vb = embed(p["question_b"])
        sims = {pid: cosine_similarity(vb, va) for pid, va in stored.items()}
        own = sims[p["id"]]
        best_id = max(sims, key=lambda pid: (sims[pid], pid == p["id"]))
        rows.append({"id": p["id"], "group": p["group"], "own": own, "best_id": best_id, "best": sims[best_id]})

    near_best = [r["best"] for r in rows if r["group"] == "near_miss"]
    threshold = choose_threshold(near_best)
    para = [r for r in rows if r["group"] == "paraphrase"]
    para_hits = [r for r in para if r["best_id"] == r["id"] and r["best"] >= threshold]

    print(f"{'id':<5} {'group':<10} {'own_sim':>8} {'best_sim':>8} best_match")
    for r in sorted(rows, key=lambda r: (r["group"], -r["own"])):
        print(f"{r['id']:<5} {r['group']:<10} {r['own']:8.4f} {r['best']:8.4f} {r['best_id']}")
    print()
    print(f"near-miss: max best similarity {max(near_best):.4f}, min {min(near_best):.4f}")
    print(f"paraphrase: own similarity min {min(r['own'] for r in para):.4f}, max {max(r['own'] for r in para):.4f}")
    print(f"chosen threshold (max near-miss + {MARGIN}, rounded up): {threshold:.3f}")
    print(f"paraphrase hits at threshold: {len(para_hits)}/{len(para)}")
    print(f"near-miss false hits at threshold: {sum(1 for b in near_best if b >= threshold)}/{len(near_best)}")


if __name__ == "__main__":
    main()
