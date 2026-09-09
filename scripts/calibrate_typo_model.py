#!/usr/bin/env python3
"""Calibrate the synthetic typo generator against a public misspelling list.

The generator's realism must be checked against *some* corpus of authentic
human misspellings. Using the held-out benchmark for that would quietly turn it
into a development set, so this script uses Wikipedia's
"Lists of common misspellings" instead: ~4.3k real misspelling -> correction
pairs, public, and unrelated to any benchmark.

It reports, for both the real list and freshly generated synthetic typos:

  * the Levenshtein distance mixture (the property the generator is tuned to),
  * how deep in Hunspell's suggestion list the correction sits.

Run it after touching `spelling_reranker/typo_gen.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spelling_reranker.byte_encoding import N_CANDIDATE_SLOTS
from spelling_reranker.candidates import build_pool, gold_index
from spelling_reranker.data_build import edit_distance
from spelling_reranker.hunspell import default_engine
from spelling_reranker.seed import DEFAULT_SEED
from spelling_reranker.serialization import MAX_CANDIDATE_BYTES
from spelling_reranker.typo_gen import corrupt_word, is_eligible_word

SOURCE_URL = (
    "https://en.wikipedia.org/wiki/"
    "Wikipedia:Lists_of_common_misspellings/For_machines?action=raw"
)


def load_pairs(cache: Path) -> list[tuple[str, str]]:
    if not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(SOURCE_URL, timeout=120)
        response.raise_for_status()
        cache.write_text(response.text, encoding="utf-8")
    pairs: list[tuple[str, str]] = []
    for line in cache.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "->" not in line or line.startswith("="):
            continue
        bad, _, good = line.partition("->")
        bad = bad.strip()
        good = good.split(",")[0].strip()
        if bad.isalpha() and good.isalpha() and is_eligible_word(bad):
            pairs.append((bad, good))
    return pairs


def profile(pairs: list[tuple[str, str]], engine) -> dict:
    ed: Counter[int] = Counter()
    rank: Counter[str] = Counter()
    n_sugg: list[int] = []
    for typo, gold in pairs:
        ed[edit_distance(typo.lower(), gold.lower(), cap=3)] += 1
        if engine.spell(typo):
            rank["not_flagged"] += 1
            continue
        suggestions = engine.suggest(typo)
        n_sugg.append(len(suggestions))
        pool = build_pool(suggestions, limit=N_CANDIDATE_SLOTS, max_bytes=MAX_CANDIDATE_BYTES)
        gi = gold_index(pool, gold)
        if gi is None:
            rank["absent"] += 1
        elif gi == 0:
            rank["0"] += 1
        elif gi < 10:
            rank["1-9"] += 1
        else:
            rank["10-15"] += 1
    total = sum(ed.values())
    total_rank = sum(rank.values())
    return {
        "n": total,
        "edit_distance_pct": {str(k): round(100 * v / total, 2) for k, v in sorted(ed.items())},
        "hunspell_gold_rank_pct": {k: round(100 * v / total_rank, 2) for k, v in sorted(rank.items())},
        "mean_suggestions": round(float(np.mean(n_sugg)), 2) if n_sugg else None,
        "pct_with_10plus_suggestions": (
            round(100 * sum(1 for x in n_sugg if x >= 10) / len(n_sugg), 2) if n_sugg else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "raw" / "wikipedia_misspellings.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "typo_calibration.json")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-synthetic", type=int, default=4000)
    args = parser.parse_args()

    engine = default_engine()
    real = load_pairs(args.cache)
    print(f"real misspelling pairs: {len(real)}")

    rng = np.random.default_rng(args.seed)
    golds = [good for _, good in real]
    synthetic: list[tuple[str, str]] = []
    while len(synthetic) < args.n_synthetic:
        word = golds[int(rng.integers(0, len(golds)))]
        if not engine.spell(word):
            continue
        typo, _ = corrupt_word(word, rng)
        if typo == word or engine.spell(typo):
            continue
        synthetic.append((typo, word))

    payload = {
        "source": SOURCE_URL,
        "note": "Public misspelling list, independent of any held-out benchmark.",
        "real": profile(real, engine),
        "synthetic": profile(synthetic, engine),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
