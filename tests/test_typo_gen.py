"""The synthetic typo generator must resemble authentic human misspellings.

The reference mixture comes from Wikipedia's public "Lists of common
misspellings" (see scripts/calibrate_typo_model.py), never from a held-out
benchmark. If these bounds start failing, the generator drifted and the model
is being trained on the wrong error distribution.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from spelling_reranker.data_build import edit_distance
from spelling_reranker.typo_gen import CORRUPTION_PROBS, corrupt_word, is_eligible_word

# Authentic reference: ED1 72.6% / ED2 25.0% / ED3+ 2.4%.
ED1_BOUNDS = (0.62, 0.80)
ED2_BOUNDS = (0.15, 0.33)

WORDS = [
    "necessary", "committee", "information", "believe", "government",
    "successful", "explanation", "teacher", "separate", "definitely",
    "receive", "tomorrow", "restaurant", "beautiful", "occurrence",
    "immediately", "knowledge", "dilemma", "rhythm", "length",
]


def _sample(n: int = 4000) -> list[tuple[str, str]]:
    rng = np.random.default_rng(1337)
    out: list[tuple[str, str]] = []
    while len(out) < n:
        word = WORDS[int(rng.integers(0, len(WORDS)))]
        typo, _ = corrupt_word(word, rng)
        if typo != word:
            out.append((typo, word))
    return out


def test_edit_distance_mixture_matches_authentic_misspellings() -> None:
    counts: Counter[int] = Counter()
    for typo, gold in _sample():
        counts[edit_distance(typo.lower(), gold.lower(), cap=3)] += 1
    total = sum(counts.values())
    ed1 = counts[1] / total
    ed2 = counts[2] / total
    assert ED1_BOUNDS[0] <= ed1 <= ED1_BOUNDS[1], f"ED1 fraction {ed1:.3f} out of range"
    assert ED2_BOUNDS[0] <= ed2 <= ED2_BOUNDS[1], f"ED2 fraction {ed2:.3f} out of range"
    assert counts[3] / total > 0.005, "generator produces no ED3+ typos at all"


def test_multi_edit_typos_are_produced() -> None:
    rng = np.random.default_rng(7)
    labels = [corrupt_word("information", rng)[1] for _ in range(500)]
    assert any("+" in label for label in labels), "no composed multi-edit typos"


def test_every_corruption_type_can_fire() -> None:
    rng = np.random.default_rng(11)
    seen: set[str] = set()
    for kind in CORRUPTION_PROBS:
        for word in WORDS:
            typo, label = corrupt_word(word, rng, corruption_type=kind, n_edits=1)
            if typo != word and label == kind:
                seen.add(kind)
                break
    assert seen == set(CORRUPTION_PROBS), f"never fired: {set(CORRUPTION_PROBS) - seen}"


def test_typos_stay_word_shaped() -> None:
    for typo, gold in _sample(1000):
        assert typo, "empty typo"
        assert typo != gold
        assert is_eligible_word(gold)
        assert len(typo) <= len(gold) + 4
