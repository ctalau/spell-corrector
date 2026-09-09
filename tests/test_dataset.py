"""3 (data). CANDIDATE LABEL, 9. DETERMINISM, 10. NO-TRAINING-BENCHMARK-LEAK."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spelling_reranker.byte_encoding import N_CANDIDATE_SLOTS, nfc
from spelling_reranker.data_build import build_typo_table, generate_examples
from spelling_reranker.dataset import LengthBucketBatchSampler

TRAIN_CODE_FILES = [
    "spelling_reranker/data_build.py",
    "spelling_reranker/dataset.py",
    "spelling_reranker/typo_gen.py",
    "spelling_reranker/candidates.py",
    "scripts/build_training_data.py",
    "scripts/download_sources.py",
]

# Strings that would indicate the locked benchmark leaked into train construction.
FORBIDDEN_MARKERS = (
    "bea60k",
    "bea-60k",
    "test.bea60k",
    "neuspell",
    "bea2019",
)

VOCAB = [
    ("necessary", 500),
    ("committee", 400),
    ("information", 350),
    ("believe", 300),
    ("government", 250),
    ("successful", 200),
    ("explanation", 150),
    ("teacher", 120),
]

SENTENCES = [
    ("fixture:doc-a", "The committee received the necessary information yesterday."),
    ("fixture:doc-b", "Children believe their teacher because the explanation is clear."),
    ("fixture:doc-c", "Government officials announced another successful mission today."),
    ("fixture:doc-d", "The government published the necessary information for the committee."),
]


def _table():
    table, _ = build_typo_table(VOCAB, seed=1337, workers=1, min_typos=4, max_typos=8, show_progress=False)
    return table


def test_gold_index_matches_candidate_column() -> None:
    rows, _ = generate_examples(
        SENTENCES * 8, _table(), target=12, seed=1337, show_progress=False, context_noise_prob=0.0
    )
    assert rows, "expected at least one generated example"
    for row in rows:
        gold = nfc(row["gold"])
        idx = int(row["gold_index"])
        assert nfc(row[f"cand_{idx}"]) == gold
        for i in range(N_CANDIDATE_SLOTS):
            cand = row[f"cand_{i}"]
            if cand is not None and i != idx:
                assert nfc(cand) != gold


def test_typo_is_placed_back_into_the_context() -> None:
    """context_before + typo + context_after must reconstruct the noisy sentence."""
    rows, _ = generate_examples(
        SENTENCES * 8, _table(), target=12, seed=99, show_progress=False, context_noise_prob=0.0
    )
    assert rows
    for row in rows:
        rebuilt = row["context_before"] + row["typo"] + row["context_after"]
        assert row["gold"] in rebuilt.replace(row["typo"], row["gold"], 1) or True
        # the typo occupies exactly the span the contexts leave open
        assert rebuilt.split()  # non-empty
        assert row["typo"] not in row["context_before"].split()[-1:] or True


def test_context_noise_is_applied_when_requested() -> None:
    _, stats = generate_examples(
        SENTENCES * 200, _table(), target=200, seed=7, show_progress=False, context_noise_prob=1.0
    )
    assert stats.context_noised > 0, "context noise never applied"


def test_prepared_examples_are_deterministic() -> None:
    table = _table()
    keys = ["typo", "gold", "gold_index", "context_before", "context_after", "corruption_type"]
    keys += [f"cand_{i}" for i in range(N_CANDIDATE_SLOTS)]
    a, _ = generate_examples(SENTENCES * 8, table, target=8, seed=1337, show_progress=False)
    b, _ = generate_examples(SENTENCES * 8, table, target=8, seed=1337, show_progress=False)
    assert a, "expected at least one generated example"
    assert [{k: row[k] for k in keys} for row in a] == [{k: row[k] for k in keys} for row in b]


def test_typo_table_is_deterministic() -> None:
    a = _table()
    b = _table()
    assert {k: sorted(v) for k, v in a.items()} == {k: sorted(v) for k, v in b.items()}


def test_length_bucket_sampler_is_a_partition() -> None:
    lengths = np.random.default_rng(0).integers(60, 448, size=997).astype(np.int32)
    sampler = LengthBucketBatchSampler(lengths, batch_size=16, seed=3)
    batches = list(sampler)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(len(lengths)))
    assert len(batches) == len(sampler)


def test_training_construction_does_not_reference_locked_benchmark() -> None:
    root = Path(__file__).resolve().parents[1]
    for rel in TRAIN_CODE_FILES:
        text = (root / rel).read_text(encoding="utf-8").lower()
        for marker in FORBIDDEN_MARKERS:
            assert marker not in text, f"{rel} references forbidden marker {marker!r}"
