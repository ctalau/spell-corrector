"""3 (data). CANDIDATE LABEL, 9. DETERMINISM, 10. NO-TRAINING-BENCHMARK-LEAK."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spelling_reranker.byte_encoding import nfc
from spelling_reranker.data_build import BuildStats, generate_examples, try_make_example
from spelling_reranker.hunspell import HunspellEngine
from spelling_reranker.typo_gen import tokenize_sentence

TRAIN_CODE_FILES = [
    "spelling_reranker/data_build.py",
    "spelling_reranker/dataset.py",
    "spelling_reranker/typo_gen.py",
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


MINI_JOBS = [
    {
        "sentence": "The committee received the necessary information yesterday.",
        "tokens": tokenize_sentence("The committee received the necessary information yesterday."),
        "eligible": None,
        "document_id": "fixture:doc-a",
        "source": "fixture-synthetic",
        "sent_i": 0,
    },
    {
        "sentence": "Children believe their teacher because the explanation is clear.",
        "tokens": tokenize_sentence("Children believe their teacher because the explanation is clear."),
        "eligible": None,
        "document_id": "fixture:doc-b",
        "source": "fixture-synthetic",
        "sent_i": 0,
    },
    {
        "sentence": "Government officials announced another successful mission today.",
        "tokens": tokenize_sentence("Government officials announced another successful mission today."),
        "eligible": None,
        "document_id": "fixture:doc-c",
        "source": "fixture-synthetic",
        "sent_i": 0,
    },
]


def _ready_jobs() -> list[dict]:
    jobs = []
    for job in MINI_JOBS:
        item = dict(job)
        item["eligible"] = list(range(len(item["tokens"])))
        jobs.append(item)
    return jobs


def test_gold_index_matches_candidate_column() -> None:
    engine = HunspellEngine()
    rng = np.random.default_rng(1337)
    sentence = "The necessary information arrived yesterday."
    tokens = tokenize_sentence(sentence)
    stats = BuildStats()
    found = None
    for span in tokens:
        found = try_make_example(
            sentence,
            span,
            engine=engine,
            rng=rng,
            source="fixture-synthetic",
            document_id="fixture:doc",
            example_prefix="fixture",
            max_seq_len=384,
            stats=stats,
        )
        if found is not None:
            break
    assert found is not None, "could not construct a usable fixture example"
    gold = nfc(found["gold"])
    idx = int(found["gold_index"])
    assert nfc(found[f"cand_{idx}"]) == gold
    for i in range(10):
        cand = found[f"cand_{i}"]
        if cand is None:
            continue
        if i != idx:
            assert nfc(cand) != gold


def test_prepared_examples_are_deterministic() -> None:
    engine = HunspellEngine()
    jobs = _ready_jobs()
    a, _ = generate_examples(
        jobs,
        target=8,
        seed=1337,
        engine=engine,
        show_progress=False,
        max_attempts_per_sentence=8,
        workers=1,
    )
    b, _ = generate_examples(
        jobs,
        target=8,
        seed=1337,
        engine=engine,
        show_progress=False,
        max_attempts_per_sentence=8,
        workers=1,
    )
    assert a, "expected at least one generated example"
    keys = ["typo", "gold", "gold_index", "context_before", "context_after", "corruption_type"]
    keys += [f"cand_{i}" for i in range(10)]
    assert [{k: row[k] for k in keys} for row in a] == [{k: row[k] for k in keys} for row in b]


def test_training_construction_does_not_reference_locked_benchmark() -> None:
    root = Path(__file__).resolve().parents[1]
    for rel in TRAIN_CODE_FILES:
        text = (root / rel).read_text(encoding="utf-8").lower()
        for marker in FORBIDDEN_MARKERS:
            assert marker not in text, f"{rel} references forbidden marker {marker!r}"
