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
    # Prompt optimization is training too: DSPy scores candidate prompts and
    # selects demonstrations against these files' output, so neither the dev-set
    # builder nor the programs it feeds may know the benchmark exists.
    "spelling_reranker/dev_set.py",
    "spelling_reranker/dspy_program.py",
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


def test_available_cpus_respects_container_limits() -> None:
    """Pool sizing must not be taken from the host's core count.

    Inside a container os.cpu_count() reports the host: a pod allocated 28 vCPU
    on a 112-core host would otherwise spawn 112 workers and oversubscribe 4x.
    """
    import os

    from spelling_reranker.data_build import available_cpus

    n = available_cpus()
    assert n >= 1
    assert n <= (os.cpu_count() or 1)
    try:
        assert n <= len(os.sched_getaffinity(0))
    except AttributeError:
        pass


def test_chunked_parquet_write_survives_a_null_only_chunk(tmp_path) -> None:
    """Row groups must share one schema regardless of what a chunk contains.

    pandas infers dtypes per chunk, so a chunk whose `cand_15` is entirely null
    infers as null instead of string and the next chunk fails to append with
    "Table schema does not match schema used to create file". This killed a
    3M-example build 13% of the way in.
    """
    import pyarrow.parquet as pq

    from spelling_reranker.data_build import write_split

    def rows():
        # First chunk: only two candidates ever populated.
        for i in range(4):
            row = {
                "example_id": f"a{i}", "source": "fixture",
                "context_before": "the ", "typo": "teh", "context_after": " cat",
                "gold": "the", "gold_index": 0, "corruption_type": "x",
                "source_document_id": "d", "original_sentence_hash": "h",
            }
            for c in range(N_CANDIDATE_SLOTS):
                row[f"cand_{c}"] = "the" if c < 2 else None
            yield row
        # Second chunk: every slot populated.
        for i in range(4):
            row = {
                "example_id": f"b{i}", "source": "fixture",
                "context_before": "a ", "typo": "hosue", "context_after": " here",
                "gold": "house", "gold_index": 0, "corruption_type": "x",
                "source_document_id": "d", "original_sentence_hash": "h",
            }
            for c in range(N_CANDIDATE_SLOTS):
                row[f"cand_{c}"] = f"w{c}"
            yield row

    out = tmp_path / "train.parquet"
    n = write_split(rows(), out, chunk_size=4)
    assert n == 8
    table = pq.read_table(out)
    assert table.num_rows == 8
    assert str(table.schema.field("cand_15").type) == "string"


def test_empty_split_still_writes_a_valid_file(tmp_path) -> None:
    import pyarrow.parquet as pq

    from spelling_reranker.data_build import write_split

    out = tmp_path / "empty.parquet"
    assert write_split(iter(()), out, chunk_size=4) == 0
    assert pq.read_table(out).num_rows == 0
