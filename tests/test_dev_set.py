"""Development set for prompt optimization: determinism, shape, and no benchmark leak.

The dev set is what DSPy *trains* on, so the properties that matter are the
ones that make a training population trustworthy: it reproduces exactly at a
given seed, it is drawn from the same Hunspell path the scored harness uses,
and its construction never touches the locked benchmark.
"""

from __future__ import annotations

import pytest

from spelling_reranker.dev_set import (
    DevExample,
    build_dev_set,
    cache_path,
    composition_stats,
    format_stats,
    load_dev_set,
    load_misspelling_pairs,
    save_dev_set,
    split_sentences,
    usable_sentences,
)

SENTENCES = [
    "The committee received the necessary information yesterday .",
    "Children believe their teacher because the explanation is clear .",
    "Government officials announced another successful mission today .",
    "The government published the necessary information for the committee .",
    "Researchers describe the experiment as a definitely separate achievement .",
    "Her colleague recommended the restaurant across the street this morning .",
]


class FakeEngine:
    """Hunspell stand-in: everything in `known` is a word, the rest gets suggestions."""

    def __init__(self, known: set[str], suggestions: dict[str, list[str]] | None = None) -> None:
        self.known = {w.lower() for w in known}
        self.suggestions = suggestions or {}

    def spell(self, word: str) -> bool:
        return word.lower() in self.known

    def suggest(self, word: str) -> list[str]:
        return list(self.suggestions.get(word.lower(), ["placeholder", "another"]))

    def gold_index(self, candidates, gold):
        for i, cand in enumerate(candidates):
            if cand.lower() == gold.lower():
                return i
        return None


def _engine() -> FakeEngine:
    known = {tok.strip(".,") for s in SENTENCES for tok in s.split()}
    return FakeEngine(known)


def _build(n: int = 12, **kwargs):
    return build_dev_set(n, sentences=SENTENCES, engine=_engine(), **kwargs)


def test_split_sentences_keeps_pretokenised_spacing() -> None:
    """Sentence-final punctuation is its own token and must stay one."""
    text = "He left . She stayed ."
    assert split_sentences(text) == ["He left .", "She stayed ."]


def test_usable_sentences_drops_headings_and_markup() -> None:
    text = "\n".join(
        [
            " = = Description = = ",
            " The lobster is blue @-@ green and lives in cold water off the coast .",
            " Short one .",
            " A perfectly ordinary sentence of about the right length for this purpose .",
        ]
    )
    kept = usable_sentences(text, min_tokens=6, max_tokens=40)
    assert kept == ["A perfectly ordinary sentence of about the right length for this purpose ."]


def test_misspelling_pairs_are_authentic_and_sorted() -> None:
    pairs = load_misspelling_pairs()
    assert len(pairs) > 1000, "the vendored list should yield thousands of pairs"
    assert pairs == sorted(pairs)
    assert ("recieve", "receive") in pairs
    assert all(bad.lower() != good.lower() for bad, good in pairs)


def test_dev_set_is_deterministic_at_a_fixed_seed() -> None:
    a = _build(seed=7)
    b = _build(seed=7)
    assert [ex.to_dict() for ex in a.examples] == [ex.to_dict() for ex in b.examples]


def test_a_different_seed_gives_a_different_sample() -> None:
    a = _build(seed=7)
    b = _build(seed=8)
    assert [ex.typo for ex in a.examples] != [ex.typo for ex in b.examples]


def test_contexts_reconstruct_the_sentence_exactly() -> None:
    """The prompt builders splice on this invariant; a stray space breaks scoring."""
    for ex in _build().examples:
        assert ex.context_before + ex.typo + ex.context_after == ex.sentence
        assert ex.typo in ex.sentence


def test_examples_are_only_typos_hunspell_actually_flags() -> None:
    engine = _engine()
    for ex in _build().examples:
        assert not engine.spell(ex.typo), f"{ex.typo} is a dictionary word, not an error"
        assert ex.candidates, "an example with no suggestions is outside the scored population"


def test_gold_is_never_the_typo_and_gold_index_agrees_with_the_pool() -> None:
    for ex in _build().examples:
        assert ex.gold.lower() != ex.typo.lower()
        if ex.gold_index is None:
            assert all(c.lower() != ex.gold.lower() for c in ex.candidates)
        else:
            assert ex.candidates[ex.gold_index].lower() == ex.gold.lower()


def test_authentic_fraction_controls_the_mix() -> None:
    dev = _build(24, authentic_fraction=0.0)
    assert all(ex.source == "synthetic" for ex in dev.examples)
    with pytest.raises(ValueError):
        _build(4, authentic_fraction=1.5)


def test_require_known_gold_rejects_out_of_dictionary_golds() -> None:
    """A typo of an unknown proper noun is corpus noise, not a correctable error."""
    engine = FakeEngine(known={"committee"})
    dev = build_dev_set(
        8, sentences=SENTENCES, engine=engine, authentic_fraction=0.0, require_known_gold=True
    )
    assert all(ex.gold.lower() == "committee" for ex in dev.examples)


def test_composition_stats_report_the_population_shape() -> None:
    dev = _build(16)
    stats = composition_stats(dev.examples)
    assert stats["n"] == len(dev.examples)
    ed = stats["edit_distance"]
    assert abs(ed["ed1_pct"] + ed["ed2_pct"] + ed["ed3plus_pct"] - 100.0) < 1e-6
    assert 0.0 <= stats["gold_in_pool_pct"] <= 100.0
    assert stats["sources"]["authentic"] + stats["sources"]["synthetic"] == stats["n"]
    assert "gold in pool" in format_stats(stats)


def test_composition_stats_on_an_empty_set() -> None:
    assert composition_stats([]) == {"n": 0}
    assert "empty" in format_stats({"n": 0})


def test_split_is_a_partition_in_order() -> None:
    dev = _build(20)
    train, val = dev.split(0.5)
    assert train + val == dev.examples
    assert len(train) == round(len(dev.examples) * 0.5)


def test_cache_round_trip_preserves_examples(tmp_path) -> None:
    dev = _build(10)
    path = cache_path(tmp_path, 10, 1337, 0.5)
    save_dev_set(dev, path)
    reloaded = load_dev_set(path)
    assert [ex.to_dict() for ex in reloaded.examples] == [ex.to_dict() for ex in dev.examples]
    assert reloaded.stats == dev.stats
    assert isinstance(reloaded.examples[0], DevExample)
    assert isinstance(reloaded.examples[0].candidates, tuple)
