"""Non-benchmark development set for prompt optimization.

Why this module exists
----------------------
Automatic prompt optimizers (DSPy's ``BootstrapFewShot``/``MIPROv2``) are
*training*: they score candidate prompts on labelled examples and keep the ones
that win. The locked evaluation benchmark this repo is scored on may never be
used that way -- not for training, not for validation, not for few-shot demo
selection, not for picking between programs. So prompt optimization needs its
own development population, built only from material that is already in the
repository and has nothing to do with the locked set:

* ``data/wikipedia_misspellings.txt`` -- ~4.3k authentic human
  ``misspelling -> correction`` pairs from Wikipedia's public
  "Lists of common misspellings" (CC BY-SA, vendored, the same list
  ``scripts/calibrate_typo_model.py`` calibrates the typo generator against);
* ``spelling_reranker/typo_gen.py`` -- the repo's own generator, whose edit
  mixture is itself calibrated to that authentic list;
* WikiText-103 (the corpus ``scripts/download_sources.py`` already fetches) for
  the sentence contexts the typos are dropped into.

Shape matching, and where it is imperfect
-----------------------------------------
An optimizer only ever sees this population, so if its shape differs from the
population the frozen prompt is finally scored on, it optimizes for the wrong
thing. Three properties are deliberately kept aligned, and the builder
*measures* all of them (:func:`composition_stats`) instead of asserting them:

1. **Candidate lists come from the scored path.** Hunspell is queried through
   ``spelling_reranker.hunspell`` exactly as the scored harness queries it, at
   the same ``max_candidates`` (8), in Hunspell's own order. The *gold-in-pool*
   rate is therefore whatever Hunspell actually achieves on these typos -- it is
   reported, never filtered to a target. Filtering it would delete exactly the
   examples where a free-form answer beats a forced choice.
2. **Edit-distance mixture.** Half the examples are authentic misspellings, so
   the mixture is anchored to real human error statistics rather than to the
   generator's priors alone. ``authentic_fraction`` was fixed a priori at 0.5;
   it is not tuned against any held-out score.
3. **Pre-tokenised text.** WikiText-103 raw is Moses-tokenised (`" ... on the
   internet ."`, `"do n't"`), which is the same convention the evaluation text
   uses. This matters more than it sounds: the documented dominant failure of
   whole-sentence-rewrite prompting is the model reflowing pre-tokenised input
   into natural prose and re-attaching punctuation. A dev set of *untokenised*
   prose would never expose that failure, and an optimizer would never fix it.

What is still different, stated plainly: WikiText is encyclopedic prose, while
the locked benchmark is learner English. Register, sentence length and error
density differ, and no material in this repository fixes that. Compare the
numbers printed by :func:`composition_stats` against the population statistics
recorded in ``reports/experiments/02-byte-reranker-87m/RUN_NOTES.md`` section 4 before trusting an optimized prompt; a dev
set with a very different gold-at-0 rate or edit-distance mixture is optimizing
for a different task.

Determinism
-----------
Everything is driven by one seeded ``numpy.random.Generator`` and by sorted
iteration order, so a rebuild at the same seed/size reproduces the same
examples byte-for-byte. Built sets are cached under ``data/dev_set/``
(gitignored) so repeated optimizer runs do not re-query Hunspell.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from spelling_reranker.byte_encoding import nfc
from spelling_reranker.data_build import edit_distance
from spelling_reranker.hunspell import HunspellEngine, default_engine
from spelling_reranker.typo_gen import corrupt_word, is_eligible_word

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MISSPELLINGS = ROOT / "data" / "wikipedia_misspellings.txt"
DEFAULT_CACHE_DIR = ROOT / "data" / "dev_set"
#: WikiText-103 raw as `scripts/download_sources.py` lays it out, if present.
LOCAL_CORPUS_CANDIDATES = (
    ROOT / "data" / "raw" / "wiki.valid.raw",
    ROOT / "data" / "raw" / "wikitext-103-raw" / "wiki.valid.raw",
)
#: Fallback: the *validation* split only (~650KB), small enough to fetch on a
#: pod that has not run the full corpus download. Same corpus, same licence.
CORPUS_PARQUET_URL = (
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
    "wikitext-103-raw-v1/validation-00000-of-00001.parquet"
)

#: The candidate-list width the scored harness shows the model.
MAX_CANDIDATES = 8
DEFAULT_SEED = 1337

_SENTENCE_END = {".", "!", "?"}


@dataclass(frozen=True)
class DevExample:
    """One development example, shaped like a scored example.

    ``context_before + typo + context_after`` reconstructs ``sentence``
    exactly, which is the invariant the prompt builders rely on.
    """

    example_id: str
    sentence: str
    context_before: str
    typo: str
    context_after: str
    gold: str
    candidates: tuple[str, ...]
    gold_index: int | None
    source: str  # "authentic" | "synthetic"
    corruption_type: str
    edit_distance: int

    def to_dict(self) -> dict:
        out = asdict(self)
        out["candidates"] = list(self.candidates)
        return out

    @classmethod
    def from_dict(cls, payload: dict) -> "DevExample":
        payload = dict(payload)
        payload["candidates"] = tuple(payload["candidates"])
        return cls(**payload)


@dataclass
class DevSet:
    examples: list[DevExample]
    stats: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.examples)

    def split(self, train_fraction: float = 0.5) -> tuple[list[DevExample], list[DevExample]]:
        """Deterministic train/validation split (the list is already shuffled)."""
        n_train = int(round(len(self.examples) * train_fraction))
        return self.examples[:n_train], self.examples[n_train:]


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


def load_misspelling_pairs(path: Path = DEFAULT_MISSPELLINGS) -> list[tuple[str, str]]:
    """Parse the vendored ``misspelling->correction`` list.

    Lines look like ``recieve->receive`` and occasionally offer several
    corrections (``abandonned->abandoned, abandon``); the first is taken, which
    is the same convention `scripts/calibrate_typo_model.py` uses. Pairs are
    returned sorted, so downstream sampling does not inherit file order.
    """
    pairs: set[tuple[str, str]] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "->" not in line or line.startswith("="):
            continue
        bad, _, good = line.partition("->")
        bad = bad.strip()
        good = good.split(",")[0].strip()
        if not (bad.isalpha() and good.isalpha()):
            continue
        if not is_eligible_word(bad) or not is_eligible_word(good):
            continue
        if bad.lower() == good.lower():
            continue
        pairs.add((nfc(bad), nfc(good)))
    return sorted(pairs)


def _corpus_text(corpus_file: Path | None, cache_dir: Path) -> str:
    if corpus_file is not None:
        return Path(corpus_file).read_text(encoding="utf-8", errors="replace")
    for candidate in LOCAL_CORPUS_CANDIDATES:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8", errors="replace")
    cached = Path(cache_dir) / "wiki.valid.raw"
    if not cached.is_file():
        cached.parent.mkdir(parents=True, exist_ok=True)
        _download_corpus(cached)
    return cached.read_text(encoding="utf-8", errors="replace")


def _download_corpus(dest: Path) -> None:
    """Fetch the WikiText-103 validation split and write it as raw text."""
    import io

    import pyarrow.parquet as pq
    import requests

    response = requests.get(CORPUS_PARQUET_URL, timeout=300)
    response.raise_for_status()
    table = pq.read_table(io.BytesIO(response.content))
    lines = table.column("text").to_pylist()
    dest.write_text("".join(lines), encoding="utf-8")


def split_sentences(paragraph: str) -> list[str]:
    """Split a pre-tokenised paragraph into sentences on standalone end marks.

    The corpus is Moses-tokenised, so sentence-final punctuation is already its
    own whitespace token and no abbreviation heuristics are needed. Joining the
    tokens back with single spaces preserves that convention exactly.
    """
    tokens = paragraph.split()
    sentences: list[str] = []
    current: list[str] = []
    for token in tokens:
        current.append(token)
        if token in _SENTENCE_END:
            sentences.append(" ".join(current))
            current = []
    if current:
        sentences.append(" ".join(current))
    return sentences


def usable_sentences(
    text: str, *, min_tokens: int = 8, max_tokens: int = 40
) -> list[str]:
    """Sentences fit to host a typo: plain, bounded length, no markup noise.

    WikiText carries section headings (``= = Description = =``) and escaped
    hyphens (``@-@``); both are dropped rather than repaired, since they would
    show up in a prompt as text no writer would ever type.
    """
    out: list[str] = []
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph or paragraph.startswith("=") or "@" in paragraph:
            continue
        for sentence in split_sentences(paragraph):
            tokens = sentence.split()
            if not (min_tokens <= len(tokens) <= max_tokens):
                continue
            if any(ch in sentence for ch in "=@<>"):
                continue
            if not sentence.isascii():
                continue
            out.append(sentence)
    return out


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def _match_case(source: str, replacement: str) -> str:
    if source.isupper() and len(source) > 1:
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _place(sentence_tokens: Sequence[str], index: int, typo: str) -> tuple[str, str, str]:
    before = " ".join(sentence_tokens[:index])
    after = " ".join(sentence_tokens[index + 1 :])
    if before:
        before += " "
    if after:
        after = " " + after
    return before, typo, after


def _index_sentences_by_word(sentences: Sequence[str]) -> dict[str, list[tuple[int, int]]]:
    """word (lowercased) -> [(sentence index, token index)], in corpus order."""
    index: dict[str, list[tuple[int, int]]] = {}
    for s_idx, sentence in enumerate(sentences):
        for t_idx, token in enumerate(sentence.split()):
            if not is_eligible_word(token):
                continue
            index.setdefault(token.lower(), []).append((s_idx, t_idx))
    return index


def _make_example(
    engine: HunspellEngine,
    *,
    example_id: str,
    sentence_tokens: Sequence[str],
    token_index: int,
    typo: str,
    gold: str,
    source: str,
    corruption_type: str,
    max_candidates: int,
    require_known_gold: bool,
) -> DevExample | None:
    """Query Hunspell and assemble one example, or None if it is out of scope.

    Out of scope means Hunspell does not flag the typo, or flags it and offers
    nothing -- the same eligibility rule the scored harness applies, so the dev
    population is drawn from the same slice of the task.

    `require_known_gold` additionally drops examples whose *gold* word is itself
    outside the dictionary. Corrupting a rare proper noun out of an encyclopedia
    ("Eyton" -> "Eayton") produces an example no speller-backed system can ever
    get right and that no writer would ever produce; that is corpus noise, not
    task difficulty. Note this is not the same as requiring the gold to be in
    the candidate pool -- that rate stays whatever Hunspell achieves, because
    the out-of-pool examples are precisely where a free-form answer earns its
    keep.
    """
    if require_known_gold and not engine.spell(gold):
        return None
    if engine.spell(typo):
        return None
    candidates = engine.suggest(typo)[:max_candidates]
    if not candidates:
        return None
    before, typo_text, after = _place(sentence_tokens, token_index, typo)
    gold_index = engine.gold_index(candidates, gold)
    return DevExample(
        example_id=example_id,
        sentence=before + typo_text + after,
        context_before=before,
        typo=typo_text,
        context_after=after,
        gold=nfc(gold),
        candidates=tuple(candidates),
        gold_index=gold_index,
        source=source,
        corruption_type=corruption_type,
        edit_distance=edit_distance(typo.lower(), gold.lower(), cap=4),
    )


def build_dev_set(
    n: int = 300,
    *,
    seed: int = DEFAULT_SEED,
    authentic_fraction: float = 0.5,
    max_candidates: int = MAX_CANDIDATES,
    corpus_file: Path | None = None,
    misspellings_file: Path = DEFAULT_MISSPELLINGS,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    require_known_gold: bool = True,
    engine: HunspellEngine | None = None,
    sentences: Sequence[str] | None = None,
) -> DevSet:
    """Build ``n`` development examples, half authentic and half generated.

    `sentences` is an injection point for tests; production callers let the
    corpus be resolved from disk (or fetched once into ``cache_dir``).
    """
    if not 0.0 <= authentic_fraction <= 1.0:
        raise ValueError(f"authentic_fraction must be in [0, 1]; got {authentic_fraction}")
    engine = engine or default_engine()
    rng = np.random.default_rng(seed)

    if sentences is None:
        sentences = usable_sentences(_corpus_text(corpus_file, Path(cache_dir)))
    sentences = list(sentences)
    if not sentences:
        raise ValueError("no usable sentences in the corpus")
    word_index = _index_sentences_by_word(sentences)

    n_authentic = int(round(n * authentic_fraction))
    examples: list[DevExample] = []
    seen: set[tuple[str, str]] = set()

    # --- authentic misspellings dropped into a sentence containing the gold ---
    pairs = [p for p in load_misspelling_pairs(misspellings_file) if p[1].lower() in word_index]
    order = rng.permutation(len(pairs))
    for pos in order:
        if len(examples) >= n_authentic:
            break
        typo, gold = pairs[int(pos)]
        sites = word_index[gold.lower()]
        s_idx, t_idx = sites[int(rng.integers(len(sites)))]
        tokens = sentences[s_idx].split()
        surface_typo = _match_case(tokens[t_idx], typo)
        key = (surface_typo.lower(), gold.lower())
        if key in seen:
            continue
        example = _make_example(
            engine,
            example_id=f"auth-{len(examples):04d}",
            sentence_tokens=tokens,
            token_index=t_idx,
            typo=surface_typo,
            gold=tokens[t_idx],
            source="authentic",
            corruption_type="wikipedia_list",
            max_candidates=max_candidates,
            require_known_gold=require_known_gold,
        )
        if example is None:
            continue
        seen.add(key)
        examples.append(example)

    # --- generator typos over the same corpus ---
    sentence_order = rng.permutation(len(sentences))
    cursor = 0
    while len(examples) < n and cursor < len(sentence_order):
        sentence = sentences[int(sentence_order[cursor])]
        cursor += 1
        tokens = sentence.split()
        eligible = [i for i, tok in enumerate(tokens) if is_eligible_word(tok)]
        if not eligible:
            continue
        t_idx = int(eligible[int(rng.integers(len(eligible)))])
        gold = tokens[t_idx]
        typo, corruption_type = corrupt_word(gold, rng)
        if typo.lower() == gold.lower():
            continue
        key = (typo.lower(), gold.lower())
        if key in seen:
            continue
        example = _make_example(
            engine,
            example_id=f"synth-{len(examples):04d}",
            sentence_tokens=tokens,
            token_index=t_idx,
            typo=typo,
            gold=gold,
            source="synthetic",
            corruption_type=corruption_type,
            max_candidates=max_candidates,
            require_known_gold=require_known_gold,
        )
        if example is None:
            continue
        seen.add(key)
        examples.append(example)

    # Interleave sources so a train/validation prefix split cannot land all of
    # one source on one side.
    shuffle = rng.permutation(len(examples))
    examples = [examples[int(i)] for i in shuffle]
    return DevSet(examples=examples, stats=composition_stats(examples))


def composition_stats(examples: Sequence[DevExample]) -> dict:
    """Population statistics of a dev set, for comparison with the scored one.

    Reported rather than enforced: ``gold_in_pool`` and ``gold_at_0`` are
    whatever Hunspell does on these typos. The comparison to make by hand is
    against the statistics recorded in ``reports/experiments/02-byte-reranker-87m/RUN_NOTES.md`` section 4 -- a dev set
    whose gold-at-0 rate or edit-distance mixture is far from those is
    optimizing a prompt for a different population than it will be scored on.
    """
    n = len(examples)
    if n == 0:
        return {"n": 0}
    ed = {1: 0, 2: 0, 3: 0}
    for ex in examples:
        bucket = min(max(ex.edit_distance, 1), 3)
        ed[bucket] += 1
    in_pool = [ex for ex in examples if ex.gold_index is not None]
    at_zero = [ex for ex in in_pool if ex.gold_index == 0]
    n_cands = [len(ex.candidates) for ex in examples]
    ed_in_pool = {1: 0, 2: 0, 3: 0}
    for ex in in_pool:
        ed_in_pool[min(max(ex.edit_distance, 1), 3)] += 1
    return {
        "n": n,
        "sources": {
            "authentic": sum(1 for ex in examples if ex.source == "authentic"),
            "synthetic": sum(1 for ex in examples if ex.source == "synthetic"),
        },
        "edit_distance": {
            "ed1_pct": 100.0 * ed[1] / n,
            "ed2_pct": 100.0 * ed[2] / n,
            "ed3plus_pct": 100.0 * ed[3] / n,
        },
        "edit_distance_gold_in_pool": (
            {
                "ed1_pct": 100.0 * ed_in_pool[1] / len(in_pool),
                "ed2_pct": 100.0 * ed_in_pool[2] / len(in_pool),
                "ed3plus_pct": 100.0 * ed_in_pool[3] / len(in_pool),
            }
            if in_pool
            else {}
        ),
        "gold_in_pool_pct": 100.0 * len(in_pool) / n,
        "gold_at_0_pct_of_in_pool": (100.0 * len(at_zero) / len(in_pool)) if in_pool else 0.0,
        "hunspell_top1_pct": 100.0 * len(at_zero) / n,
        "mean_candidates": sum(n_cands) / n,
        "mean_sentence_tokens": sum(len(ex.sentence.split()) for ex in examples) / n,
    }


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


def cache_path(cache_dir: Path, n: int, seed: int, authentic_fraction: float) -> Path:
    tag = f"n{n}_seed{seed}_auth{int(round(authentic_fraction * 100)):03d}"
    return Path(cache_dir) / f"dev_set_{tag}.json"


def save_dev_set(dev: DevSet, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stats": dev.stats,
        "examples": [ex.to_dict() for ex in dev.examples],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_dev_set(path: Path) -> DevSet:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    examples = [DevExample.from_dict(item) for item in payload["examples"]]
    return DevSet(examples=examples, stats=payload.get("stats") or composition_stats(examples))


def load_or_build(
    n: int = 300,
    *,
    seed: int = DEFAULT_SEED,
    authentic_fraction: float = 0.5,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    rebuild: bool = False,
    **kwargs,
) -> tuple[DevSet, Path]:
    """Return a cached dev set, building (and caching) it if necessary."""
    path = cache_path(Path(cache_dir), n, seed, authentic_fraction)
    if path.is_file() and not rebuild:
        return load_dev_set(path), path
    dev = build_dev_set(
        n, seed=seed, authentic_fraction=authentic_fraction, cache_dir=cache_dir, **kwargs
    )
    save_dev_set(dev, path)
    return dev, path


def format_stats(stats: dict) -> str:
    """Human-readable one-block summary of :func:`composition_stats`."""
    if not stats.get("n"):
        return "dev set: empty"
    # Tolerant of a partial dict: a cache written by an older build, or a
    # caller passing `{"n": ...}` only, should still print something useful
    # rather than crash a run that is otherwise fine.
    ed = stats.get("edit_distance") or {"ed1_pct": 0.0, "ed2_pct": 0.0, "ed3plus_pct": 0.0}
    stats = {
        "sources": {"authentic": 0, "synthetic": 0},
        "gold_in_pool_pct": 0.0,
        "gold_at_0_pct_of_in_pool": 0.0,
        "hunspell_top1_pct": 0.0,
        "mean_candidates": 0.0,
        "mean_sentence_tokens": 0.0,
        **stats,
    }
    lines = [
        f"dev set: n={stats['n']} "
        f"(authentic {stats['sources']['authentic']}, synthetic {stats['sources']['synthetic']})",
        f"  edit distance      ED1 {ed['ed1_pct']:.1f}%  ED2 {ed['ed2_pct']:.1f}%  "
        f"ED3+ {ed['ed3plus_pct']:.1f}%",
        f"  gold in pool       {stats['gold_in_pool_pct']:.1f}%",
        f"  gold at rank 0     {stats['gold_at_0_pct_of_in_pool']:.1f}% of in-pool "
        f"({stats['hunspell_top1_pct']:.1f}% overall = Hunspell top-1)",
        f"  mean candidates    {stats['mean_candidates']:.2f}",
        f"  mean sentence len  {stats['mean_sentence_tokens']:.1f} tokens",
    ]
    if stats.get("edit_distance_gold_in_pool"):
        edp = stats["edit_distance_gold_in_pool"]
        lines.insert(
            2,
            f"  ED | gold in pool  ED1 {edp['ed1_pct']:.1f}%  ED2 {edp['ed2_pct']:.1f}%  "
            f"ED3+ {edp['ed3plus_pct']:.1f}%",
        )
    return "\n".join(lines)


def iter_examples(dev: DevSet) -> Iterable[DevExample]:
    return iter(dev.examples)
