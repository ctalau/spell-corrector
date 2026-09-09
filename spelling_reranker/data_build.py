"""Construct synthetic Hunspell-reranking examples from WikiText-103.

Speed
-----
The first pipeline called Hunspell once per *example*, so speller cost scaled
with dataset size (~18 min for 240k examples). Hunspell's `suggest` is the
dominant cost and depends only on the typo string, so this build instead:

  1. counts eligible words over the corpus (pass 1),
  2. builds a `word -> [(typo, gold_index, pool)]` table, calling Hunspell once
     per *unique typo* in a process pool,
  3. streams the corpus again and instantiates examples by dropping a
     precomputed typo into each sentence (pass 2, pure string work).

Speller calls therefore scale with the vocabulary, not the example count, which
is what makes a multi-million example build practical.

Realism
-------
Two mismatches with the benchmark are corrected here:

* **Noisy context.** A spelling corrector reads text the writer has not yet
  corrected, so the words *around* an error are themselves often misspelled.
  Training context used to be clean WikiText, which teaches the model to trust
  context it will not get at inference time. `context_noise_prob` corrupts
  neighbouring words so the model learns to weigh unreliable context.
* **Gold-index skew.** Hunspell already ranks the gold repair first for ~88% of
  synthetic typos. Those examples teach the reranker only to agree with
  Hunspell; all of the value is in the rest, so they are capped at
  `gold0_fraction`. This is ordinary class balancing on a skewed label. The cap
  was fixed a priori and never tuned against a held-out benchmark.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import pandas as pd
from tqdm import tqdm

from spelling_reranker.byte_encoding import nfc
from spelling_reranker.candidates import build_pool, gold_index as pool_gold_index
from spelling_reranker.hunspell import HunspellEngine, default_engine
from spelling_reranker.serialization import (
    DEFAULT_MAX_SEQ_LEN,
    MAX_CANDIDATE_BYTES,
    N_CANDIDATES,
    PathologicalExampleError,
    serialize_example,
)
from spelling_reranker.typo_gen import (
    corrupt_word,
    eligible_token_indices,
    is_eligible_word,
    tokenize_sentence,
)

ARTICLE_START_RE = re.compile(r"^ = [^=].* = $")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
POOL_SEP = "\x00"


# ---------------------------------------------------------------------------
# Corpus iteration
# ---------------------------------------------------------------------------


def iter_wikitext_articles(path: Path) -> Iterator[tuple[str, str]]:
    title = "preamble"
    buf: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if ARTICLE_START_RE.match(line):
                if buf:
                    yield title, "\n".join(buf)
                title = line.strip().strip("=").strip()
                buf = []
                continue
            buf.append(line)
        if buf:
            yield title, "\n".join(buf)


def paragraphs_from_article(text: str) -> list[str]:
    out: list[str] = []
    for chunk in re.split(r"\n\s*\n", text):
        joined = re.sub(r"\s+", " ", chunk).strip()
        if joined:
            out.append(joined)
    return out


def is_section_heading(sentence: str) -> bool:
    text = sentence.strip()
    return text.startswith("=") and text.endswith("=")


def sentences_from_paragraph(paragraph: str) -> list[str]:
    parts = SENT_SPLIT_RE.split(paragraph)
    return [p.strip() for p in parts if p.strip() and not is_section_heading(p.strip())]


def iter_sentences(path: Path, *, max_articles: int | None = None) -> Iterator[tuple[str, str]]:
    """Yield (document_id, sentence) over a WikiText raw file."""
    for art_i, (title, body) in enumerate(iter_wikitext_articles(path)):
        if max_articles is not None and art_i >= max_articles:
            break
        document_id = f"wikitext103:{title}"
        for para in paragraphs_from_article(body):
            for sentence in sentences_from_paragraph(para):
                yield document_id, sentence


# ---------------------------------------------------------------------------
# Pass 1: vocabulary
# ---------------------------------------------------------------------------


def count_vocabulary(path: Path, *, max_articles: int | None = None) -> Counter[str]:
    counts: Counter[str] = Counter()
    for _, sentence in iter_sentences(path, max_articles=max_articles):
        for token in sentence.split():
            if is_eligible_word(token):
                counts[token] += 1
    return counts


def select_vocabulary(counts: Counter[str], *, vocab_size: int, min_count: int = 2) -> list[tuple[str, int]]:
    """Most frequent eligible words, frequency-ordered.

    Frequency ordering matters: the benchmark is everyday prose, so the model's
    training effort should follow the words people actually misspell rather than
    WikiText's long tail of proper nouns.
    """
    items = [(w, c) for w, c in counts.items() if c >= min_count]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return items[:vocab_size]


def typos_per_word(rank: int, count: int, *, min_typos: int, max_typos: int) -> int:
    """Allocate more distinct typos to more frequent words."""
    n = int(round(min_typos + (max_typos - min_typos) * math.sqrt(1.0 / (1.0 + rank / 2000.0))))
    return max(min_typos, min(max_typos, n))


# ---------------------------------------------------------------------------
# Pass 2: typo table (the only phase that calls Hunspell)
# ---------------------------------------------------------------------------


def available_cpus() -> int:
    """CPUs this process may actually use.

    `os.cpu_count()` reports the host's cores, which inside a container is a
    lie: a Runpod pod advertising 28 vCPU sits on a 112-core host, and sizing a
    process pool from cpu_count oversubscribes it 4x. Prefer the affinity mask,
    then the cgroup quota, and take the tightest bound.
    """
    limits: list[int] = []
    try:
        limits.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass

    # cgroup v2, then v1.
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            limits.append(max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        try:
            quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if quota > 0 and period > 0:
                limits.append(max(1, quota // period))
        except (OSError, ValueError):
            pass

    if not limits:
        limits.append(os.cpu_count() or 1)
    return max(1, min(limits))


_WORKER_ENGINE: HunspellEngine | None = None


def _init_worker() -> None:
    global _WORKER_ENGINE
    _WORKER_ENGINE = default_engine()


def _build_typos_for_shard(payload: tuple) -> tuple[list[tuple[str, str, str, int, str]], dict]:
    """Return (entries, stats) for one shard of the vocabulary.

    entry = (word, typo, kind, gold_index, POOL_SEP-joined pool)
    """
    shard, seed, min_typos, max_typos = payload
    engine = _WORKER_ENGINE or default_engine()
    rng = np.random.default_rng(seed)
    entries: list[tuple[str, str, str, int, str]] = []
    stats: Counter[str] = Counter()
    for rank, word, count in shard:
        if not engine.spell(word):
            stats["source_not_in_dictionary"] += 1
            continue
        wanted = typos_per_word(rank, count, min_typos=min_typos, max_typos=max_typos)
        seen: set[str] = set()
        for _ in range(wanted * 3):
            if len(seen) >= wanted:
                break
            typo, kind = corrupt_word(word, rng)
            if typo == word or typo in seen:
                continue
            seen.add(typo)
            if engine.spell(typo):
                stats["typo_still_a_word"] += 1
                continue
            suggestions = engine.suggest(typo)
            if not suggestions:
                stats["no_suggestions"] += 1
                continue
            pool = build_pool(suggestions, limit=N_CANDIDATES, max_bytes=MAX_CANDIDATE_BYTES)
            gi = pool_gold_index(pool, word)
            if gi is None:
                stats["gold_not_in_pool"] += 1
                continue
            stats["kept"] += 1
            stats[f"kind:{kind.split('+')[0]}"] += 1
            stats[f"gold_index:{gi}"] += 1
            entries.append((word, typo, kind, gi, POOL_SEP.join(pool)))
    return entries, dict(stats)


def build_typo_table(
    vocabulary: list[tuple[str, int]],
    *,
    seed: int,
    workers: int | None = None,
    min_typos: int = 4,
    max_typos: int = 48,
    show_progress: bool = True,
) -> tuple[dict[str, list[tuple[str, str, int, str]]], Counter[str]]:
    """Map each word to its usable (typo, kind, gold_index, pool) entries."""
    n_workers = workers if workers is not None else available_cpus()
    ranked = [(rank, word, count) for rank, (word, count) in enumerate(vocabulary)]
    shard_size = max(64, len(ranked) // (n_workers * 8) or 1)
    shards = [ranked[i : i + shard_size] for i in range(0, len(ranked), shard_size)]
    payloads = [(shard, seed + 7919 * i, min_typos, max_typos) for i, shard in enumerate(shards)]

    table: dict[str, list[tuple[str, str, int, str]]] = {}
    stats: Counter[str] = Counter()
    progress = tqdm(total=len(payloads), desc="typo table", disable=not show_progress)

    def _absorb(result: tuple[list[tuple[str, str, str, int, str]], dict]) -> None:
        entries, part = result
        for word, typo, kind, gi, pool in entries:
            table.setdefault(word, []).append((typo, kind, gi, pool))
        for key, value in part.items():
            stats[key] += int(value)
        progress.update(1)

    if n_workers <= 1:
        global _WORKER_ENGINE
        _WORKER_ENGINE = default_engine()
        try:
            for payload in payloads:
                _absorb(_build_typos_for_shard(payload))
        finally:
            _WORKER_ENGINE = None
    else:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as pool_exec:
            for result in pool_exec.map(_build_typos_for_shard, payloads, chunksize=1):
                _absorb(result)
    progress.close()
    return table, stats


# ---------------------------------------------------------------------------
# Pass 3: example instantiation (no speller calls)
# ---------------------------------------------------------------------------


@dataclass
class BuildStats:
    discarded: Counter[str] = field(default_factory=Counter)
    corruption_type: Counter[str] = field(default_factory=Counter)
    gold_index: Counter[int] = field(default_factory=Counter)
    candidate_count: Counter[int] = field(default_factory=Counter)
    edit_distance: Counter[int] = field(default_factory=Counter)
    typo_lengths: list[int] = field(default_factory=list)
    serialized_lengths: list[int] = field(default_factory=list)
    source_mixture: Counter[str] = field(default_factory=Counter)
    context_noised: int = 0
    attempts: int = 0
    kept: int = 0

    def as_dict(self) -> dict:
        lengths = sorted(self.serialized_lengths)
        typo_lens = sorted(self.typo_lengths)

        def _pct(xs: list[int], q: float) -> int | None:
            if not xs:
                return None
            idx = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
            return int(xs[idx])

        return {
            "attempts": self.attempts,
            "kept": self.kept,
            "context_noised": self.context_noised,
            "discarded": dict(self.discarded),
            "corruption_type": dict(self.corruption_type),
            "gold_index": {str(k): int(v) for k, v in sorted(self.gold_index.items())},
            "candidate_count": {str(k): int(v) for k, v in sorted(self.candidate_count.items())},
            "edit_distance": {str(k): int(v) for k, v in sorted(self.edit_distance.items())},
            "source_mixture": dict(self.source_mixture),
            "typo_length": {
                "mean": float(np.mean(typo_lens)) if typo_lens else None,
                "p50": _pct(typo_lens, 0.50),
                "p95": _pct(typo_lens, 0.95),
                "max": typo_lens[-1] if typo_lens else None,
            },
            "serialized_byte_length": {
                "mean": float(np.mean(lengths)) if lengths else None,
                "p50": _pct(lengths, 0.50),
                "p95": _pct(lengths, 0.95),
                "max": lengths[-1] if lengths else None,
            },
        }


def edit_distance(a: str, b: str, cap: int = 4) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return min(prev[-1], cap)


def sha256_text(text: str) -> str:
    return hashlib.sha256(nfc(text).encode("utf-8")).hexdigest()


def _noise_context(
    tokens: list[str],
    target_i: int,
    rng: np.random.Generator,
    *,
    max_words: int,
) -> bool:
    """Corrupt up to `max_words` other eligible tokens in place. True if changed."""
    candidates = [
        i
        for i, tok in enumerate(tokens)
        if i != target_i and is_eligible_word(tok)
    ]
    if not candidates:
        return False
    rng.shuffle(candidates)
    changed = False
    for i in candidates[: max_words]:
        typo, _ = corrupt_word(tokens[i], rng)
        if typo != tokens[i]:
            tokens[i] = typo
            changed = True
    return changed


def iter_examples(
    sentences: Iterable[tuple[str, str]] | Callable[[], Iterable[tuple[str, str]]],
    typo_table: dict[str, list[tuple[str, str, int, str]]],
    *,
    target: int,
    seed: int,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    source: str = "wikitext103-synthetic",
    context_noise_prob: float = 0.25,
    max_noise_words: int = 2,
    gold0_fraction: float = 0.65,
    max_uses_per_typo: int = 8,
    max_passes: int = 3,
    show_progress: bool = True,
    stats: BuildStats | None = None,
) -> Iterator[dict]:
    """Yield up to `target` examples by dropping table typos into sentences.

    `sentences` may be an iterable or a callable returning a fresh one. The
    corpus holds a finite number of usable sentences, so when `target` exceeds
    that, a callable lets the stream be walked again (up to `max_passes`); each
    pass draws a different target word and typo, so a revisited sentence yields
    a different example rather than a duplicate.
    """
    stats = stats if stats is not None else BuildStats()
    rng = np.random.default_rng(seed)
    rows_emitted = 0
    uses: Counter[tuple[str, str]] = Counter()
    emitted: set[tuple[str, str, str]] = set()
    gold0_budget = int(target * gold0_fraction)
    gold0_used = 0
    progress = tqdm(total=target, desc="examples", disable=not show_progress)

    def _streams():
        if callable(sentences):
            for _ in range(max(1, max_passes)):
                yield sentences()
        else:
            yield sentences

    for stream in _streams():
        if rows_emitted >= target:
            break
        for document_id, sentence in stream:
            if rows_emitted >= target:
                break
            tokens = sentence.split()
            eligible = [i for i, tok in enumerate(tokens) if tok in typo_table]
            if not eligible:
                continue
            stats.attempts += 1
            target_i = int(eligible[int(rng.integers(0, len(eligible)))])
            word = tokens[target_i]
            entries = typo_table[word]

            entry = None
            for _ in range(4):
                cand = entries[int(rng.integers(0, len(entries)))]
                if uses[(word, cand[0])] >= max_uses_per_typo:
                    continue
                if cand[2] == 0 and gold0_used >= gold0_budget:
                    continue
                entry = cand
                break
            if entry is None:
                stats.discarded["no_usable_typo"] += 1
                continue

            typo, kind, gi, pool_joined = entry
            key = (document_id, sentence, typo)
            if key in emitted:
                stats.discarded["duplicate_example"] += 1
                continue

            working = list(tokens)
            noised = False
            if rng.random() < context_noise_prob:
                noised = _noise_context(
                    working, target_i, rng, max_words=int(rng.integers(1, max_noise_words + 1))
                )
            working[target_i] = typo
            noisy_sentence = " ".join(working)
            start = len(" ".join(working[:target_i]))
            if target_i > 0:
                start += 1
            end = start + len(typo)
            context_before = noisy_sentence[:start]
            context_after = noisy_sentence[end:]

            pool = pool_joined.split(POOL_SEP)
            padded: list[str | None] = list(pool) + [None] * (N_CANDIDATES - len(pool))
            try:
                serialized = serialize_example(
                    context_before, typo, context_after, padded,
                    max_seq_len=max_seq_len, gold_index=gi,
                )
            except PathologicalExampleError:
                stats.discarded["pathological_serialization"] += 1
                continue

            emitted.add(key)
            uses[(word, typo)] += 1
            if gi == 0:
                gold0_used += 1

            stats.kept += 1
            stats.corruption_type[kind] += 1
            stats.gold_index[gi] += 1
            stats.candidate_count[len(pool)] += 1
            stats.edit_distance[edit_distance(typo.lower(), word.lower())] += 1
            stats.typo_lengths.append(len(typo.encode("utf-8")))
            stats.serialized_lengths.append(serialized.seq_len)
            stats.source_mixture[source] += 1
            if noised:
                stats.context_noised += 1

            row = {
                "example_id": f"{sha256_text(document_id + sentence + typo)[:16]}:{gi}",
                "source": source,
                "context_before": context_before,
                "typo": typo,
                "context_after": context_after,
                "gold": nfc(word),
                "gold_index": np.int8(gi),
                "corruption_type": kind,
                "source_document_id": document_id,
                "original_sentence_hash": sha256_text(sentence),
            }
            for i in range(N_CANDIDATES):
                row[f"cand_{i}"] = padded[i]
            rows_emitted += 1
            progress.update(1)
            yield row

    progress.close()


def generate_examples(*args, **kwargs) -> tuple[list[dict], BuildStats]:
    """List-returning wrapper around `iter_examples`.

    Convenient for tests and small builds. Large builds should consume
    `iter_examples` directly and stream to parquet -- holding several million
    row dicts in memory costs multiple gigabytes.
    """
    stats = kwargs.pop("stats", None) or BuildStats()
    rows = list(iter_examples(*args, stats=stats, **kwargs))
    return rows, stats


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    columns = [
        "example_id",
        "source",
        "context_before",
        "typo",
        "context_after",
        "gold",
        *[f"cand_{i}" for i in range(N_CANDIDATES)],
        "gold_index",
        "corruption_type",
        "source_document_id",
        "original_sentence_hash",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    frame["gold_index"] = frame["gold_index"].astype("int8")
    return frame


def write_split(
    rows: Iterator[dict],
    path: Path,
    *,
    chunk_size: int = 200_000,
) -> int:
    """Stream example dicts to a parquet file in row-group chunks.

    Buffering several million row dicts before writing costs multiple GiB; this
    keeps peak memory at roughly one chunk.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    writer: "pq.ParquetWriter | None" = None
    buffer: list[dict] = []
    total = 0

    def _flush() -> None:
        nonlocal writer, buffer, total
        if not buffer:
            return
        table = pa.Table.from_pandas(rows_to_frame(buffer), preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(path), table.schema)
        writer.write_table(table)
        total += len(buffer)
        buffer = []

    try:
        for row in rows:
            buffer.append(row)
            if len(buffer) >= chunk_size:
                _flush()
        _flush()
        if writer is None:
            # Nothing generated: still emit a valid empty file with the schema.
            table = pa.Table.from_pandas(rows_to_frame([]), preserve_index=False)
            writer = pq.ParquetWriter(str(path), table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return total


def write_manifest(
    n_train: int,
    n_valid: int,
    *,
    out_dir: Path,
    seed: int,
    source_meta: dict,
    train_stats: BuildStats,
    valid_stats: BuildStats,
    elapsed_sec: float,
    extra: dict | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.parquet"
    valid_path = out_dir / "validation.parquet"

    def _file_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    stats = {
        "seed": seed,
        "elapsed_sec": elapsed_sec,
        "n_candidate_slots": N_CANDIDATES,
        "train": {"n": n_train, **train_stats.as_dict()},
        "validation": {"n": n_valid, **valid_stats.as_dict()},
        "source": source_meta,
        "created_unix": int(time.time()),
        **(extra or {}),
    }
    manifest = {
        "seed": seed,
        "files": {
            "train": {
                "path": "data/processed/train.parquet",
                "n": n_train,
                "sha256": _file_hash(train_path),
                "bytes": train_path.stat().st_size,
            },
            "validation": {
                "path": "data/processed/validation.parquet",
                "n": n_valid,
                "sha256": _file_hash(valid_path),
                "bytes": valid_path.stat().st_size,
            },
        },
        "source": source_meta,
        "notes": [
            "Synthetic typos derived from WikiText-103 raw only.",
            "Candidate pools come from Hunspell suggestions only.",
            "Held-out benchmark data is never used here.",
        ],
    }
    (out_dir / "data_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
