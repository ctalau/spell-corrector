"""Construct synthetic Hunspell-reranking examples from WikiText-103."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from tqdm import tqdm

from spelling_reranker.byte_encoding import nfc
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
MAX_CANDIDATE_UTF8 = MAX_CANDIDATE_BYTES


@dataclass
class BuildStats:
    discarded: Counter[str] = field(default_factory=Counter)
    corruption_type: Counter[str] = field(default_factory=Counter)
    gold_index: Counter[int] = field(default_factory=Counter)
    candidate_count: Counter[int] = field(default_factory=Counter)
    typo_lengths: list[int] = field(default_factory=list)
    serialized_lengths: list[int] = field(default_factory=list)
    source_mixture: Counter[str] = field(default_factory=Counter)
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
            "discarded": dict(self.discarded),
            "corruption_type": dict(self.corruption_type),
            "gold_index": {str(k): int(v) for k, v in sorted(self.gold_index.items())},
            "candidate_count": {str(k): int(v) for k, v in sorted(self.candidate_count.items())},
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(nfc(text).encode("utf-8")).hexdigest()


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
    chunks = re.split(r"\n\s*\n", text)
    out: list[str] = []
    for chunk in chunks:
        joined = re.sub(r"\s+", " ", chunk).strip()
        if joined:
            out.append(joined)
    return out


def sentences_from_paragraph(paragraph: str) -> list[str]:
    parts = SENT_SPLIT_RE.split(paragraph)
    return [p.strip() for p in parts if p.strip()]


def _candidate_too_long(candidates: list[str]) -> bool:
    return any(len(c.encode("utf-8")) > MAX_CANDIDATE_UTF8 for c in candidates)


def try_make_example(
    sentence: str,
    word_span: tuple[int, int, str],
    *,
    engine: HunspellEngine,
    rng: np.random.Generator,
    source: str,
    document_id: str,
    example_prefix: str,
    max_seq_len: int,
    stats: BuildStats,
) -> dict | None:
    stats.attempts += 1
    start, end, word = word_span
    if not is_eligible_word(word):
        stats.discarded["ineligible_word"] += 1
        return None
    if not engine.spell(word):
        stats.discarded["source_not_in_dictionary"] += 1
        return None

    typo, kind = corrupt_word(word, rng)
    if typo == word:
        stats.discarded["corruption_unchanged"] += 1
        return None
    if engine.spell(typo):
        stats.discarded["typo_still_correct"] += 1
        return None

    suggestions = engine.candidates(typo)
    if not suggestions:
        stats.discarded["no_suggestions"] += 1
        return None
    if _candidate_too_long(suggestions):
        stats.discarded["candidate_too_long"] += 1
        return None

    gold = nfc(word)
    gold_index = engine.gold_index(suggestions, gold)
    if gold_index is None:
        stats.discarded["gold_not_in_top10"] += 1
        return None

    context_before = sentence[:start]
    context_after = sentence[end:]
    padded: list[str | None] = list(suggestions) + [None] * (N_CANDIDATES - len(suggestions))
    try:
        serialized = serialize_example(
            context_before,
            typo,
            context_after,
            padded,
            max_seq_len=max_seq_len,
            gold_index=gold_index,
        )
    except PathologicalExampleError:
        stats.discarded["pathological_serialization"] += 1
        return None

    stats.kept += 1
    stats.corruption_type[kind] += 1
    stats.gold_index[gold_index] += 1
    stats.candidate_count[len(suggestions)] += 1
    stats.typo_lengths.append(len(typo.encode("utf-8")))
    stats.serialized_lengths.append(serialized.seq_len)
    stats.source_mixture[source] += 1

    row = {
        "example_id": f"{example_prefix}:{kind}:{gold_index}:{sha256_text(typo + gold)[:12]}",
        "source": source,
        "context_before": context_before,
        "typo": typo,
        "context_after": context_after,
        "gold": gold,
        "gold_index": np.int8(gold_index),
        "corruption_type": kind,
        "source_document_id": document_id,
        "original_sentence_hash": sha256_text(sentence),
    }
    for i in range(N_CANDIDATES):
        row[f"cand_{i}"] = padded[i]
    return row


def collect_sentence_jobs(
    articles: Iterator[tuple[str, str]],
    *,
    split_name: str,
    source: str,
    max_articles: int | None = None,
) -> list[dict]:
    jobs: list[dict] = []
    for art_i, (title, body) in enumerate(articles):
        if max_articles is not None and art_i >= max_articles:
            break
        document_id = f"wikitext103:{split_name}:{title}"
        for para in paragraphs_from_article(body):
            for sent_i, sentence in enumerate(sentences_from_paragraph(para)):
                tokens = tokenize_sentence(sentence)
                idxs = eligible_token_indices(tokens)
                if not idxs:
                    continue
                jobs.append(
                    {
                        "sentence": sentence,
                        "tokens": tokens,
                        "eligible": idxs,
                        "document_id": document_id,
                        "source": source,
                        "sent_i": sent_i,
                    }
                )
    return jobs


_WORKER_ENGINE: HunspellEngine | None = None


def _init_worker() -> None:
    global _WORKER_ENGINE
    _WORKER_ENGINE = default_engine()


def _process_job(payload: tuple) -> tuple[int, list[dict], dict]:
    """Deterministic per-job worker. Returns (order_index, rows, stats_dict)."""
    order_index, job, seed, max_attempts, max_seq_len = payload
    engine = _WORKER_ENGINE or default_engine()
    rng = np.random.default_rng(int(seed) + int(order_index) * 10007)
    stats = BuildStats()
    eligible = list(job["eligible"])
    rng.shuffle(eligible)
    rows: list[dict] = []
    for tok_i in eligible[:max_attempts]:
        span = job["tokens"][tok_i]
        row = try_make_example(
            job["sentence"],
            span,
            engine=engine,
            rng=rng,
            source=job["source"],
            document_id=job["document_id"],
            example_prefix=f"{job['document_id']}:{job['sent_i']}:{tok_i}",
            max_seq_len=max_seq_len,
            stats=stats,
        )
        if row is not None:
            rows.append(row)
    return order_index, rows, stats.as_dict()


def _merge_stat_dicts(parts: list[dict]) -> BuildStats:
    merged = BuildStats()
    for part in parts:
        merged.attempts += int(part.get("attempts", 0))
        merged.kept += int(part.get("kept", 0))
        for key, value in (part.get("discarded") or {}).items():
            merged.discarded[key] += int(value)
        for key, value in (part.get("corruption_type") or {}).items():
            merged.corruption_type[key] += int(value)
        for key, value in (part.get("gold_index") or {}).items():
            merged.gold_index[int(key)] += int(value)
        for key, value in (part.get("candidate_count") or {}).items():
            merged.candidate_count[int(key)] += int(value)
        for key, value in (part.get("source_mixture") or {}).items():
            merged.source_mixture[key] += int(value)
        typo = part.get("typo_length") or {}
        if typo.get("max") is not None:
            merged.typo_lengths.append(int(typo["max"]))
        ser = part.get("serialized_byte_length") or {}
        if ser.get("max") is not None:
            merged.serialized_lengths.append(int(ser["max"]))
    return merged


def generate_examples(
    jobs: list[dict],
    *,
    target: int,
    seed: int,
    engine: HunspellEngine | None = None,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    max_attempts_per_sentence: int = 6,
    show_progress: bool = True,
    workers: int | None = None,
) -> tuple[list[dict], BuildStats]:
    """Generate up to `target` examples.

    Job order is a permutation of `seed`. Each job uses an isolated RNG
    (`seed + order_index * 10007`) so multiprocessing stays deterministic.
    """
    if not jobs:
        return [], BuildStats()
    n_workers = workers if workers is not None else max(1, (os.cpu_count() or 1))
    perm = np.random.default_rng(seed).permutation(len(jobs))
    payloads = [
        (int(order), jobs[int(job_i)], seed, max_attempts_per_sentence, max_seq_len)
        for order, job_i in enumerate(perm)
    ]

    rows: list[dict] = []
    stat_parts: list[dict] = []
    progress = tqdm(total=target, desc="examples", disable=not show_progress)

    if n_workers <= 1:
        local_engine = engine or default_engine()
        global _WORKER_ENGINE
        _WORKER_ENGINE = local_engine
        try:
            for payload in payloads:
                if len(rows) >= target:
                    break
                _, found, part = _process_job(payload)
                stat_parts.append(part)
                if found:
                    take = found[: max(0, target - len(rows))]
                    rows.extend(take)
                    progress.update(len(take))
        finally:
            _WORKER_ENGINE = None
    else:
        chunk = max(32, n_workers * 16)
        cursor = 0
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as pool:
            while len(rows) < target and cursor < len(payloads):
                batch = payloads[cursor : cursor + chunk]
                cursor += len(batch)
                results = list(pool.map(_process_job, batch, chunksize=8))
                results.sort(key=lambda item: item[0])
                for _, found, part in results:
                    stat_parts.append(part)
                    if not found or len(rows) >= target:
                        continue
                    take = found[: max(0, target - len(rows))]
                    rows.extend(take)
                    progress.update(len(take))

    progress.close()
    # Recompute precise length stats on the kept rows.
    stats = _merge_stat_dicts(stat_parts)
    stats.kept = len(rows)
    stats.typo_lengths = [len(str(r["typo"]).encode("utf-8")) for r in rows]
    stats.serialized_lengths = []
    for row in rows:
        padded = [row.get(f"cand_{i}") for i in range(N_CANDIDATES)]
        try:
            ser = serialize_example(
                row["context_before"],
                row["typo"],
                row["context_after"],
                padded,
                max_seq_len=max_seq_len,
                gold_index=int(row["gold_index"]),
            )
            stats.serialized_lengths.append(ser.seq_len)
        except PathologicalExampleError:
            continue
    stats.corruption_type = Counter(r["corruption_type"] for r in rows)
    stats.gold_index = Counter(int(r["gold_index"]) for r in rows)
    stats.source_mixture = Counter(r["source"] for r in rows)
    stats.candidate_count = Counter(
        sum(1 for i in range(N_CANDIDATES) if r.get(f"cand_{i}") not in (None, "")) for r in rows
    )
    return rows, stats


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


def write_processed(
    train_rows: list[dict],
    valid_rows: list[dict],
    *,
    out_dir: Path,
    seed: int,
    source_meta: dict,
    train_stats: BuildStats,
    valid_stats: BuildStats,
    elapsed_sec: float,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.parquet"
    valid_path = out_dir / "validation.parquet"
    rows_to_frame(train_rows).to_parquet(train_path, index=False)
    rows_to_frame(valid_rows).to_parquet(valid_path, index=False)

    def _file_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    stats = {
        "seed": seed,
        "elapsed_sec": elapsed_sec,
        "train": {"n": len(train_rows), **train_stats.as_dict()},
        "validation": {"n": len(valid_rows), **valid_stats.as_dict()},
        "source": source_meta,
        "created_unix": int(time.time()),
    }
    manifest = {
        "seed": seed,
        "files": {
            "train": {
                "path": str(train_path.as_posix()),
                "n": len(train_rows),
                "sha256": _file_hash(train_path),
                "bytes": train_path.stat().st_size,
            },
            "validation": {
                "path": str(valid_path.as_posix()),
                "n": len(valid_rows),
                "sha256": _file_hash(valid_path),
                "bytes": valid_path.stat().st_size,
            },
        },
        "source": source_meta,
        "notes": [
            "Synthetic typos derived from WikiText-103 raw only.",
            "Authentic typo corpora were omitted due to redistribution uncertainty.",
            "Held-out benchmark data is never used here.",
        ],
    }
    (out_dir / "data_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
