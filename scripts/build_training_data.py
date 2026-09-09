#!/usr/bin/env python3
"""Build synthetic Hunspell-reranking parquet datasets from WikiText-103."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spelling_reranker.data_build import (
    build_typo_table,
    count_vocabulary,
    generate_examples,
    iter_sentences,
    select_vocabulary,
    write_processed,
)
from spelling_reranker.hunspell import default_engine, write_hunspell_metadata
from spelling_reranker.seed import DEFAULT_SEED


def _find_wikitext(raw_dir: Path) -> Path:
    for child in raw_dir.rglob("wiki.train.raw"):
        return child.parent
    raise FileNotFoundError(
        f"wiki.train.raw not found under {raw_dir}. Run scripts/download_sources.py first."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target-train", type=int, default=4_000_000)
    parser.add_argument("--max-passes", type=int, default=3)
    parser.add_argument("--target-valid", type=int, default=60_000)
    parser.add_argument("--vocab-size", type=int, default=120_000)
    parser.add_argument("--min-typos", type=int, default=3)
    parser.add_argument("--max-typos", type=int, default=32)
    parser.add_argument("--context-noise-prob", type=float, default=0.25)
    parser.add_argument("--gold0-fraction", type=float, default=0.65)
    parser.add_argument("--max-uses-per-typo", type=int, default=6)
    parser.add_argument("--max-train-articles", type=int, default=None)
    parser.add_argument("--max-valid-articles", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None, help="Process workers (default: CPU count)")
    args = parser.parse_args()

    wt_dir = _find_wikitext(args.raw_dir)
    train_raw = wt_dir / "wiki.train.raw"
    valid_raw = wt_dir / "wiki.valid.raw"
    source_meta = {
        "dataset": "WikiText-103 raw",
        "config": "wikitext-103-raw-v1",
        "identifier": "Salesforce/wikitext wikitext-103-raw-v1",
        "license": "CC BY-SA",
        "train_file": "data/raw/wikitext-103-raw/wiki.train.raw",
        "valid_file": "data/raw/wikitext-103-raw/wiki.valid.raw",
        "download": "scripts/download_sources.py",
        "split_policy": "official WikiText train/valid article files",
    }

    started = time.time()

    print("pass 1: counting vocabulary...", flush=True)
    counts = count_vocabulary(train_raw, max_articles=args.max_train_articles)
    vocabulary = select_vocabulary(counts, vocab_size=args.vocab_size)
    print(f"  eligible word types: {len(counts):,} -> vocabulary {len(vocabulary):,}", flush=True)
    t_vocab = time.time()

    print("pass 2: building typo table (Hunspell)...", flush=True)
    typo_table, table_stats = build_typo_table(
        vocabulary,
        seed=args.seed,
        workers=args.workers,
        min_typos=args.min_typos,
        max_typos=args.max_typos,
    )
    n_entries = sum(len(v) for v in typo_table.values())
    print(f"  words with usable typos: {len(typo_table):,}; entries: {n_entries:,}", flush=True)
    t_table = time.time()

    print("pass 3: instantiating examples...", flush=True)
    train_rows, train_stats = generate_examples(
        lambda: iter_sentences(train_raw, max_articles=args.max_train_articles),
        typo_table,
        target=args.target_train,
        seed=args.seed,
        context_noise_prob=args.context_noise_prob,
        gold0_fraction=args.gold0_fraction,
        max_uses_per_typo=args.max_uses_per_typo,
        max_passes=args.max_passes,
    )
    valid_rows, valid_stats = generate_examples(
        lambda: iter_sentences(valid_raw, max_articles=args.max_valid_articles),
        typo_table,
        target=args.target_valid,
        seed=args.seed + 1,
        context_noise_prob=args.context_noise_prob,
        gold0_fraction=args.gold0_fraction,
        max_uses_per_typo=args.max_uses_per_typo,
        max_passes=args.max_passes,
    )
    elapsed = time.time() - started

    manifest = write_processed(
        train_rows,
        valid_rows,
        out_dir=args.out_dir,
        seed=args.seed,
        source_meta=source_meta,
        train_stats=train_stats,
        valid_stats=valid_stats,
        elapsed_sec=elapsed,
        extra={
            "typo_table": {
                "vocabulary": len(vocabulary),
                "words_with_typos": len(typo_table),
                "entries": n_entries,
                "stats": dict(table_stats),
            },
            "phase_seconds": {
                "vocabulary": t_vocab - started,
                "typo_table": t_table - t_vocab,
                "examples": time.time() - t_table,
            },
        },
    )
    write_hunspell_metadata(ROOT / "artifacts" / "hunspell_metadata.json", default_engine().metadata())
    print(json.dumps({"train": len(train_rows), "valid": len(valid_rows), "elapsed_sec": elapsed}, indent=2))
    print(json.dumps(manifest["files"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
