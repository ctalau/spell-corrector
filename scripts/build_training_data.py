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
    collect_sentence_jobs,
    generate_examples,
    iter_wikitext_articles,
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
    parser.add_argument("--target-train", type=int, default=240_000)
    parser.add_argument("--target-valid", type=int, default=20_000)
    parser.add_argument("--max-train-articles", type=int, default=None)
    parser.add_argument("--max-valid-articles", type=int, default=None)
    parser.add_argument("--max-attempts-per-sentence", type=int, default=6)
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

    print("collecting train sentences...")
    train_jobs = collect_sentence_jobs(
        iter_wikitext_articles(train_raw),
        split_name="train",
        source="wikitext103-synthetic",
        max_articles=args.max_train_articles,
    )
    print(f"train sentences with eligible words: {len(train_jobs)}")
    print("collecting validation sentences...")
    valid_jobs = collect_sentence_jobs(
        iter_wikitext_articles(valid_raw),
        split_name="valid",
        source="wikitext103-synthetic",
        max_articles=args.max_valid_articles,
    )
    print(f"valid sentences with eligible words: {len(valid_jobs)}")

    engine = default_engine()
    started = time.time()
    train_rows, train_stats = generate_examples(
        train_jobs,
        target=args.target_train,
        seed=args.seed,
        engine=engine,
        max_attempts_per_sentence=args.max_attempts_per_sentence,
        workers=args.workers,
    )
    valid_rows, valid_stats = generate_examples(
        valid_jobs,
        target=args.target_valid,
        seed=args.seed + 1,
        engine=engine,
        max_attempts_per_sentence=args.max_attempts_per_sentence,
        workers=args.workers,
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
    )
    write_hunspell_metadata(ROOT / "artifacts" / "hunspell_metadata.json", engine.metadata())
    print(json.dumps({"train": len(train_rows), "valid": len(valid_rows), "elapsed_sec": elapsed}, indent=2))
    print(json.dumps(manifest["files"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
