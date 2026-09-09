#!/usr/bin/env python3
"""Print the headline table for a finished run.

Ties together the three artifacts a run produces -- the training summary, the
BEA-60K results, and the typo calibration -- and states plainly whether the
75% target was met and where the remaining error mass sits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "artifacts" / "train_summary.json")
    parser.add_argument("--results", type=Path, default=ROOT / "reports" / "bea60k" / "results.json")
    parser.add_argument("--target", type=float, default=0.75)
    args = parser.parse_args()

    summary = load(args.summary)
    results = load(args.results)

    if summary:
        print("== training ==")
        print(f"  parameters          {summary.get('total_trainable_parameters', 0):,}")
        print(f"  train / valid       {summary.get('train_examples', 0):,} / {summary.get('validation_examples', 0):,}")
        print(f"  optimizer steps     {summary.get('optimizer_steps')}")
        print(f"  duration            {summary.get('training_duration_sec', 0) / 3600:.2f} h")
        print(f"  gpu                 {summary.get('gpu')}")
        print(f"  best val loss       {summary.get('best_validation_loss'):.4f}")
        print(f"  best val top-1      {pct(summary.get('best_validation_acc_top1'))}")
        last = summary.get("last_validation") or {}
        print(f"  val top-1 @slot0    {pct(last.get('acc_gold_index_0'))}")
        print(f"  val top-1 @slot>0   {pct(last.get('acc_gold_index_nonzero'))}")

    if not results:
        print("\nno BEA-60K results yet")
        return 0

    n = results.get("n_word_errors") or 0
    oracle = results.get("hunspell_oracle_at_slots")
    cond = results.get("model_conditional_accuracy")
    overall = results.get("model_overall_success")
    aspell = results.get("aspell_top1")

    print("\n== BEA-60K ==")
    print(f"  word errors         {n:,}")
    print(f"  Hunspell top-1      {pct(results.get('hunspell_top1'))}")
    print(f"  Hunspell oracle@10  {pct(results.get('hunspell_oracle_at_10'))}")
    print(f"  Hunspell oracle@16  {pct(oracle)}  <- ceiling")
    print(f"  Aspell top-1        {pct(aspell)}")
    print(f"  model conditional   {pct(cond)}")
    print(f"  model overall       {pct(overall)}")

    print("\n== target ==")
    if overall is None or oracle is None:
        print("  incomplete run")
        return 0
    print(f"  target              {pct(args.target)}")
    print(f"  achieved            {pct(overall)}   {'MET' if overall >= args.target else 'NOT MET'}")
    needed = args.target / oracle if oracle else None
    if needed is not None:
        print(f"  conditional needed  {pct(needed)} (had {pct(cond)})")
    if aspell is not None:
        print(f"  vs Aspell           {100 * (overall - aspell):+.2f} pp")

    missing = 1.0 - oracle
    print("\n== where the remaining error sits ==")
    print(f"  gold outside pool   {pct(missing)}  (unreachable: Hunspell never offers it)")
    if cond is not None:
        print(f"  model picked wrong  {pct(oracle * (1 - cond))}  (reachable: better reranking)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
