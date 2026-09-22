#!/usr/bin/env python3
"""Paired comparison of two prediction files over the same rows.

Two checkpoints scored on a 2,000-row split have 95% intervals near ±1.5 pp,
which overlap for any difference worth arguing about. But they answered the
*same* rows, so the disagreements can be counted directly and tested with
McNemar's exact test — which is what decides whether a quantization tax is
real or is three rows of noise.

    python scripts/distill/compare_predictions.py --baseline fp16.jsonl \\
        --candidate w8a8.jsonl --examples 5
"""

from __future__ import annotations

import argparse
import json
import math
from math import comb
from pathlib import Path


def load(path: Path) -> dict[int, dict]:
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[int(row["error_index"])] = row
    return rows


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--field", default="casefold", choices=("casefold", "exact"))
    parser.add_argument("--examples", type=int, default=0, help="print this many disagreements")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    base, cand = load(args.baseline), load(args.candidate)
    keys = sorted(set(base) & set(cand))
    if not keys:
        raise SystemExit("the two files share no error_index")

    agree = sum(1 for k in keys if base[k]["pred"].casefold() == cand[k]["pred"].casefold())
    broke = [k for k in keys if base[k][args.field] and not cand[k][args.field]]
    fixed = [k for k in keys if not base[k][args.field] and cand[k][args.field]]
    base_correct = sum(1 for k in keys if base[k][args.field])
    cand_correct = sum(1 for k in keys if cand[k][args.field])

    report = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "field": args.field,
        "n": len(keys),
        "baseline_accuracy": base_correct / len(keys),
        "candidate_accuracy": cand_correct / len(keys),
        "baseline_ci95": wilson(base_correct, len(keys)),
        "candidate_ci95": wilson(cand_correct, len(keys)),
        "prediction_agreement": agree / len(keys),
        "baseline_right_candidate_wrong": len(broke),
        "baseline_wrong_candidate_right": len(fixed),
        "net": cand_correct - base_correct,
        "mcnemar_exact_p": mcnemar_exact(len(broke), len(fixed)),
    }
    print(json.dumps(report, indent=2))
    for k in (broke + fixed)[: args.examples]:
        print(f"  {k}: typo={base[k]['typo']!r} gold={base[k]['gold']!r} "
              f"baseline={base[k]['pred']!r} candidate={cand[k]['pred']!r}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
