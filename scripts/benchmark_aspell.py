#!/usr/bin/env python3
"""Aspell top-1 baseline on BEA-60K."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spelling_reranker.aspell import AspellEngine, aspell_metadata
from spelling_reranker.byte_encoding import nfc


def _load_bea_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_bea60k", ROOT / "scripts" / "benchmark_bea60k.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load benchmark_bea60k.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bea-dir", type=Path, default=ROOT / "data" / "bea60k")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "bea60k" / "aspell_baseline.json")
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    bea = _load_bea_module()
    pairs = bea.load_bea_pairs(args.bea_dir)
    errors = bea.extract_word_errors(pairs)
    if args.max_examples is not None:
        errors = errors[: args.max_examples]

    engine = AspellEngine()
    ok = 0
    no_sug = 0
    try:
        for err in errors:
            top = engine.top1(err["typo"])
            if top is None:
                no_sug += 1
                continue
            if nfc(top) == nfc(err["gold"]):
                ok += 1
    finally:
        engine.close()

    n = len(errors)
    payload = {
        "n": n,
        "aspell_top1": ok / n if n else 0.0,
        "no_suggestion": no_sug,
        "correct": ok,
        "metadata": aspell_metadata(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
