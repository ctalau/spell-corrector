#!/usr/bin/env python3
"""LLM-judge experiment on the locked BEA-60K benchmark.

Prompts a small instruction-tuned LLM to pick the best Hunspell suggestion
for each BEA-60K word error, and measures per-call latency. This is a
separate experiment track from the trained byte-level reranker
(scripts/benchmark_bea60k.py) -- here the "reranker" is a general-purpose
LLM prompted zero-shot, not a purpose-trained model.

Requires a GPU (transformers + torch); run this on Runpod, never on the
local CPU-only box (see CLAUDE.md). See scripts/runpod/bootstrap_llm_judge.sh.

Two phases per model, both drawn from the same seeded shuffle of BEA errors
that Hunspell flagged and produced at least one suggestion for, so the two
models see directly comparable examples:

  1. A fixed random sample (--n-samples, default 100): accuracy + latency.
  2. A wall-clock-budgeted run (--time-budget-seconds, default 300 = 5 min):
     as many examples as fit in the budget, for throughput/latency at scale.

Usage:
    python scripts/llm_judge_bea60k.py \\
        --model-id Qwen/Qwen3.5-0.8B --model-name qwen3.5-0.8b \\
        --bea-dir data/bea60k --output reports/llm_judge/qwen3.5-0.8b
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spelling_reranker.bea60k import extract_word_errors, load_bea_pairs
from spelling_reranker.byte_encoding import nfc
from spelling_reranker.candidates import build_pool
from spelling_reranker.hunspell import default_engine
from spelling_reranker.llm_judge import (
    build_messages,
    build_open_messages,
    generate_once,
    latency_stats,
    load_llm,
    parse_choice,
    parse_open_word,
    write_latency_histogram,
)


def build_eligible_errors(errors: list[dict], hunspell, max_candidates: int) -> tuple[list[dict], dict]:
    """Attach Hunspell suggestions to every error; cache suggest() by typo."""
    cache: dict[str, tuple[bool, list[str]]] = {}
    n_flagged = 0
    for err in errors:
        typo = err["typo"]
        hit = cache.get(typo)
        if hit is None:
            flagged = not hunspell.spell(typo)
            suggestions = hunspell.suggest(typo) if flagged else []
            hit = (flagged, suggestions)
            cache[typo] = hit
        flagged, suggestions = hit
        if flagged:
            n_flagged += 1
        pool = build_pool(suggestions, limit=max_candidates)
        err["hunspell_flagged"] = flagged
        err["candidates"] = pool
        err["gold_n"] = nfc(err["gold"])
        err["gold_in_pool"] = any(nfc(c) == err["gold_n"] for c in pool)
        err["hunspell_top1_ok"] = bool(pool) and nfc(pool[0]) == err["gold_n"]
    payload_meta = {"n_word_errors": len(errors), "n_hunspell_flagged": n_flagged}
    return errors, payload_meta


def run_phase(
    loaded,
    indices: list[int],
    errors: list[dict],
    *,
    mode: str,
    max_new_tokens: int,
    time_budget_seconds: float | None,
    max_examples: int | None,
) -> dict:
    predictions: list[dict] = []
    latencies: list[float] = []
    start = time.perf_counter()
    i = 0
    n = len(indices)
    while True:
        if max_examples is not None and len(predictions) >= max_examples:
            break
        if time_budget_seconds is not None and (time.perf_counter() - start) >= time_budget_seconds:
            break
        if n == 0:
            break
        err = errors[indices[i % n]]
        i += 1
        if mode == "open":
            messages = build_open_messages(
                err["context_before"], err["typo"], err["context_after"], err["candidates"]
            )
        else:
            messages = build_messages(err["context_before"], err["typo"], err["context_after"], err["candidates"])
        try:
            text, latency = generate_once(loaded, messages, max_new_tokens=max_new_tokens)
        except Exception as exc:  # noqa: BLE001
            predictions.append({**_slim(err), "error": str(exc)})
            continue
        if mode == "open":
            choice = None
            chosen_word = parse_open_word(text)
        else:
            choice = parse_choice(text, len(err["candidates"]))
            chosen_word = err["candidates"][choice - 1] if choice is not None else None
        correct = chosen_word is not None and nfc(chosen_word) == err["gold_n"]
        latencies.append(latency)
        predictions.append(
            {
                **_slim(err),
                "raw_output": text,
                "choice_index": choice,
                "chosen_word": chosen_word,
                "chosen_word_in_pool": chosen_word is not None
                and any(nfc(chosen_word) == nfc(c) for c in err["candidates"]),
                "correct": correct,
                "latency_s": latency,
            }
        )
    elapsed = time.perf_counter() - start
    n_scored = len(predictions)
    n_correct = sum(1 for p in predictions if p.get("correct"))
    cond_pool = [p for p in predictions if p.get("gold_in_pool")]
    cond_correct = sum(1 for p in cond_pool if p.get("correct"))
    hunspell_top1_on_sample = sum(1 for p in predictions if p.get("hunspell_top1_ok"))
    no_answer = sum(1 for p in predictions if p.get("chosen_word") is None and "error" not in p)
    errors_raised = sum(1 for p in predictions if "error" in p)
    # Only meaningful in "open" mode: how often the model stepped outside the
    # shown Hunspell candidates, and whether doing so was correct -- this is
    # what distinguishes "open" from "index" mode, where escaping the pool is
    # impossible by construction.
    outside_pool = [p for p in predictions if p.get("chosen_word") is not None and not p.get("chosen_word_in_pool")]
    outside_pool_correct = sum(1 for p in outside_pool if p.get("correct"))
    not_in_pool = [p for p in predictions if not p.get("gold_in_pool")]
    not_in_pool_correct = sum(1 for p in not_in_pool if p.get("correct"))
    return {
        "n_requested": n_scored,
        "n_ok": len(latencies),
        "n_errors_raised": errors_raised,
        "n_no_answer": no_answer,
        "elapsed_seconds": elapsed,
        "throughput_qps": (len(latencies) / elapsed) if elapsed > 0 else None,
        "overall_accuracy": (n_correct / n_scored) if n_scored else None,
        "conditional_accuracy": (cond_correct / len(cond_pool)) if cond_pool else None,
        "n_gold_in_pool": len(cond_pool),
        "hunspell_top1_accuracy_on_sample": (hunspell_top1_on_sample / n_scored) if n_scored else None,
        "n_chosen_outside_pool": len(outside_pool),
        "n_chosen_outside_pool_correct": outside_pool_correct,
        "accuracy_when_gold_outside_pool": (not_in_pool_correct / len(not_in_pool)) if not_in_pool else None,
        "latency_stats": latency_stats(latencies),
        "predictions": predictions,
        "_latencies": latencies,
    }


def _fmt(value, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def _slim(err: dict) -> dict:
    return {
        "typo": err["typo"],
        "gold": err["gold"],
        "context_before": err["context_before"],
        "context_after": err["context_after"],
        "candidates": err["candidates"],
        "gold_in_pool": err["gold_in_pool"],
        "hunspell_top1_ok": err["hunspell_top1_ok"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", required=True, help="HF model id, e.g. Qwen/Qwen3.5-0.8B")
    parser.add_argument("--model-name", required=True, help="short label for output paths/reports")
    parser.add_argument("--bea-dir", type=Path, default=ROOT / "data" / "bea60k")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--time-budget-seconds", type=float, default=300.0)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--dtype", default="auto", help="'auto' -> bfloat16 (memory-safe on CPU too)")
    parser.add_argument(
        "--answer-mode",
        choices=("index", "open"),
        default="index",
        help="'index': pick a candidate number (default). 'open': same prompt/candidates "
        "shown as a hint, but the model may write any word, not just one of them.",
    )
    parser.add_argument("--skip-timed", action="store_true", help="only run the fixed n-samples phase")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    pairs = load_bea_pairs(args.bea_dir)
    errors = extract_word_errors(pairs)
    hunspell = default_engine()
    errors, hunspell_meta = build_eligible_errors(errors, hunspell, args.max_candidates)

    eligible = [i for i, e in enumerate(errors) if e["hunspell_flagged"] and e["candidates"]]
    if not eligible:
        raise SystemExit("no eligible BEA-60K errors (Hunspell flagged + has suggestions)")
    rng = random.Random(args.seed)
    order = eligible[:]
    rng.shuffle(order)
    sample_100 = order[: args.n_samples]

    print(
        f"{len(errors)} word errors, {hunspell_meta['n_hunspell_flagged']} hunspell-flagged, "
        f"{len(eligible)} eligible (flagged + has suggestions)",
        flush=True,
    )

    meta = {
        "model_id": args.model_id,
        "model_name": args.model_name,
        "seed": args.seed,
        "n_samples": args.n_samples,
        "time_budget_seconds": args.time_budget_seconds,
        "max_candidates": args.max_candidates,
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "answer_mode": args.answer_mode,
        "bea_n_word_errors": hunspell_meta["n_word_errors"],
        "bea_n_hunspell_flagged": hunspell_meta["n_hunspell_flagged"],
        "bea_n_eligible": len(eligible),
        "hunspell_metadata": hunspell.metadata(),
        "python": platform.python_version(),
    }

    print(f"loading {args.model_id} ...", flush=True)
    t_load0 = time.perf_counter()
    try:
        loaded = load_llm(args.model_id, dtype=args.dtype)
    except Exception as exc:  # noqa: BLE001
        (args.output / "results.json").write_text(
            json.dumps({**meta, "load_error": str(exc), "traceback": traceback.format_exc()}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"FAILED to load {args.model_id}: {exc}", file=sys.stderr)
        return 1
    load_seconds = time.perf_counter() - t_load0

    import torch
    import transformers

    meta["load_seconds"] = load_seconds
    meta["load_class"] = loaded.load_class
    meta["device"] = str(loaded.device)
    meta["torch_version"] = torch.__version__
    meta["transformers_version"] = transformers.__version__
    meta["supports_system_role"] = loaded.supports_system_role
    meta["supports_enable_thinking"] = loaded.supports_enable_thinking
    if loaded.device.type == "cuda":
        meta["gpu_name"] = torch.cuda.get_device_name(0)

    # Warm up: first call pays for CUDA kernel compilation / cache warming and
    # is excluded from every latency stat below.
    warm_err = errors[sample_100[0]]
    warm_builder = build_open_messages if args.answer_mode == "open" else build_messages
    warm_messages = warm_builder(
        warm_err["context_before"], warm_err["typo"], warm_err["context_after"], warm_err["candidates"]
    )
    _, warmup_latency = generate_once(loaded, warm_messages, max_new_tokens=args.max_new_tokens)
    meta["warmup_latency_s"] = warmup_latency
    print(f"loaded in {load_seconds:.1f}s (warmup call {warmup_latency * 1000:.0f}ms)", flush=True)

    print(f"phase 1: fixed sample of {len(sample_100)} (mode={args.answer_mode})", flush=True)
    phase1 = run_phase(
        loaded,
        sample_100,
        errors,
        mode=args.answer_mode,
        max_new_tokens=args.max_new_tokens,
        time_budget_seconds=None,
        max_examples=len(sample_100),
    )
    write_latency_histogram(phase1["_latencies"], args.output, "sample100")
    (args.output / "predictions_sample100.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in phase1["predictions"]), encoding="utf-8"
    )
    print(
        f"  n={phase1['n_ok']} accuracy={_fmt(phase1['overall_accuracy'], '.3f')} "
        f"p50={_fmt(phase1['latency_stats'].get('p50_ms'), '.0f')}ms",
        flush=True,
    )

    phase2 = None
    if not args.skip_timed:
        print(f"phase 2: timed run, budget {args.time_budget_seconds:.0f}s (mode={args.answer_mode})", flush=True)
        phase2 = run_phase(
            loaded,
            order,
            errors,
            mode=args.answer_mode,
            max_new_tokens=args.max_new_tokens,
            time_budget_seconds=args.time_budget_seconds,
            max_examples=None,
        )
        write_latency_histogram(phase2["_latencies"], args.output, "timed")
        (args.output / "predictions_timed.jsonl").write_text(
            "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in phase2["predictions"]), encoding="utf-8"
        )
        print(
            f"  n={phase2['n_ok']} elapsed={phase2['elapsed_seconds']:.1f}s "
            f"qps={_fmt(phase2['throughput_qps'], '.2f')} accuracy={_fmt(phase2['overall_accuracy'], '.3f')}",
            flush=True,
        )
        phase2.pop("predictions", None)
        phase2.pop("_latencies", None)

    phase1.pop("predictions", None)
    phase1.pop("_latencies", None)

    results = {**meta, "sample_100": phase1, "timed_5min": phase2}
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
