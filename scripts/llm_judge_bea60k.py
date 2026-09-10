#!/usr/bin/env python3
"""LLM-judge experiment on the locked BEA-60K benchmark.

Prompts a small instruction-tuned LLM to pick the best Hunspell suggestion
for each BEA-60K word error, and measures per-call latency. This is a
separate experiment track from the trained byte-level reranker
(scripts/benchmark_bea60k.py) -- here the "reranker" is a general-purpose
LLM prompted zero-shot, not a purpose-trained model.

Requires a GPU (transformers + torch); run this on Runpod, never on the
local CPU-only box (see CLAUDE.md). See scripts/runpod/bootstrap_llm_judge.sh.

Phases are drawn from the same seeded shuffle of BEA errors that Hunspell
flagged and produced at least one suggestion for, so models see directly
comparable examples. Default is two phases:

  1. A fixed random sample (--n-samples, default 100): accuracy + latency.
  2. A wall-clock-budgeted run (--time-budget-seconds, default 300 = 5 min):
     as many examples as fit in the budget (wrapping), for throughput/latency.

--skip-sample runs only the timed phase (the 1-hour Gemma path).
--full runs one non-wrapping pass over the eligible set as phase full_bea60k
(optionally still capped by --time-budget-seconds, e.g. 3600).

Usage:
    python scripts/llm_judge_bea60k.py \\
        --model-id Qwen/Qwen3.5-0.8B --model-name qwen3.5-0.8b \\
        --bea-dir data/bea60k --output reports/llm_judge/qwen3.5-0.8b
    python scripts/llm_judge_bea60k.py \\
        --model-id google/gemma-4-E2B-it --model-name gemma-4-e2b \\
        --bea-dir data/bea60k --output reports/llm_judge/gemma-4-e2b \\
        --skip-sample --time-budget-seconds 3600
"""

from __future__ import annotations

import json
import platform
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
    JudgePhase,
    build_messages,
    eligible_indices,
    generate_once,
    latency_stats,
    load_llm,
    parse_choice,
    parse_llm_judge_args,
    plan_phases,
    resolve_time_budget,
    shuffle_indices,
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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in rows), encoding="utf-8")


def summarize_predictions(predictions: list[dict], latencies: list[float], elapsed: float) -> dict:
    n_scored = len(predictions)
    n_correct = sum(1 for p in predictions if p.get("correct"))
    cond_pool = [p for p in predictions if p.get("gold_in_pool")]
    cond_correct = sum(1 for p in cond_pool if p.get("correct"))
    hunspell_top1_on_sample = sum(1 for p in predictions if p.get("hunspell_top1_ok"))
    no_answer = sum(1 for p in predictions if p.get("choice_index") is None and "error" not in p)
    errors_raised = sum(1 for p in predictions if "error" in p)
    hunspell_top1 = (hunspell_top1_on_sample / n_scored) if n_scored else None
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
        "hunspell_top1_accuracy_on_sample": hunspell_top1,
        "hunspell_top1_on_same_set": hunspell_top1,
        "latency_stats": latency_stats(latencies),
    }


def _eta_seconds(
    *,
    elapsed: float,
    n_done: int,
    time_budget_seconds: float | None,
    n_indices: int,
    wrap: bool,
) -> float | None:
    if time_budget_seconds is not None:
        return max(0.0, time_budget_seconds - elapsed)
    if wrap or n_done <= 0 or elapsed <= 0:
        return None
    remaining = n_indices - n_done
    if remaining <= 0:
        return 0.0
    return remaining / (n_done / elapsed)


def run_phase(
    loaded,
    indices: list[int],
    errors: list[dict],
    *,
    max_new_tokens: int,
    time_budget_seconds: float | None,
    max_examples: int | None,
    wrap: bool = True,
    progress_every: int = 500,
    checkpoint_every: int = 2000,
    checkpoint_path: Path | None = None,
) -> dict:
    predictions: list[dict] = []
    latencies: list[float] = []
    start = time.perf_counter()
    i = 0
    n = len(indices)

    def persist(elapsed: float) -> dict:
        summary = summarize_predictions(predictions, latencies, elapsed)
        if checkpoint_path is not None:
            _write_jsonl(checkpoint_path, predictions)
            sidecar = checkpoint_path.with_suffix(".checkpoint.json")
            sidecar.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary

    try:
        while True:
            if max_examples is not None and len(predictions) >= max_examples:
                break
            if time_budget_seconds is not None and (time.perf_counter() - start) >= time_budget_seconds:
                break
            if n == 0:
                break
            if not wrap and i >= n:
                break
            err = errors[indices[i % n]]
            i += 1
            messages = build_messages(err["context_before"], err["typo"], err["context_after"], err["candidates"])
            try:
                text, latency = generate_once(loaded, messages, max_new_tokens=max_new_tokens)
            except Exception as exc:  # noqa: BLE001
                predictions.append({**_slim(err), "error": str(exc)})
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
                        "correct": correct,
                        "latency_s": latency,
                    }
                )
            n_done = len(predictions)
            elapsed = time.perf_counter() - start
            if progress_every > 0 and n_done > 0 and n_done % progress_every == 0:
                summary = summarize_predictions(predictions, latencies, elapsed)
                eta = _eta_seconds(
                    elapsed=elapsed,
                    n_done=n_done,
                    time_budget_seconds=time_budget_seconds,
                    n_indices=n,
                    wrap=wrap,
                )
                print(
                    f"  progress n={n_done} acc={_fmt(summary['overall_accuracy'], '.3f')} "
                    f"elapsed={elapsed:.0f}s eta={_fmt(eta, '.0f')}s",
                    flush=True,
                )
            if checkpoint_every > 0 and n_done > 0 and n_done % checkpoint_every == 0:
                persist(elapsed)
                print(f"  checkpoint n={n_done} -> {checkpoint_path}", flush=True)
    finally:
        persist(time.perf_counter() - start)

    elapsed = time.perf_counter() - start
    summary = summarize_predictions(predictions, latencies, elapsed)
    summary["predictions"] = predictions
    summary["_latencies"] = latencies
    return summary


def _public_phase(phase: dict, *, device: str | None, gpu_name: str | None) -> dict:
    out = {k: v for k, v in phase.items() if not k.startswith("_") and k != "predictions"}
    out["device"] = device
    out["gpu_name"] = gpu_name
    return out


def _run_one_phase(
    loaded,
    phase: JudgePhase,
    errors: list[dict],
    *,
    max_new_tokens: int,
    output: Path,
    progress_every: int,
    checkpoint_every: int,
) -> dict:
    pred_path = output / f"predictions_{phase.predictions_stem}.jsonl"
    budget = phase.time_budget_seconds
    budget_note = "unbounded" if budget is None else f"{budget:.0f}s"
    print(
        f"phase {phase.name}: n_indices={len(phase.indices)} wrap={phase.wrap} budget={budget_note}",
        flush=True,
    )
    result = run_phase(
        loaded,
        list(phase.indices),
        errors,
        max_new_tokens=max_new_tokens,
        time_budget_seconds=budget,
        max_examples=phase.max_examples,
        wrap=phase.wrap,
        progress_every=progress_every,
        checkpoint_every=checkpoint_every,
        checkpoint_path=pred_path,
    )
    write_latency_histogram(result["_latencies"], output, phase.predictions_stem)
    _write_jsonl(pred_path, result["predictions"])
    print(
        f"  n={result['n_ok']} elapsed={result['elapsed_seconds']:.1f}s "
        f"qps={_fmt(result['throughput_qps'], '.2f')} "
        f"accuracy={_fmt(result['overall_accuracy'], '.3f')} "
        f"p50={_fmt(result['latency_stats'].get('p50_ms'), '.0f')}ms "
        f"p99={_fmt(result['latency_stats'].get('p99_ms'), '.0f')}ms",
        flush=True,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_llm_judge_args(argv)
    args.output = Path(args.output)
    args.bea_dir = Path(args.bea_dir) if args.bea_dir else ROOT / "data" / "bea60k"
    time_budget = resolve_time_budget(args)

    args.output.mkdir(parents=True, exist_ok=True)

    pairs = load_bea_pairs(args.bea_dir)
    errors = extract_word_errors(pairs)
    hunspell = default_engine()
    errors, hunspell_meta = build_eligible_errors(errors, hunspell, args.max_candidates)

    eligible = eligible_indices(errors)
    if not eligible:
        raise SystemExit("no eligible BEA-60K errors (Hunspell flagged + has suggestions)")
    order = shuffle_indices(eligible, args.seed)
    phases = plan_phases(args, order)
    if not phases:
        raise SystemExit("no scoring phases planned")

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
        "time_budget_seconds": time_budget,
        "full": bool(args.full),
        "skip_sample": bool(args.skip_sample),
        "also_sample": bool(args.also_sample),
        "max_candidates": args.max_candidates,
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "bea_n_word_errors": hunspell_meta["n_word_errors"],
        "bea_n_hunspell_flagged": hunspell_meta["n_hunspell_flagged"],
        "bea_n_eligible": len(eligible),
        "hunspell_metadata": hunspell.metadata(),
        "python": platform.python_version(),
        "phases": [p.name for p in phases],
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
    gpu_name = None
    if loaded.device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        meta["gpu_name"] = gpu_name

    # Warm up: first call pays for CUDA kernel compilation / cache warming and
    # is excluded from every latency stat below.
    warm_idx = phases[0].indices[0] if phases[0].indices else order[0]
    warm_err = errors[warm_idx]
    warm_messages = build_messages(
        warm_err["context_before"], warm_err["typo"], warm_err["context_after"], warm_err["candidates"]
    )
    _, warmup_latency = generate_once(loaded, warm_messages, max_new_tokens=args.max_new_tokens)
    meta["warmup_latency_s"] = warmup_latency
    print(f"loaded in {load_seconds:.1f}s (warmup call {warmup_latency * 1000:.0f}ms)", flush=True)

    phase_results: dict[str, dict] = {}
    for phase in phases:
        raw = _run_one_phase(
            loaded,
            phase,
            errors,
            max_new_tokens=args.max_new_tokens,
            output=args.output,
            progress_every=args.progress_every,
            checkpoint_every=args.checkpoint_every,
        )
        phase_results[phase.name] = _public_phase(raw, device=meta["device"], gpu_name=gpu_name)

    results = {**meta, **phase_results}
    # Backward-compatible alias for the original 5-minute wrapping phase key.
    if "timed" in phase_results:
        results["timed_5min"] = phase_results["timed"]
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
