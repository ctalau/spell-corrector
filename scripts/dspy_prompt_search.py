#!/usr/bin/env python3
"""Optimize the spelling-correction prompt with DSPy, against a non-benchmark dev set.

    python scripts/dspy_prompt_search.py \
        --base-url http://127.0.0.1:8080/v1 --model gemma-4-e2b-q4 \
        --output reports/gpu_llama/dspy

What it does, in order:

1. Builds (or loads) the development set from `spelling_reranker/dev_set.py` --
   authentic Wikipedia misspellings plus the repo's own typo generator, dropped
   into WikiText sentences, with Hunspell candidates from the same code path the
   scored harness uses. Prints its composition so it can be compared by hand
   with the population statistics in HANDOFF.md section 5.
2. Scores the **hand-written** prompt on the dev validation split, through the
   same `build_open_messages` the 87%-strict "open" baseline used. Without this
   an optimizer's number floats free: the question is never "is the optimized
   prompt good" but "is it better than what a person already wrote".
3. Scores each DSPy program zero-shot (no demos), then optimizes it.
4. Picks the winner on the dev validation split, saves it, and dumps the
   rendered prompt as plain text so a human can read exactly what DSPy produced.
5. With `--final-eval`, scores the frozen winner **once** on BEA-60K.

BEA-60K is a locked benchmark
-----------------------------
It is touched only in step 5, only with `--final-eval`, only after the program
is frozen and written to disk. Nothing about the program, the demos, the dev
set or the choice between programs depends on a BEA number, and the script
deliberately provides no way to loop on one: if the final number disappoints,
the honest move is to report it, not to re-run the search.

The BEA sample mirrors `scripts/llm_judge_bea60k_cpu.py` exactly -- the same
eligibility rule (Hunspell flagged it and returned at least one suggestion),
the same `random.Random(seed)` shuffle, the same first `--n-final` of it -- so
the number is directly comparable with the baselines in
reports/EXPERIMENT_LLM_JUDGE_CPU.md.

Cost control
------------
Every LM call goes through a counter that raises once `--budget` calls have
been made, so a misconfigured optimizer cannot quietly spend an hour of GPU
time. The predicted call count is printed before the search starts.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spelling_reranker import dspy_program as dp
from spelling_reranker.bea60k import extract_word_errors, load_bea_pairs
from spelling_reranker.byte_encoding import nfc
from spelling_reranker.candidates import build_pool
from spelling_reranker.dev_set import (
    DEFAULT_CACHE_DIR,
    MAX_CANDIDATES,
    DevExample,
    format_stats,
    load_or_build,
)
from spelling_reranker.hunspell import default_engine
from spelling_reranker.llama_cpp_backend import LlamaCppModel, LlamaServerError, normalize_base_url, wait_for_server
from spelling_reranker.llm_judge_cpu import build_open_messages, parse_open_word

DEFAULT_SUGGESTION_CACHE = ROOT / "data" / "bea60k" / "hunspell_suggestions.json"

#: The hand-written "open" mode answers with a single word and was measured at
#: `max_new_tokens=8`. The baseline is reproduced at that budget, not at the
#: (larger) budget DSPy needs for its field markup, so it is the same baseline.
BASELINE_MAX_TOKENS = 8


class BudgetExceeded(RuntimeError):
    pass


def make_budgeted_lm(base_url: str, model: str, max_tokens: int, budget: int):
    """A `dspy.LM` that counts its calls and refuses to exceed `budget`.

    DSPy optimizers decide how many candidate programs to evaluate from their
    own parameters; the arithmetic is easy to get wrong by a factor of ten, and
    on a rented GPU that is real money. This makes the bound explicit and
    enforced rather than estimated.
    """
    dspy = dp.require_dspy()
    base = dp.build_lm(base_url=base_url, model=model, max_tokens=max_tokens)

    class BudgetedLM(type(base)):  # type: ignore[misc]
        def __init__(self, inner, budget: int) -> None:
            self.__dict__.update(inner.__dict__)
            self._budget = budget
            self._calls = 0

        @property
        def n_calls(self) -> int:
            return self._calls

        def __call__(self, *args, **kwargs):
            if self._calls >= self._budget:
                raise BudgetExceeded(
                    f"LM call budget exhausted ({self._budget} calls). "
                    "Raise --budget or lower --num-candidates/--max-demos."
                )
            self._calls += 1
            return super().__call__(*args, **kwargs)

    assert dspy is not None
    return BudgetedLM(base, budget)


# --------------------------------------------------------------------------
# Hand-written baseline
# --------------------------------------------------------------------------


def handwritten_baseline_predict(server: LlamaCppModel, max_tokens: int):
    """The existing hand-written "open" prompt, as a `predict(example)` callable.

    This is the prompt that scored 87.0% strict on the q4_0 100-example sample,
    called through the same helpers the scored harness calls, so the comparison
    with DSPy is a prompt comparison and not a harness comparison.
    """

    class _Prediction:
        def __init__(self, word: str | None) -> None:
            self.corrected_word = word or ""

    def predict(example):
        sentence = example.sentence if hasattr(example, "sentence") else example["sentence"]
        typo = example.typo if hasattr(example, "typo") else example["typo"]
        candidates = example.candidates if hasattr(example, "candidates") else example["candidates"]
        if isinstance(candidates, str):
            candidates = [c.strip() for c in candidates.split(",") if c.strip() and c.strip() != "(none)"]
        before, _, rest = sentence.partition("<TYPO>")
        _, _, after = rest.partition("</TYPO>")
        messages = build_open_messages(before, typo, after, candidates)
        text, _ = server.generate(messages, max_new_tokens=max_tokens)
        return _Prediction(parse_open_word(text))

    return predict


# --------------------------------------------------------------------------
# Final held-out evaluation
# --------------------------------------------------------------------------


def load_bea_sample(bea_dir: Path, *, n: int, seed: int, max_candidates: int, cache_path: Path | None) -> list[DevExample]:
    """The same fixed sample the existing CPU harness scores, as DevExamples.

    Reuses that harness's persisted per-typo Hunspell memo when it is present
    and was written for this dictionary; the cold pass over ~68k errors costs
    about 13 minutes and is identical on every run.
    """
    engine = default_engine()
    pairs = load_bea_pairs(Path(bea_dir))
    errors = extract_word_errors(pairs)

    cache: dict[str, tuple[bool, list[str]]] = {}
    dictionary_key = json.dumps(
        {
            name: entry.get("sha256")
            for name, entry in sorted(engine.metadata().get("dictionary_hashes", {}).items())
        },
        sort_keys=True,
    )
    if cache_path and Path(cache_path).is_file():
        try:
            payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if payload.get("dictionary_key") == dictionary_key:
            cache = {
                typo: (bool(flagged), list(sugg))
                for typo, (flagged, sugg) in payload.get("entries", {}).items()
            }

    n_cached = len(cache)
    eligible: list[tuple[int, dict, list[str]]] = []
    for idx, err in enumerate(errors):
        typo = err["typo"]
        hit = cache.get(typo)
        if hit is None:
            flagged = not engine.spell(typo)
            hit = (flagged, engine.suggest(typo) if flagged else [])
            cache[typo] = hit
        flagged, suggestions = hit
        pool = build_pool(suggestions, limit=max_candidates) if flagged else []
        if flagged and pool:
            eligible.append((idx, err, pool))

    if cache_path and len(cache) > n_cached:
        # Persist the memo in the same format scripts/llm_judge_bea60k_cpu.py
        # reads, so whichever of the two runs first pays the ~13-minute cold
        # pass and the other starts warm. Written atomically: a truncated cache
        # would silently change what the locked benchmark is scored against.
        payload = {
            "dictionary_key": dictionary_key,
            "hunspell_metadata": engine.metadata(),
            "entries": {typo: [flagged, sugg] for typo, (flagged, sugg) in cache.items()},
        }
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(cache_path).with_suffix(Path(cache_path).suffix + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(cache_path)

    if not eligible:
        raise SystemExit("no eligible benchmark errors (Hunspell flagged + has suggestions)")

    order = list(range(len(eligible)))
    random.Random(seed).shuffle(order)
    chosen = order[:n]

    out: list[DevExample] = []
    for rank, position in enumerate(chosen):
        _, err, pool = eligible[position]
        gold_n = nfc(err["gold"])
        gold_index = next((i for i, c in enumerate(pool) if nfc(c) == gold_n), None)
        out.append(
            DevExample(
                example_id=f"bea-{rank:04d}",
                sentence=err["context_before"] + err["typo"] + err["context_after"],
                context_before=err["context_before"],
                typo=err["typo"],
                context_after=err["context_after"],
                gold=gold_n,
                candidates=tuple(pool),
                gold_index=gold_index,
                source="benchmark",
                corruption_type="human",
                edit_distance=-1,
            )
        )
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1", help="OpenAI-compatible endpoint of llama-server")
    parser.add_argument("--model", default="gemma-4-e2b-q4", help="model name sent to the server (any label it echoes back)")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "gpu_llama" / "dspy")
    parser.add_argument("--max-tokens", type=int, default=256, help="generation cap per LM call")

    parser.add_argument("--dev-size", type=int, default=300, help="development examples to build")
    parser.add_argument("--dev-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--authentic-fraction", type=float, default=0.5)
    parser.add_argument("--train-fraction", type=float, default=0.5, help="dev split: train vs validation")
    parser.add_argument("--rebuild-dev", action="store_true", help="ignore any cached dev set")
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument(
        "--programs",
        nargs="+",
        default=list(dp.PROGRAM_NAMES),
        choices=list(dp.PROGRAM_NAMES),
        help="which signatures to optimize and choose between",
    )
    parser.add_argument("--use-cot", action="store_true", help="wrap each signature in dspy.ChainOfThought (costs tokens)")
    parser.add_argument(
        "--optimizer",
        default="bootstrap-rs",
        choices=("bootstrap", "bootstrap-rs", "mipro"),
        help="bootstrap-rs is the default; see --help text in the module docstring",
    )
    parser.add_argument("--max-demos", type=int, default=3, help="max bootstrapped demonstrations")
    parser.add_argument("--max-labeled-demos", type=int, default=3, help="max labelled (non-bootstrapped) demonstrations")
    parser.add_argument("--num-candidates", type=int, default=4, help="candidate programs for bootstrap-rs / mipro")
    parser.add_argument("--budget", type=int, default=8000, help="hard cap on LM calls for the whole run")
    parser.add_argument("--num-threads", type=int, default=1, help="parallel LM calls (llama-server is started with -np 1)")

    parser.add_argument("--bea-dir", type=Path, default=ROOT / "data" / "bea60k")
    parser.add_argument("--final-eval", action="store_true", help="score the frozen winner once on the locked benchmark")
    parser.add_argument("--n-final", type=int, default=100, help="held-out sample size for --final-eval")
    parser.add_argument("--suggestion-cache", type=Path, default=DEFAULT_SUGGESTION_CACHE)
    parser.add_argument("--skip-baseline", action="store_true", help="do not score the hand-written prompt")
    return parser.parse_args(argv)


def estimate_calls(args: argparse.Namespace, n_train: int, n_val: int) -> int:
    """Rough upper bound on LM calls, printed before anything is spent."""
    per_program = n_val  # zero-shot evaluation
    if args.optimizer == "bootstrap":
        per_program += n_train + n_val
    else:
        # BootstrapFewShotWithRandomSearch evaluates `num_candidate_programs`
        # random seeds *plus* three fixed ones (zero-shot, labelled-demos-only,
        # and one unshuffled bootstrap), each costing at most a bootstrap pass
        # over the trainset and a full pass over the valset.
        per_program += (args.num_candidates + 3) * (n_train + n_val)
    baseline = 0 if args.skip_baseline else n_val
    final = args.n_final if args.final_eval else 0
    return baseline + per_program * len(args.programs) + final


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t_start = time.perf_counter()

    try:
        dspy = dp.require_dspy()
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    server_root = normalize_base_url(args.base_url)
    try:
        wait_for_server(server_root, timeout_s=30.0)
    except LlamaServerError as exc:
        print(
            f"ERROR: no llama-server at {server_root} ({exc}).\n"
            "Start one with: llama-server -m <gguf> --host 127.0.0.1 --port 8080 --reasoning off -np 1",
            file=sys.stderr,
        )
        return 3

    args.output.mkdir(parents=True, exist_ok=True)

    # ---- dev set -------------------------------------------------------
    dev, dev_path = load_or_build(
        args.dev_size,
        seed=args.seed,
        authentic_fraction=args.authentic_fraction,
        cache_dir=args.dev_cache_dir,
        rebuild=args.rebuild_dev,
    )
    train_raw, val_raw = dev.split(args.train_fraction)
    print(f"dev set: {dev_path}")
    print(format_stats(dev.stats))
    print(f"  split              train {len(train_raw)} / validation {len(val_raw)}", flush=True)
    (args.output / "dev_set_stats.json").write_text(
        json.dumps({"path": str(dev_path), "stats": dev.stats}, indent=2) + "\n", encoding="utf-8"
    )

    trainset = dp.to_dspy_examples(train_raw)
    valset = dp.to_dspy_examples(val_raw)
    if not trainset or not valset:
        print(
            f"ERROR: the dev set split is empty (train {len(trainset)}, validation {len(valset)}). "
            "Raise --dev-size or adjust --train-fraction.",
            file=sys.stderr,
        )
        return 5

    predicted = estimate_calls(args, len(trainset), len(valset))
    print(f"predicted LM calls <= {predicted} (budget {args.budget})", flush=True)
    if predicted > args.budget:
        print(
            f"ERROR: the configured search needs up to {predicted} LM calls but --budget is "
            f"{args.budget}. Lower --num-candidates/--dev-size or raise --budget.",
            file=sys.stderr,
        )
        return 4

    lm = make_budgeted_lm(server_root + "/v1", args.model, args.max_tokens, args.budget)
    dspy.configure(lm=lm)

    results: dict = {
        "config": {
            "base_url": server_root,
            "model": args.model,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "dev_size": args.dev_size,
            "dev_path": str(dev_path),
            "train_n": len(trainset),
            "val_n": len(valset),
            "programs": args.programs,
            "use_cot": args.use_cot,
            "optimizer": args.optimizer,
            "max_demos": args.max_demos,
            "max_labeled_demos": args.max_labeled_demos,
            "num_candidates": args.num_candidates,
            "budget": args.budget,
            "predicted_max_calls": predicted,
        },
        "dev_set_stats": dev.stats,
        "programs": {},
    }

    # ---- hand-written baseline ----------------------------------------
    if not args.skip_baseline:
        server = LlamaCppModel(model_id=args.model, base_url=server_root)
        print("\nbaseline: hand-written open-mode prompt on the dev validation split", flush=True)
        baseline = dp.evaluate(
            handwritten_baseline_predict(server, BASELINE_MAX_TOKENS), valset, progress_every=25
        )
        # One call per example, made directly over HTTP rather than through
        # DSPy, so it is counted separately from the optimizer's LM calls.
        results["baseline_handwritten"] = {**baseline.to_dict(), "lm_calls": baseline.n}
        print(
            f"  strict {baseline.strict_accuracy:.1%}  lenient {baseline.lenient_accuracy:.1%} "
            f"({baseline.wall_clock_s:.0f}s)",
            flush=True,
        )
        (args.output / "predictions_baseline.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in baseline.records), encoding="utf-8"
        )

    # ---- optimize each program ----------------------------------------
    best_name = None
    best_score = -1.0
    best_program = None
    for name in args.programs:
        print(f"\nprogram {name}: zero-shot on the dev validation split", flush=True)
        program = dp.build_program(name, use_cot=args.use_cot)
        zero_shot = dp.evaluate_program(program, valset, progress_every=25)
        print(f"  strict {zero_shot.strict_accuracy:.1%}  lenient {zero_shot.lenient_accuracy:.1%}", flush=True)

        print(f"program {name}: optimizing with {args.optimizer}", flush=True)
        try:
            optimized = run_optimizer(dspy, args, program, trainset, valset)
            optimize_error = None
        except BudgetExceeded as exc:
            print(f"  {exc}", file=sys.stderr)
            optimized, optimize_error = program, str(exc)

        tuned = dp.evaluate_program(optimized, valset, progress_every=25)
        print(f"  optimized strict {tuned.strict_accuracy:.1%}  lenient {tuned.lenient_accuracy:.1%}", flush=True)

        program_path = args.output / f"program_{name}.json"
        optimized.save(str(program_path))
        prompt_path = args.output / f"prompt_{name}.txt"
        prompt_path.write_text(dp.render_prompt(optimized, valset[0]), encoding="utf-8")
        (args.output / f"predictions_dev_{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in tuned.records), encoding="utf-8"
        )

        results["programs"][name] = {
            "zero_shot": zero_shot.to_dict(),
            "optimized": tuned.to_dict(),
            "optimize_error": optimize_error,
            "description": dp.describe_program(optimized),
            "saved_program": str(program_path),
            "rendered_prompt": str(prompt_path),
        }
        if tuned.strict_accuracy > best_score:
            best_name, best_score, best_program = name, tuned.strict_accuracy, optimized

    results["selected_program"] = best_name
    results["selected_dev_strict_accuracy"] = best_score
    print(f"\nselected on the dev validation split: {best_name} ({best_score:.1%} strict)", flush=True)

    # ---- single held-out evaluation ------------------------------------
    if args.final_eval and best_program is not None:
        print("\nfinal evaluation: one pass over the locked benchmark sample, frozen prompt", flush=True)
        sample = load_bea_sample(
            args.bea_dir,
            n=args.n_final,
            seed=args.seed,
            max_candidates=MAX_CANDIDATES,
            cache_path=args.suggestion_cache,
        )
        final_examples = dp.to_dspy_examples(sample)
        final = dp.evaluate_program(best_program, final_examples, progress_every=25)
        results["final_eval"] = {
            "program": best_name,
            "sample_size": len(final_examples),
            "seed": args.seed,
            "bea_dir": str(args.bea_dir),
            **final.to_dict(),
        }
        (args.output / "predictions_final.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in final.records), encoding="utf-8"
        )
        print(
            f"  strict {final.strict_accuracy:.1%}  lenient {final.lenient_accuracy:.1%} "
            f"(n={final.n})",
            flush=True,
        )

    #: DSPy LM calls only -- the hand-written baseline's calls are counted in
    #: results["baseline_handwritten"]["lm_calls"].
    results["lm_calls"] = int(getattr(lm, "n_calls", -1))
    results["wall_clock_s"] = time.perf_counter() - t_start
    (args.output / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "summary.txt").write_text(summarize(results) + "\n", encoding="utf-8")
    print("\n" + summarize(results))
    print(f"\nwrote {args.output}/results.json ({results['lm_calls']} LM calls, "
          f"{results['wall_clock_s']:.0f}s)")
    return 0


def run_optimizer(dspy, args: argparse.Namespace, program, trainset, valset):
    """Run the chosen DSPy optimizer and return the compiled program.

    `bootstrap-rs` (BootstrapFewShotWithRandomSearch) is the default and the
    safe choice here: it only needs the task model itself, its cost is a
    predictable `num_candidates x |train ∪ val|` evaluations, and it targets the
    lever that matters most for a 2B model -- concrete demonstrations of the
    exact output format. MIPROv2 additionally *proposes instruction text* with
    an LM; asked to do that with the same small quantized model it is optimizing
    (the only model on the pod), the proposals are weak and the extra long-form
    generations are expensive on a CPU/GPU box serving one request at a time.
    It stays available behind `--optimizer mipro` for a larger budget.
    """
    common = {
        "metric": dp.correction_metric,
        "max_bootstrapped_demos": args.max_demos,
        "max_labeled_demos": args.max_labeled_demos,
    }
    if args.optimizer == "bootstrap":
        optimizer = dspy.BootstrapFewShot(**common)
        return optimizer.compile(program, trainset=trainset)
    if args.optimizer == "bootstrap-rs":
        optimizer = dspy.BootstrapFewShotWithRandomSearch(
            num_candidate_programs=args.num_candidates,
            num_threads=args.num_threads,
            **common,
        )
        return optimizer.compile(program, trainset=trainset, valset=valset)
    optimizer = dspy.MIPROv2(
        metric=dp.correction_metric,
        num_candidates=args.num_candidates,
        num_threads=args.num_threads,
        auto=None,
    )
    return optimizer.compile(
        program,
        trainset=trainset,
        valset=valset,
        num_trials=args.num_candidates,
        max_bootstrapped_demos=args.max_demos,
        max_labeled_demos=args.max_labeled_demos,
        requires_permission_to_run=False,
    )


def summarize(results: dict) -> str:
    lines = ["=" * 72, "DSPy prompt search", "=" * 72]
    baseline = results.get("baseline_handwritten")
    if baseline:
        lines.append(
            f"hand-written open prompt   strict {baseline['strict_accuracy']:.1%}  "
            f"lenient {baseline['lenient_accuracy']:.1%}  (n={baseline['n']}, "
            f"p50 {baseline.get('p50_latency_s', 0.0):.2f}s)"
        )
    for name, payload in results.get("programs", {}).items():
        zero = payload["zero_shot"]
        tuned = payload["optimized"]
        lines.append(
            f"{name:<26} zero-shot strict {zero['strict_accuracy']:.1%} -> "
            f"optimized strict {tuned['strict_accuracy']:.1%} "
            f"(lenient {tuned['lenient_accuracy']:.1%}, {payload['description']['n_demos']} demos, "
            f"p50 {tuned.get('p50_latency_s', 0.0):.2f}s)"
        )
        if payload.get("optimize_error"):
            lines.append(f"{'':<26} optimization stopped early: {payload['optimize_error']}")
    if results.get("selected_program"):
        lines.append(f"selected: {results['selected_program']} on the dev validation split")
    final = results.get("final_eval")
    if final:
        lines.append(
            f"held-out benchmark (n={final['sample_size']}, one pass, frozen prompt): "
            f"strict {final['strict_accuracy']:.1%}  lenient {final['lenient_accuracy']:.1%}"
        )
    if baseline and results.get("programs"):
        best = max(p["optimized"]["strict_accuracy"] for p in results["programs"].values())
        delta = best - baseline["strict_accuracy"]
        verdict = "beats" if delta > 0 else ("ties" if delta == 0 else "loses to")
        lines.append(
            f"verdict: the optimized prompt {verdict} the hand-written one on the dev "
            f"validation split by {delta:+.1%} (strict)"
        )
    lines.append(f"LM calls: {results.get('lm_calls')}   wall clock: {results.get('wall_clock_s', 0):.0f}s")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
