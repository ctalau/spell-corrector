"""CLI contract for scripts/dspy_prompt_search.py -- argument handling and failure modes.

No live LM and no llama-server: what is checked here is that the entry point
the pod's bootstrap calls parses the way the bootstrap expects, that the cost
bound is enforced *before* anything is spent, and that the two ways a run
realistically fails (dspy missing, server down) exit non-zero with a message
that says what to do.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "dspy_prompt_search_cli", ROOT / "scripts" / "dspy_prompt_search.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cli = _load_cli()


def _dev_example(i: int):
    from spelling_reranker.dev_set import DevExample

    return DevExample(
        example_id=f"dev-{i:04d}",
        sentence="Now I must buy it on the interenet .",
        context_before="Now I must buy it on the ",
        typo="interenet",
        context_after=" .",
        gold="internet",
        candidates=("internet", "interment"),
        gold_index=0,
        source="synthetic",
        corruption_type="transposition",
        edit_distance=1,
    )



def test_the_bootstrap_invocation_parses() -> None:
    """Exactly the command line documented for the pod's bootstrap."""
    args = cli.parse_args(
        [
            "--base-url", "http://127.0.0.1:8080/v1",
            "--model", "gemma-4-e2b-q4",
            "--output", "reports/gpu_llama/dspy",
        ]
    )
    assert args.base_url == "http://127.0.0.1:8080/v1"
    assert args.model == "gemma-4-e2b-q4"
    assert args.output == Path("reports/gpu_llama/dspy")
    assert args.final_eval is False  # the locked benchmark is opt-in, always


def test_final_eval_and_bea_dir_are_accepted() -> None:
    args = cli.parse_args(["--bea-dir", "data/bea60k", "--final-eval", "--n-final", "100"])
    assert args.final_eval is True
    assert args.bea_dir == Path("data/bea60k")
    assert args.n_final == 100


def test_budget_knobs_have_conservative_defaults() -> None:
    args = cli.parse_args([])
    assert args.optimizer == "bootstrap-rs"
    assert args.budget > 0
    assert args.max_demos >= 1
    assert args.num_threads == 1  # llama-server is started with -np 1
    assert args.seed == 1337


def test_unknown_program_name_is_rejected_by_argparse() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--programs", "not-a-program"])


def test_estimate_calls_grows_with_candidates_and_covers_every_phase() -> None:
    base = cli.parse_args(["--optimizer", "bootstrap", "--programs", "candidate_guided"])
    cheap = cli.estimate_calls(base, 50, 50)
    assert cheap == 50 + (50 + 50 + 50)  # baseline + zero-shot + bootstrap + rescore

    rs = cli.parse_args(["--optimizer", "bootstrap-rs", "--num-candidates", "4", "--programs", "candidate_guided"])
    assert cli.estimate_calls(rs, 50, 50) > cheap

    both = cli.parse_args(["--optimizer", "bootstrap"])
    assert cli.estimate_calls(both, 50, 50) > cheap  # two programs cost more than one

    no_baseline = cli.parse_args(["--optimizer", "bootstrap", "--programs", "candidate_guided", "--skip-baseline"])
    assert cli.estimate_calls(no_baseline, 50, 50) == cheap - 50

    with_final = cli.parse_args(
        ["--optimizer", "bootstrap", "--programs", "candidate_guided", "--final-eval", "--n-final", "100"]
    )
    assert cli.estimate_calls(with_final, 50, 50) == cheap + 100


def test_a_search_that_would_blow_the_budget_exits_before_any_lm_call(monkeypatch, capsys, tmp_path) -> None:
    """The budget is checked against the estimate before the LM is even built."""
    pytest.importorskip("dspy")
    from spelling_reranker.dev_set import DevSet

    monkeypatch.setattr(cli, "wait_for_server", lambda *a, **k: {})
    examples = [_dev_example(i) for i in range(20)]
    monkeypatch.setattr(
        cli, "load_or_build", lambda *a, **k: (DevSet(examples=examples, stats={"n": 20}), tmp_path / "dev.json")
    )

    def fail(*args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("an LM was created despite an impossible budget")

    monkeypatch.setattr(cli, "make_budgeted_lm", fail)
    code = cli.main(["--output", str(tmp_path), "--dev-size", "20", "--budget", "1"])
    assert code == 4
    assert "--budget is 1" in capsys.readouterr().err


def test_missing_dspy_is_reported_clearly(monkeypatch, capsys, tmp_path) -> None:
    def no_dspy():
        raise ImportError("DSPy is required for prompt optimization but is not installed.")

    monkeypatch.setattr(cli.dp, "require_dspy", no_dspy)
    assert cli.main(["--output", str(tmp_path)]) == 2
    assert "not installed" in capsys.readouterr().err


def test_an_unreachable_server_is_reported_clearly(monkeypatch, capsys, tmp_path) -> None:
    def down(*args, **kwargs):
        raise cli.LlamaServerError("llama-server not ready within 30s")

    monkeypatch.setattr(cli, "wait_for_server", down)
    assert cli.main(["--output", str(tmp_path), "--base-url", "http://127.0.0.1:9/v1"]) == 3
    err = capsys.readouterr().err
    assert "no llama-server at http://127.0.0.1:9" in err
    assert "--reasoning off" in err  # the fix, not just the symptom


def test_summarize_states_the_verdict_both_ways() -> None:
    results = {
        "baseline_handwritten": {"strict_accuracy": 0.87, "lenient_accuracy": 0.87, "n": 150},
        "programs": {
            "candidate_guided": {
                "zero_shot": {"strict_accuracy": 0.80, "lenient_accuracy": 0.84},
                "optimized": {"strict_accuracy": 0.84, "lenient_accuracy": 0.88},
                "description": {"n_demos": 3},
            }
        },
        "selected_program": "candidate_guided",
        "lm_calls": 512,
        "wall_clock_s": 900.0,
    }
    text = cli.summarize(results)
    assert "loses to" in text and "-3.0%" in text

    results["programs"]["candidate_guided"]["optimized"]["strict_accuracy"] = 0.91
    assert "beats" in cli.summarize(results)


def test_summarize_reports_a_final_benchmark_number_with_its_sample_size() -> None:
    text = cli.summarize(
        {
            "programs": {},
            "final_eval": {"sample_size": 100, "strict_accuracy": 0.88, "lenient_accuracy": 0.9},
            "lm_calls": 10,
            "wall_clock_s": 1.0,
        }
    )
    assert "n=100" in text and "88.0%" in text
