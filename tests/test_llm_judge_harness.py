"""Argparse / phase-planning tests for the LLM-judge harness.

Loaded from file so this does not import torch, transformers, or Hunspell
and never downloads a model.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_LLM_JUDGE_PATH = Path(__file__).resolve().parents[1] / "spelling_reranker" / "llm_judge.py"


def _load_llm_judge():
    spec = importlib.util.spec_from_file_location("llm_judge_noload", _LLM_JUDGE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


lj = _load_llm_judge()

_BASE = ["--model-id", "google/gemma-4-E2B-it", "--model-name", "gemma-4-e2b", "--output", "/tmp/out"]

# Hunspell-flagged + has suggestions at 0 and 2; 1 is not flagged; 3 is flagged but empty pool.
_ERRORS = [
    {"hunspell_flagged": True, "candidates": ["store"]},
    {"hunspell_flagged": False, "candidates": []},
    {"hunspell_flagged": True, "candidates": ["their", "there"]},
    {"hunspell_flagged": True, "candidates": []},
    {"hunspell_flagged": True, "candidates": ["receive"]},
]


def test_eligible_indices_require_flagged_and_suggestions():
    assert lj.eligible_indices(_ERRORS) == [0, 2, 4]


def test_full_selects_all_eligible_indices():
    args = lj.parse_llm_judge_args([*_BASE, "--full"])
    assert args.full is True
    order = lj.shuffle_indices(lj.eligible_indices(_ERRORS), args.seed)
    assert sorted(order) == [0, 2, 4]
    phases = lj.plan_phases(args, order)
    assert [p.name for p in phases] == ["full_bea60k"]
    full = phases[0]
    assert sorted(full.indices) == [0, 2, 4]
    assert full.indices == tuple(order)
    assert full.max_examples is None
    assert full.wrap is False
    assert full.time_budget_seconds is None
    assert full.predictions_stem == "full"


def test_full_with_one_hour_budget_still_uses_all_eligible():
    args = lj.parse_llm_judge_args([*_BASE, "--full", "--time-budget-seconds", "3600"])
    order = lj.shuffle_indices(lj.eligible_indices(_ERRORS), seed=1337)
    phases = lj.plan_phases(args, order)
    assert len(phases) == 1
    full = phases[0]
    assert full.name == "full_bea60k"
    assert sorted(full.indices) == [0, 2, 4]
    assert full.time_budget_seconds == 3600.0
    assert full.wrap is False
    assert full.max_examples is None


def test_skip_sample_with_3600s_is_single_timed_phase():
    args = lj.parse_llm_judge_args([*_BASE, "--skip-sample", "--time-budget-seconds", "3600"])
    order = [0, 2, 4]
    phases = lj.plan_phases(args, order)
    assert [p.name for p in phases] == ["timed"]
    timed = phases[0]
    assert timed.indices == (0, 2, 4)
    assert timed.time_budget_seconds == 3600.0
    assert timed.max_examples is None
    assert timed.wrap is True
    assert lj.resolve_time_budget(args) == 3600.0


def test_default_plans_sample_then_five_minute_timed():
    args = lj.parse_llm_judge_args(_BASE)
    assert args.full is False
    assert args.skip_sample is False
    assert args.time_budget_seconds is None
    assert lj.resolve_time_budget(args) == 300.0
    order = list(range(250))
    phases = lj.plan_phases(args, order)
    assert [p.name for p in phases] == ["sample_100", "timed"]
    assert phases[0].indices == tuple(order[:100])
    assert phases[0].max_examples == 100
    assert phases[0].time_budget_seconds is None
    assert phases[1].indices == tuple(order)
    assert phases[1].time_budget_seconds == 300.0
    assert phases[1].wrap is True


def test_full_also_sample_keeps_sample_then_full():
    args = lj.parse_llm_judge_args([*_BASE, "--full", "--also-sample", "--n-samples", "2"])
    order = [4, 0, 2]
    phases = lj.plan_phases(args, order)
    assert [p.name for p in phases] == ["sample_100", "full_bea60k"]
    assert phases[0].indices == (4, 0)
    assert phases[1].indices == (4, 0, 2)


def test_zero_time_budget_is_unbounded():
    args = lj.parse_llm_judge_args([*_BASE, "--full", "--time-budget-seconds", "0"])
    assert lj.resolve_time_budget(args) is None
    phases = lj.plan_phases(args, [0, 2, 4])
    assert phases[0].name == "full_bea60k"
    assert phases[0].time_budget_seconds is None
    assert phases[0].wrap is False


def test_unbounded_timed_does_not_wrap():
    args = lj.parse_llm_judge_args([*_BASE, "--skip-sample", "--time-budget-seconds", "0"])
    phases = lj.plan_phases(args, [0, 2, 4])
    assert phases[0].name == "timed"
    assert phases[0].wrap is False
    assert phases[0].time_budget_seconds is None


def test_sample_uses_all_eligible_when_n_samples_exceeds_pool():
    args = lj.parse_llm_judge_args([*_BASE, "--n-samples", "100"])
    phases = lj.plan_phases(args, [0, 2])
    assert phases[0].indices == (0, 2)
    assert phases[0].max_examples == 2


def test_llm_judge_setup_requires_torch25_cu124_and_gemma4():
    """Gemma-4 load imports DTensor from torch.distributed.tensor (torch>=2.5).

    cu128 wheels fall back to CPU on Community hosts with a CUDA 12.4 driver.
    Pinning transformers>=5.5,<5.15 on image torch 2.4 is not enough: 5.14.1
    still hard-imports DTensor.
    """
    root = Path(__file__).resolve().parents[1]
    setup = (root / "scripts/runpod/setup_llm_judge.sh").read_text()
    assert '"torch>=2.5.1"' in setup
    assert "https://download.pytorch.org/whl/cu124" in setup
    assert "from torch.distributed.tensor import DTensor" in setup
    assert "transformers.models.gemma4" in setup
    assert '"transformers>=5.5"' in setup
    assert '"transformers>=5.5,<5.15"' not in setup
    assert "transformers>=4.57" not in setup
    assert "is_torch_available" in setup
    assert "AutoModelForMultimodalLM" in setup
    assert "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" in setup
    bootstrap = (root / "scripts/runpod/bootstrap_llm_judge.sh").read_text()
    assert "torch>=2.5" in bootstrap
    assert "cu124" in bootstrap
    launch = (root / "scripts/runpod/launch.py").read_text()
    assert (
        'LLM_JUDGE_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"'
        in launch
    )
    report = (root / "reports/EXPERIMENT_LLM_JUDGE.md").read_text()
    assert "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" in report
    assert "torch>=2.5.1+cu124" in report
    assert "`transformers>=5.5`" in report
    assert "rwfef0kegboxvk" in report
