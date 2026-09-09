"""Training must fail fast when CUDA is missing; never silently use CPU."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from spelling_reranker import device as device_mod
from spelling_reranker.device import (
    cuda_required_message,
    describe_cuda,
    select_training_device,
)

ROOT = Path(__file__).resolve().parents[1]


def test_select_training_device_refuses_silent_cpu_fallback(monkeypatch) -> None:
    monkeypatch.setattr(device_mod.torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as ei:
        select_training_device(None)
    assert "CUDA is not available" in str(ei.value)
    assert "refusing to run GPU training configs on CPU" in str(ei.value)


def test_select_training_device_refuses_explicit_cuda_without_gpu(monkeypatch) -> None:
    monkeypatch.setattr(device_mod.torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as ei:
        select_training_device("cuda")
    assert "CUDA is not available" in str(ei.value)


def test_select_training_device_allows_explicit_cpu(monkeypatch) -> None:
    monkeypatch.setattr(device_mod.torch.cuda, "is_available", lambda: False)
    device = select_training_device("cpu")
    assert device.type == "cpu"


def test_cuda_required_message_includes_torch_cuda_build() -> None:
    msg = cuda_required_message()
    assert "torch.version.cuda" in msg
    assert str(torch.version.cuda) in msg


def test_describe_cuda_empty_on_cpu() -> None:
    assert describe_cuda(torch.device("cpu")) == ""


def test_setup_sh_installs_hunspell_into_python_and_requires_cuda() -> None:
    text = (ROOT / "scripts/runpod/setup.sh").read_text()
    assert '"$PYTHON" -m pip install' in text
    assert "hunspell==0.5.5" in text
    assert "--no-build-isolation" in text
    assert "import hunspell" in text
    assert 'die "import hunspell' in text
    assert "torch.cuda.is_available()" in text
    assert 'die "CUDA not available"' in text


def test_run_experiment_sh_requires_cuda_before_data_build() -> None:
    text = (ROOT / "scripts/runpod/run_experiment.sh").read_text()
    cuda_at = text.index("assert torch.cuda.is_available()")
    data_at = text.index("build_training_data.py")
    train_at = text.index("scripts/train.py")
    assert cuda_at < data_at < train_at


def test_bootstrap_sh_fails_status_when_cuda_missing() -> None:
    text = (ROOT / "scripts/runpod/bootstrap.sh").read_text()
    setup_at = text.index("scripts/runpod/setup.sh")
    cuda_at = text.index('fail "CUDA not available"')
    experiment_at = text.index("scripts/runpod/run_experiment.sh")
    assert setup_at < cuda_at < experiment_at


def test_preflight_and_train_do_not_silently_fall_back_to_cpu() -> None:
    for rel in ("scripts/preflight.py", "scripts/train.py"):
        text = (ROOT / rel).read_text()
        assert "select_training_device" in text
        assert 'if torch.cuda.is_available() else "cpu"' not in text
