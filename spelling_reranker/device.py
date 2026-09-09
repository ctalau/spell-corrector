"""Training device selection. GPU configs must not silently fall back to CPU."""

from __future__ import annotations

import torch


def cuda_required_message() -> str:
    return (
        "CUDA is not available; refusing to run GPU training configs on CPU. "
        f"torch={torch.__version__} torch.version.cuda={torch.version.cuda}. "
        "This is usually a NVIDIA driver vs PyTorch CUDA build mismatch "
        "(for example host driver CUDA 12.4 with a cu128 image)."
    )


def select_training_device(requested: str | None = None) -> torch.device:
    """Return the training device.

    Defaults to CUDA. CPU is allowed only when explicitly requested
    (``requested='cpu'``). A CUDA request with no GPU raises SystemExit
    instead of falling back to CPU.
    """
    device = torch.device(requested or "cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(cuda_required_message())
    return device


def describe_cuda(device: torch.device) -> str:
    if device.type != "cuda":
        return ""
    return (
        f" torch.version.cuda={torch.version.cuda}"
        f" gpu={torch.cuda.get_device_name(0)}"
    )
