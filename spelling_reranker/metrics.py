"""Shared evaluation helpers."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import torch

from spelling_reranker.model import topk_accuracy


def accuracy_by_gold_index(
    logits: torch.Tensor,
    gold_index: torch.Tensor,
    n_candidates: int = 10,
) -> dict[int, float]:
    pred = logits.argmax(dim=-1)
    out: dict[int, float] = {}
    for idx in range(n_candidates):
        mask = gold_index == idx
        if int(mask.sum()) == 0:
            continue
        out[idx] = float((pred[mask] == gold_index[mask]).float().mean().item())
    return out


def summarize_batch(
    logits: torch.Tensor,
    gold_index: torch.Tensor,
    candidate_valid: torch.Tensor | None = None,
) -> dict[str, float]:
    loss = torch.nn.functional.cross_entropy(logits, gold_index).item()
    return {
        "loss": float(loss),
        "acc_top1": float(topk_accuracy(logits, gold_index, k=1, candidate_valid=candidate_valid).item()),
        "acc_top3": float(topk_accuracy(logits, gold_index, k=3, candidate_valid=candidate_valid).item()),
    }


def histogram_percentages(counts: Counter[str] | dict[str, int]) -> dict[str, float]:
    total = sum(counts.values()) or 1
    return {k: 100.0 * v / total for k, v in counts.items()}


def merge_counts(items: Iterable[str]) -> Counter[str]:
    return Counter(items)
