"""5. MODEL SHAPE  6. PARAMETER COUNT  7. LOSS backward."""

from __future__ import annotations

from pathlib import Path

import torch

from spelling_reranker.byte_encoding import N_CANDIDATE_SLOTS
from spelling_reranker.config import load_yaml, model_config_from_mapping
from spelling_reranker.dataset import collate_examples
from spelling_reranker.model import ByteSpellingReranker, ModelConfig, count_parameters, masked_cross_entropy
from spelling_reranker.serialization import serialize_example


def _batch(n: int = 3):
    examples = []
    for i in range(n):
        examples.append(
            serialize_example(
                f"Context {i} left ",
                "quik",
                " right.",
                ["quick", "quirk", "quack", "quit"] + [None] * (N_CANDIDATE_SLOTS - 4),
                gold_index=0,
                max_seq_len=96,
            )
        )
    return collate_examples(examples)


def test_logits_shape_is_batch_by_candidate_slots() -> None:
    model = ByteSpellingReranker(ModelConfig(n_layers=2, max_seq_len=96))
    batch = _batch(4)
    logits = model(
        batch["token_ids"],
        batch["attention_mask"],
        batch["typo_mask"],
        batch["candidate_masks"],
        batch["candidate_valid"],
    )
    assert logits.shape == (4, N_CANDIDATE_SLOTS)


def test_config_parameter_counts_match_their_names() -> None:
    """Each shipped model config must be the size its filename claims."""
    expected = {
        "configs/model_28m.yaml": (27_000_000, 29_000_000),
        "configs/model_87m.yaml": (85_000_000, 89_000_000),
    }
    root = Path(__file__).resolve().parents[1]
    for rel, (low, high) in expected.items():
        cfg = model_config_from_mapping(load_yaml(root / rel))
        n = count_parameters(ByteSpellingReranker(cfg))
        print(f"{rel}: trainable parameters {n:,}")
        assert low <= n <= high, f"{rel} has {n} params, expected {low}-{high}"


def test_candidate_pooling_matches_the_naive_formulation() -> None:
    """The bmm pooling must equal the elementwise mask-and-sum it replaced."""
    torch.manual_seed(0)
    model = ByteSpellingReranker(ModelConfig(n_layers=2, max_seq_len=96))
    batch = _batch(3)
    hidden = model.encode(batch["token_ids"], batch["attention_mask"])
    masks = batch["candidate_masks"]
    weights = masks.to(hidden.dtype)
    fast = torch.bmm(weights, hidden) / weights.sum(dim=2).clamp(min=1e-6).unsqueeze(-1)
    naive_w = weights.unsqueeze(-1)
    naive = (hidden.unsqueeze(1) * naive_w).sum(dim=2) / naive_w.sum(dim=2).clamp(min=1e-6)
    assert torch.allclose(fast, naive, atol=1e-5)


def test_cross_entropy_forward_backward() -> None:
    model = ByteSpellingReranker(ModelConfig(n_layers=2, max_seq_len=96))
    batch = _batch(2)
    logits = model(
        batch["token_ids"],
        batch["attention_mask"],
        batch["typo_mask"],
        batch["candidate_masks"],
        batch["candidate_valid"],
    )
    loss = masked_cross_entropy(logits, batch["gold_index"])
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "expected gradients"
    assert all(torch.isfinite(g).all() for g in grads)
