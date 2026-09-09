"""5. MODEL SHAPE  6. PARAMETER COUNT  7. LOSS backward."""

from __future__ import annotations

import torch

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
                ["quick", "quirk", "quack", "quit"] + [None] * 6,
                gold_index=0,
                max_seq_len=96,
            )
        )
    return collate_examples(examples)


def test_logits_shape_is_batch_by_10() -> None:
    model = ByteSpellingReranker(ModelConfig(n_layers=2, max_seq_len=96))
    batch = _batch(4)
    logits = model(
        batch["token_ids"],
        batch["attention_mask"],
        batch["typo_mask"],
        batch["candidate_masks"],
        batch["candidate_valid"],
    )
    assert logits.shape == (4, 10)


def test_parameter_count_approximately_28m() -> None:
    model = ByteSpellingReranker(ModelConfig())
    n = count_parameters(model)
    print(f"trainable parameters: {n:,}")
    assert n <= 29_000_000, f"model has {n} params, exceeds 29M"
    assert 27_000_000 <= n <= 28_500_000, f"model has {n} params, expected ~28M"


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
