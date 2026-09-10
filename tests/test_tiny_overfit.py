"""8. OVERFIT TEST — model can substantially overfit a tiny fixed set."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from spelling_reranker.dataset import collate_examples
from spelling_reranker.model import ByteSpellingReranker, ModelConfig, masked_cross_entropy
from spelling_reranker.seed import seed_everything
from spelling_reranker.serialization import serialize_example


WORDS = [
    ("quik", "quick", ["quick", "quirk", "quack", "quit"]),
    ("teh", "the", ["the", "teh", "ten", "tea"]),
    ("recieve", "receive", ["receive", "relieve", "revive", "recipe"]),
    ("seperate", "separate", ["separate", "desperate", "temperate", "serrate"]),
    ("definately", "definitely", ["definitely", "defiantly", "definably", "definity"]),
    ("occured", "occurred", ["occurred", "occured", "occupied", "occur"]),
    ("untill", "until", ["until", "untile", "unfill", "unfit"]),
    ("wich", "which", ["which", "witch", "wish", "with"]),
]


def _tiny_set(n: int = 128):
    examples = []
    for i in range(n):
        typo, gold, cands = WORDS[i % len(WORDS)]
        extras = [f"alt{i % 17}", f"zzz{i % 9}"]
        full = list(cands) + extras
        full = full[:10]
        gold_index = full.index(gold)
        examples.append(
            serialize_example(
                f"Prefix number {i} about ",
                typo,
                f" and suffix {i}.",
                full,
                gold_index=gold_index,
                max_seq_len=96,
            )
        )
    return examples


@pytest.mark.slow
def test_tiny_overfit_loss_drops() -> None:
    seed_everything(1337)
    examples = _tiny_set(128)
    loader = DataLoader(examples, batch_size=16, shuffle=True, collate_fn=collate_examples)
    # Use the real width but fewer layers so the CPU test stays cheap.
    # The full 8-layer 28M model is covered by the parameter-count test.
    cfg = ModelConfig(n_layers=2, max_seq_len=96, dropout=0.0, score_dropout=0.0)
    model = ByteSpellingReranker(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    def _run_epoch() -> float:
        model.train()
        total = 0.0
        n = 0
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            logits = model(
                batch["token_ids"],
                batch["attention_mask"],
                batch["typo_mask"],
                batch["candidate_masks"],
                batch["candidate_valid"],
            )
            loss = masked_cross_entropy(logits, batch["gold_index"])
            loss.backward()
            opt.step()
            total += float(loss.item()) * int(batch["gold_index"].size(0))
            n += int(batch["gold_index"].size(0))
        return total / max(1, n)

    first = _run_epoch()
    last = first
    for _ in range(12):
        last = _run_epoch()
    print(f"overfit loss {first:.4f} -> {last:.4f}")
    assert last < first * 0.5 or last < 0.35, f"loss did not drop enough: {first} -> {last}"
