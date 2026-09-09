#!/usr/bin/env python3
"""Exercise every training config on the real device before the expensive work.

A config error does not surface at construction time. A vocab_size one smaller
than the byte vocabulary, for instance, fails as an asynchronous device-side
gather assert on the first batch that happens to contain the offending id --
and only on the code path that produces it, which for the masked-byte
objective means the full train and not the sanity train.

That is a 20+ minute data build thrown away for a one-line config mistake, so
this runs first: build each config on the target device and take a real
optimizer step with every code path enabled, including byte masking.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spelling_reranker.byte_encoding import N_CANDIDATE_SLOTS, VOCAB_SIZE
from spelling_reranker.config import load_train_config, model_config_from_mapping
from spelling_reranker.dataset import collate_examples
from spelling_reranker.model import (
    ByteSpellingReranker,
    apply_byte_masking,
    count_parameters,
    masked_byte_loss,
    masked_cross_entropy,
)
from spelling_reranker.serialization import serialize_example


def _batch(n: int, max_seq_len: int):
    examples = []
    for i in range(n):
        cands = [f"cand{j}" for j in range(N_CANDIDATE_SLOTS)]
        examples.append(
            serialize_example(
                f"some left context number {i} here ",
                "teh",
                " and some right context after it",
                cands,
                gold_index=i % N_CANDIDATE_SLOTS,
                max_seq_len=max_seq_len,
            )
        )
    return collate_examples(examples)


def check(config_path: Path, device: torch.device, amp_dtype) -> None:
    cfg = load_train_config(config_path)
    train_cfg = cfg.get("training", {})
    model_cfg = model_config_from_mapping(cfg.get("model"))

    if model_cfg.vocab_size < VOCAB_SIZE:
        raise SystemExit(
            f"{config_path}: vocab_size={model_cfg.vocab_size} < {VOCAB_SIZE}"
        )

    model = ByteSpellingReranker(model_cfg).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    batch = {k: v.to(device) for k, v in _batch(8, model_cfg.max_seq_len).items()}

    # Always exercise the masking path, whatever the config's weight, so a
    # vocabulary that cannot represent MASK is caught here rather than later.
    masked_ids, labels = apply_byte_masking(
        batch["token_ids"], batch["attention_mask"],
        batch["typo_mask"], batch["candidate_masks"],
        probability=max(0.5, float(train_cfg.get("mlm_probability", 0.12))),
    )
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        logits, hidden = model(
            masked_ids, batch["attention_mask"], batch["typo_mask"],
            batch["candidate_masks"], batch["candidate_valid"], return_hidden=True,
        )
        loss = masked_cross_entropy(logits, batch["gold_index"])
        if model.mlm_head is not None:
            loss = loss + 0.2 * masked_byte_loss(hidden, model.mlm_head, labels)
    loss.backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()  # surface async device asserts here

    if not torch.isfinite(loss):
        raise SystemExit(f"{config_path}: non-finite loss {loss.item()}")
    print(
        f"  {config_path.name}: {count_parameters(model):,} params, "
        f"vocab {model_cfg.vocab_size}, seq {model_cfg.max_seq_len}, "
        f"loss {loss.item():.4f} OK"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="*", type=Path)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    configs = args.configs or sorted((ROOT / "configs").glob("train_*.yaml"))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else None
    )
    print(f"preflight on {device} (bf16={amp_dtype is not None})")
    for path in configs:
        check(path, device, amp_dtype)
    print("preflight OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
