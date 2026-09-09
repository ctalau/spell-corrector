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
from spelling_reranker.device import describe_cuda, select_training_device
from spelling_reranker.model import (
    ByteSpellingReranker,
    apply_byte_masking,
    count_parameters,
    masked_byte_loss,
    masked_cross_entropy,
)
from spelling_reranker.serialization import serialize_example


def _worst_case_batch(n: int, max_seq_len: int):
    """A full microbatch of maximum-length sequences.

    Peak memory is set by the longest batch, not the average one. With length
    bucketing most batches are short, so an undersized microbatch can train for
    tens of steps before the first max-length batch OOMs it in backward -- which
    is exactly how a 256 microbatch died at step 18, 25 minutes into a run.
    """
    filler = "wordy context " * (max_seq_len // 7)
    examples = []
    for i in range(n):
        cands = [f"candidate{j}" for j in range(N_CANDIDATE_SLOTS)]
        examples.append(
            serialize_example(
                filler,
                "teh",
                filler,
                cands,
                gold_index=i % N_CANDIDATE_SLOTS,
                max_seq_len=max_seq_len,
            )
        )
    batch = collate_examples(examples)
    assert batch["token_ids"].shape[1] == max_seq_len, "worst-case batch is not full length"
    return batch


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
    microbatch = int(train_cfg.get("microbatch", 8))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    batch = {
        k: v.to(device)
        for k, v in _worst_case_batch(microbatch, model_cfg.max_seq_len).items()
    }

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
    peak = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
    total = (
        torch.cuda.get_device_properties(0).total_memory / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    note = ""
    if device.type == "cuda":
        note = f", peak {peak:.1f}/{total:.1f} GiB"
        if peak > 0.85 * total:
            raise SystemExit(
                f"{config_path}: worst-case batch peaks at {peak:.1f} GiB of "
                f"{total:.1f} GiB. Lower microbatch (and raise grad_accumulation "
                "to keep the effective batch)."
            )
    print(
        f"  {config_path.name}: {count_parameters(model):,} params, "
        f"vocab {model_cfg.vocab_size}, seq {model_cfg.max_seq_len}, "
        f"microbatch {microbatch}, loss {loss.item():.4f} OK{note}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="*", type=Path)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    configs = args.configs or sorted((ROOT / "configs").glob("train_*.yaml"))
    device = select_training_device(args.device)
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else None
    )
    print(f"preflight on {device} (bf16={amp_dtype is not None}){describe_cuda(device)}")
    for path in configs:
        check(path, device, amp_dtype)
    print("preflight OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
