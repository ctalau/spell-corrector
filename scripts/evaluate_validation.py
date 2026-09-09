#!/usr/bin/env python3
"""Evaluate a checkpoint on the prepared validation parquet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spelling_reranker.config import load_train_config, model_config_from_mapping
from spelling_reranker.dataset import SpellingParquetDataset, make_collate
from spelling_reranker.inference import load_model_dir
from spelling_reranker.model import masked_cross_entropy, topk_accuracy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train_full.yaml")
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts" / "model")
    parser.add_argument("--split", choices=("validation", "train"), default="validation")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_train_config(args.config)
    model_cfg = model_config_from_mapping(cfg.get("model"))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model_dir(args.model, device=device)

    data_key = "validation" if args.split == "validation" else "train"
    ds = SpellingParquetDataset(
        cfg["data"][data_key],
        max_examples=args.max_examples,
        max_seq_len=model_cfg.max_seq_len,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=make_collate(model_cfg.max_seq_len))

    total_loss = 0.0
    total_top1 = 0.0
    total_top3 = 0.0
    n = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                batch["token_ids"],
                batch["attention_mask"],
                batch["typo_mask"],
                batch["candidate_masks"],
                batch["candidate_valid"],
            )
            loss = masked_cross_entropy(logits, batch["gold_index"])
            bsz = int(batch["gold_index"].size(0))
            total_loss += float(loss.item()) * bsz
            total_top1 += float(topk_accuracy(logits, batch["gold_index"], 1).item()) * bsz
            total_top3 += float(topk_accuracy(logits, batch["gold_index"], 3).item()) * bsz
            n += bsz

    out = {
        "split": args.split,
        "n": n,
        "loss": total_loss / max(1, n),
        "acc_top1": total_top1 / max(1, n),
        "acc_top3": total_top3 / max(1, n),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
