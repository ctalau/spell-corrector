#!/usr/bin/env python3
"""Evaluate a frozen selector checkpoint on named splits or a BEA-60K subset.

Does not run the full BEA benchmark unless the caller passes --split bea.
Training uses this path for the 1k monitoring gate; it is not invoked
automatically by the byte-model trainer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cache_frozen_features import bea_records, extract_records
from spelling_reranker.config import load_yaml
from spelling_reranker.device import describe_cuda, select_training_device
from spelling_reranker.frozen_encoder import (
    DEFAULT_HIDDEN_SIZE,
    FROZEN_CANDIDATE_SLOTS,
    FrozenEncoder,
    ScalarScaler,
    SelectorHead,
    assemble_selector_features,
    load_feature_shards,
    mask_invalid_logits,
    preferred_encoder_dtype,
    predict_index_from_logits,
)
from spelling_reranker.model import topk_accuracy


class FeatureBundleDataset(Dataset):
    def __init__(self, bundle: dict[str, np.ndarray], *, solvable_only: bool = False) -> None:
        n = int(bundle["x"].shape[0])
        if solvable_only:
            mask = (bundle["solvable"].astype(bool)) & (bundle["gold_index"] >= 0)
        else:
            mask = np.ones(n, dtype=bool)
        idx = np.flatnonzero(mask)
        self.x = bundle["x"][idx]
        self.t = bundle["t"][idx]
        self.c = bundle["c"][idx]
        self.scalars = bundle["scalars"][idx]
        self.valid = bundle["valid"][idx]
        self.gold = bundle["gold_index"][idx]
        self.failed = bundle["serialization_failed"][idx]
        self.solvable = bundle["solvable"][idx]
        self.example_id = bundle["example_id"][idx]
        self.ids = idx

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        return {
            "x": torch.from_numpy(np.asarray(self.x[index], dtype=np.float32)),
            "t": torch.from_numpy(np.asarray(self.t[index], dtype=np.float32)),
            "c": torch.from_numpy(np.asarray(self.c[index], dtype=np.float32)),
            "scalars": torch.from_numpy(np.asarray(self.scalars[index], dtype=np.float32)),
            "valid": torch.from_numpy(np.asarray(self.valid[index], dtype=np.int64)),
            "gold_index": torch.tensor(int(self.gold[index]), dtype=torch.int64),
            "serialization_failed": torch.tensor(int(self.failed[index]), dtype=torch.int64),
            "solvable": torch.tensor(int(self.solvable[index]), dtype=torch.int64),
            "example_id": str(self.example_id[index]),
            "row": int(self.ids[index]),
        }


def _collate(rows: list[dict]) -> dict:
    out: dict = {}
    for key in ("x", "t", "c", "scalars", "valid", "gold_index", "serialization_failed", "solvable"):
        out[key] = torch.stack([row[key] for row in rows], dim=0)
    out["example_id"] = [row["example_id"] for row in rows]
    out["row"] = [row["row"] for row in rows]
    return out


def load_head_checkpoint(path: Path, device: torch.device) -> tuple[SelectorHead, ScalarScaler, dict]:
    meta_path = path / "meta.json" if path.is_dir() else path.parent / "meta.json"
    weights = path / "head.safetensors" if path.is_dir() else path
    if path.is_dir() and not weights.is_file():
        for name in ("best.safetensors", "last.safetensors"):
            cand = path / name
            if cand.is_file():
                weights = cand
                break
    if not weights.is_file():
        raise FileNotFoundError(f"no selector weights in {path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    scaler = ScalarScaler.from_dict(meta.get("scaler"))
    head = SelectorHead(
        str(meta.get("arm", "mlp")),
        hidden_size=int(meta.get("hidden_size", DEFAULT_HIDDEN_SIZE)),
        n_candidates=int(meta.get("n_candidates", FROZEN_CANDIDATE_SLOTS)),
        dropout=0.0,
    )
    state = load_file(str(weights), device=str(device))
    head.load_state_dict(state)
    head.to(device)
    head.eval()
    return head, scaler, meta


def score_bundle(
    head: SelectorHead | None,
    bundle: dict[str, np.ndarray],
    *,
    device: torch.device,
    scaler: ScalarScaler | None = None,
    batch_size: int = 512,
    hunspell_baseline: bool = False,
) -> dict:
    ds = FeatureBundleDataset(bundle, solvable_only=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    n = len(ds)
    pred_all = np.zeros(n, dtype=np.int64)
    gold_all = np.zeros(n, dtype=np.int64)
    solvable_all = np.zeros(n, dtype=np.uint8)
    failed_all = np.zeros(n, dtype=np.uint8)
    hunspell0 = np.zeros(n, dtype=np.uint8)
    offset = 0
    total_loss = 0.0
    loss_n = 0
    if head is not None:
        head.eval()
    with torch.no_grad():
        for batch in loader:
            bsz = int(batch["gold_index"].size(0))
            gold = batch["gold_index"]
            valid = batch["valid"]
            failed = batch["serialization_failed"]
            if hunspell_baseline or head is None:
                pred = torch.zeros(bsz, dtype=torch.int64)
                logits = None
            else:
                features = assemble_selector_features(
                    batch["x"].to(device),
                    batch["t"].to(device),
                    batch["c"].to(device),
                    batch["scalars"].to(device),
                    include_encoder=head.uses_encoder_features(),
                    scaler=scaler,
                )
                logits = mask_invalid_logits(head(features), valid.to(device))
                pred = predict_index_from_logits(logits, valid.to(device), failed.to(device)).cpu()
                solvable_mask = (gold >= 0) & (batch["solvable"] > 0) & (failed == 0)
                if bool(solvable_mask.any()) and logits is not None:
                    loss = torch.nn.functional.cross_entropy(logits[solvable_mask], gold.to(device)[solvable_mask])
                    total_loss += float(loss.item()) * int(solvable_mask.sum())
                    loss_n += int(solvable_mask.sum())
            pred_all[offset : offset + bsz] = pred.numpy()
            gold_all[offset : offset + bsz] = gold.numpy()
            solvable_all[offset : offset + bsz] = batch["solvable"].numpy().astype(np.uint8)
            failed_all[offset : offset + bsz] = failed.numpy().astype(np.uint8)
            hunspell0[offset : offset + bsz] = (gold.numpy() == 0).astype(np.uint8)
            offset += bsz

    overall_ok = (solvable_all == 1) & (pred_all == gold_all) & (gold_all >= 0)
    # Serialization failures stay in the overall denominator and count as
    # candidate-0 predictions (already applied). Unsolvable rows are misses.
    n_overall = n
    n_cond = int(((solvable_all == 1) & (gold_all >= 0)).sum())
    cond_ok = int((((solvable_all == 1) & (gold_all >= 0) & (pred_all == gold_all)).sum()))
    hunspell_overall = int(hunspell0.sum()) / n_overall if n_overall else 0.0
    hunspell_cond = (
        int(((solvable_all == 1) & (gold_all == 0)).sum()) / n_cond if n_cond else 0.0
    )
    retention_d = int(((solvable_all == 1) & (gold_all == 0)).sum())
    retention_n = int(((solvable_all == 1) & (gold_all == 0) & (pred_all == gold_all)).sum())
    rescue_d = int(((solvable_all == 1) & (gold_all > 0)).sum())
    rescue_n = int(((solvable_all == 1) & (gold_all > 0) & (pred_all == gold_all)).sum())
    top3 = 0.0
    if n_cond and head is not None and not hunspell_baseline:
        # Recompute top-3 on solvable rows from stored logits would need another
        # pass; approximate via a second scoring pass below if needed. Filled by
        # score_top3 when the caller wants it.
        top3 = _top3(head, bundle, device, scaler, batch_size)

    ledger = []
    ids = ds.example_id
    for i in range(n):
        ledger.append(
            {
                "example_id": str(ids[i]),
                "gold_index": int(gold_all[i]),
                "pred_index": int(pred_all[i]),
                "correct": bool(overall_ok[i]),
                "solvable": bool(solvable_all[i]),
                "serialization_failed": bool(failed_all[i]),
                "hunspell_top1_ok": bool(hunspell0[i]),
            }
        )
    return {
        "n": n_overall,
        "n_conditional": n_cond,
        "coverage": n_cond / n_overall if n_overall else 0.0,
        "overall_accuracy": float(overall_ok.sum()) / n_overall if n_overall else 0.0,
        "conditional_accuracy": cond_ok / n_cond if n_cond else 0.0,
        "conditional_top3": top3,
        "hunspell_top1_overall": hunspell_overall,
        "hunspell_top1_conditional": hunspell_cond,
        "retention": retention_n / retention_d if retention_d else None,
        "rescue": rescue_n / rescue_d if rescue_d else None,
        "serialization_failures": int(failed_all.sum()),
        "loss": total_loss / max(1, loss_n) if loss_n else None,
        "ledger": ledger,
    }


def _top3(
    head: SelectorHead,
    bundle: dict[str, np.ndarray],
    device: torch.device,
    scaler: ScalarScaler | None,
    batch_size: int,
) -> float:
    ds = FeatureBundleDataset(bundle, solvable_only=True)
    if len(ds) == 0:
        return 0.0
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    total = 0.0
    n = 0
    head.eval()
    with torch.no_grad():
        for batch in loader:
            features = assemble_selector_features(
                batch["x"].to(device),
                batch["t"].to(device),
                batch["c"].to(device),
                batch["scalars"].to(device),
                include_encoder=head.uses_encoder_features(),
                scaler=scaler,
            )
            logits = mask_invalid_logits(head(features), batch["valid"].to(device))
            acc = topk_accuracy(logits, batch["gold_index"].to(device), k=3, candidate_valid=batch["valid"].to(device))
            bsz = int(batch["gold_index"].size(0))
            total += float(acc.item()) * bsz
            n += bsz
    return total / max(1, n)


def evaluate_bea_limit(
    head: SelectorHead | None,
    *,
    config: dict,
    bea_dir: Path,
    limit: int,
    device: torch.device,
    scaler: ScalarScaler | None,
    encoder: FrozenEncoder | None = None,
    cache_dir: Path | None = None,
    batch_size: int = 512,
    hunspell_baseline: bool = False,
) -> dict:
    if cache_dir is not None and (Path(cache_dir) / "bea").is_dir():
        bundle = load_feature_shards(cache_dir, "bea")
        if int(bundle["x"].shape[0]) >= int(limit):
            sliced = {k: v[: int(limit)] for k, v in bundle.items()}
            return score_bundle(
                head, sliced, device=device, scaler=scaler, hunspell_baseline=hunspell_baseline
            )
    if encoder is None:
        raise SystemExit("BEA-1k features are not cached; pass a frozen encoder or run cache --bea-limit")
    recs = bea_records(bea_dir, limit)
    hidden = int(getattr(getattr(encoder.encoder, "config", None), "hidden_size", DEFAULT_HIDDEN_SIZE))
    extract_bs = int(config.get("encoder", {}).get("extract_batch_size", 16))
    bundle = extract_records(recs, encoder, batch_size=extract_bs, hidden=hidden)
    return score_bundle(head, bundle, device=device, scaler=scaler, hunspell_baseline=hunspell_baseline)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train_frozen_modernbert.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--arm", choices=("scalar", "linear", "mlp", "hunspell"), default=None)
    parser.add_argument("--split", default="d-pair", help="d-pair, train, validation, bea")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--bea-dir", type=Path, default=ROOT / "data" / "bea60k")
    parser.add_argument("--bea-limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.allow_cpu:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        device = select_training_device(args.device)
    print(f"device={device}{describe_cuda(device)}")
    cache_dir = Path(args.cache_dir or cfg.get("cache", {}).get("dir", "artifacts/frozen_cache"))
    split = args.split.replace("_", "-").lower()
    split_dir = {"d-pair": "dpair", "dpair": "dpair", "train": "train", "validation": "dpair", "bea": "bea"}[split]

    hunspell_baseline = args.arm == "hunspell" or args.checkpoint is None and args.arm is None
    head = None
    scaler = ScalarScaler()
    meta: dict = {}
    if args.checkpoint is not None and args.arm != "hunspell":
        head, scaler, meta = load_head_checkpoint(args.checkpoint, device)
        hunspell_baseline = False
    elif args.arm == "hunspell" or args.checkpoint is None:
        hunspell_baseline = True

    if split_dir == "bea" or args.bea_limit:
        limit = int(args.bea_limit or cfg.get("training", {}).get("bea_limit", 1000))
        encoder = None
        if not (cache_dir / "bea").is_dir():
            encoder = FrozenEncoder(
                model_id=cfg["encoder"]["model_id"],
                revision=cfg["encoder"]["revision"],
                device=device,
                dtype=preferred_encoder_dtype(device),
                max_length=int(cfg["encoder"].get("max_length", 512)),
            )
        metrics = evaluate_bea_limit(
            head,
            config=cfg,
            bea_dir=args.bea_dir,
            limit=limit,
            device=device,
            scaler=scaler,
            encoder=encoder,
            cache_dir=cache_dir,
            batch_size=args.batch_size,
            hunspell_baseline=hunspell_baseline,
        )
        metrics["split"] = f"bea-{limit}"
    else:
        bundle = load_feature_shards(cache_dir, split_dir)
        metrics = score_bundle(
            head,
            bundle,
            device=device,
            scaler=scaler,
            batch_size=args.batch_size,
            hunspell_baseline=hunspell_baseline,
        )
        metrics["split"] = split

    metrics["arm"] = args.arm or meta.get("arm") or ("hunspell" if hunspell_baseline else None)
    metrics["checkpoint"] = str(args.checkpoint) if args.checkpoint else None
    out_dir = args.output or Path(cfg.get("output", {}).get("dir", "artifacts/frozen")) / "eval" / str(metrics["split"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = metrics.pop("ledger")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in ledger:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
