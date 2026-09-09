#!/usr/bin/env python3
"""Train a frozen-encoder selector head (H1 scalar / H2 linear / H3 MLP).

Head-only AdamW. Early-stop on D-pair conditional accuracy (patience 2).
Checkpoints every N steps and on a wall-clock interval. The 30-minute BEA-1k
user gate stops an arm when overall and conditional accuracy both fail to
improve by more than 1 percentage point versus the best prior 30-minute
checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from spelling_reranker.candidates import FROZEN_CANDIDATE_SLOTS
from spelling_reranker.config import load_yaml
from spelling_reranker.device import describe_cuda, select_training_device
from spelling_reranker.frozen_encoder import (
    DEFAULT_HIDDEN_SIZE,
    ScalarScaler,
    SelectorHead,
    assemble_selector_features,
    fit_scalar_scaler,
    load_feature_shards,
    mask_invalid_logits,
)
from spelling_reranker.model import count_parameters, topk_accuracy
from spelling_reranker.seed import DEFAULT_SEED, seed_everything

from evaluate_frozen_selector import evaluate_bea_limit, score_bundle


class SolvableFeatureDataset(Dataset):
    def __init__(self, bundle: dict[str, np.ndarray]) -> None:
        mask = (bundle["solvable"].astype(bool)) & (bundle["gold_index"] >= 0)
        mask = mask & (bundle["serialization_failed"] == 0)
        idx = np.flatnonzero(mask)
        self.x = np.asarray(bundle["x"][idx], dtype=np.float32)
        self.t = np.asarray(bundle["t"][idx], dtype=np.float32)
        self.c = np.asarray(bundle["c"][idx], dtype=np.float32)
        self.scalars = np.asarray(bundle["scalars"][idx], dtype=np.float32)
        self.valid = np.asarray(bundle["valid"][idx], dtype=np.int64)
        self.gold = np.asarray(bundle["gold_index"][idx], dtype=np.int64)

    def __len__(self) -> int:
        return int(self.gold.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.x[index]),
            "t": torch.from_numpy(self.t[index]),
            "c": torch.from_numpy(self.c[index]),
            "scalars": torch.from_numpy(self.scalars[index]),
            "valid": torch.from_numpy(self.valid[index]),
            "gold_index": torch.tensor(int(self.gold[index]), dtype=torch.int64),
        }


def _collate(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([row[key] for row in rows], dim=0) for key in rows[0]}


def _save_head(head: SelectorHead, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}
    save_file(state, str(path))


def _plot(metrics_path: Path, out_path: Path) -> None:
    if not metrics_path.is_file():
        return
    steps: list[int] = []
    losses: list[float] = []
    val_steps: list[int] = []
    val_loss: list[float] = []
    val_acc: list[float] = []
    bea_steps: list[int] = []
    bea_overall: list[float] = []
    bea_cond: list[float] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        phase = rec.get("phase")
        if phase == "train" and "loss" in rec:
            steps.append(int(rec["step"]))
            losses.append(float(rec["loss"]))
        elif phase == "valid":
            val_steps.append(int(rec["step"]))
            val_loss.append(float(rec.get("loss") or 0.0))
            val_acc.append(float(rec.get("conditional_accuracy") or 0.0))
        elif phase == "bea1k":
            bea_steps.append(int(rec["step"]))
            bea_overall.append(float(rec.get("overall_accuracy") or 0.0))
            bea_cond.append(float(rec.get("conditional_accuracy") or 0.0))
    if not steps:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].plot(steps, losses, color="#2e86de", linewidth=1.4)
    if val_steps:
        axes[0].plot(val_steps, val_loss, color="#ee5253", marker="o", markersize=3)
    axes[0].set_title("Selector loss")
    axes[0].set_xlabel("step")
    axes[0].grid(alpha=0.25)
    if val_steps:
        axes[1].plot(val_steps, [100 * a for a in val_acc], color="#10ac84", marker="o")
    axes[1].set_title("D-pair conditional acc")
    axes[1].set_xlabel("step")
    axes[1].grid(alpha=0.25)
    if bea_steps:
        axes[2].plot(bea_steps, [100 * a for a in bea_overall], label="overall", marker="o")
        axes[2].plot(bea_steps, [100 * a for a in bea_cond], label="conditional", marker="o")
        axes[2].legend()
    axes[2].set_title("BEA-1k gate")
    axes[2].set_xlabel("step")
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


@torch.no_grad()
def eval_loader(head: SelectorHead, loader: DataLoader, device: torch.device, scaler: ScalarScaler) -> dict:
    head.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top3 = 0.0
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        features = assemble_selector_features(
            batch["x"],
            batch["t"],
            batch["c"],
            batch["scalars"],
            include_encoder=head.uses_encoder_features(),
            scaler=scaler,
        )
        logits = mask_invalid_logits(head(features), batch["valid"])
        loss = F.cross_entropy(logits, batch["gold_index"])
        bsz = int(batch["gold_index"].size(0))
        total_loss += float(loss.item()) * bsz
        total_top1 += float(topk_accuracy(logits, batch["gold_index"], k=1, candidate_valid=batch["valid"]).item()) * bsz
        total_top3 += float(topk_accuracy(logits, batch["gold_index"], k=3, candidate_valid=batch["valid"]).item()) * bsz
        n += bsz
    head.train()
    return {
        "loss": total_loss / max(1, n),
        "conditional_accuracy": total_top1 / max(1, n),
        "conditional_top3": total_top3 / max(1, n),
        "n": n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train_frozen_modernbert.yaml")
    parser.add_argument("--arm", choices=("scalar", "linear", "mlp"), required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--bea-dir", type=Path, default=ROOT / "data" / "bea60k")
    parser.add_argument("--bea-limit", type=int, default=None)
    parser.add_argument("--no-bea-gate", action="store_true")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", DEFAULT_SEED))
    seed_everything(seed)
    train_cfg = cfg.get("training", {})
    enc_cfg = cfg.get("encoder", {})
    if args.allow_cpu:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        device = select_training_device(args.device)
    print(f"device={device}{describe_cuda(device)}")

    cache_dir = Path(args.cache_dir or cfg.get("cache", {}).get("dir", "artifacts/frozen_cache"))
    train_bundle = load_feature_shards(cache_dir, "train")
    valid_bundle = load_feature_shards(cache_dir, "dpair")
    scaler = fit_scalar_scaler(train_bundle["scalars"], train_bundle["valid"])
    train_ds = SolvableFeatureDataset(train_bundle)
    valid_ds = SolvableFeatureDataset(valid_bundle)
    batch_size = int(train_cfg.get("batch_size", 512))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    hidden = int(enc_cfg.get("hidden_size", train_bundle["x"].shape[-1] or DEFAULT_HIDDEN_SIZE))
    head = SelectorHead(
        args.arm,
        hidden_size=hidden,
        n_candidates=FROZEN_CANDIDATE_SLOTS,
        dropout=float(train_cfg.get("dropout", 0.1)),
    ).to(device)
    n_params = count_parameters(head)
    print(f"arm={args.arm} trainable={n_params:,} train_solvable={len(train_ds)} dpair_solvable={len(valid_ds)}")
    if n_params <= 0:
        raise SystemExit("selector head has no trainable parameters")
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(args.lr if args.lr is not None else train_cfg.get("learning_rate", 1e-3)),
        betas=tuple(train_cfg.get("betas", (0.9, 0.999))),
        eps=float(train_cfg.get("eps", 1e-8)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    out_dir = Path(args.output_dir or Path(cfg.get("output", {}).get("model_dir", "artifacts/frozen/heads")) / args.arm)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    plot_path = out_dir / "training.png"
    meta = {
        "arm": args.arm,
        "hidden_size": hidden,
        "n_candidates": FROZEN_CANDIDATE_SLOTS,
        "scaler": scaler.to_dict(),
        "trainable_parameters": n_params,
        "seed": seed,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    h0 = score_bundle(None, valid_bundle, device=device, hunspell_baseline=True)
    print(f"H0 hunspell-first d-pair overall={h0['overall_accuracy']:.4f} cond={h0['conditional_accuracy']:.4f}")

    epochs = int(train_cfg.get("epochs", 10))
    patience = int(train_cfg.get("patience", 2))
    max_grad = float(train_cfg.get("max_grad_norm", 1.0))
    ckpt_steps = int(train_cfg.get("checkpoint_every_steps", 50))
    ckpt_seconds = float(train_cfg.get("checkpoint_every_seconds", 1800))
    bea_delta = float(train_cfg.get("bea_gate_delta", 0.01))
    bea_limit = int(args.bea_limit if args.bea_limit is not None else train_cfg.get("bea_limit", 1000))
    use_bea_gate = not args.no_bea_gate
    max_steps = args.max_steps

    best_val_acc = -1.0
    best_val_loss = float("inf")
    epochs_without_improve = 0
    global_step = 0
    nan_seen = False
    stop_reason = None
    last_bea_wall = time.time()
    best_bea_overall = None
    best_bea_cond = None
    bea_gate_fired = False
    started = time.time()

    metrics_f = metrics_path.open("w", encoding="utf-8")
    try:
        head.train()
        for epoch in range(epochs):
            running = 0.0
            seen = 0
            pbar = tqdm(train_loader, desc=f"{args.arm} epoch {epoch+1}/{epochs}")
            for batch in pbar:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                features = assemble_selector_features(
                    batch["x"],
                    batch["t"],
                    batch["c"],
                    batch["scalars"],
                    include_encoder=head.uses_encoder_features(),
                    scaler=scaler,
                )
                logits = mask_invalid_logits(head(features), batch["valid"])
                loss = F.cross_entropy(logits, batch["gold_index"])
                if not torch.isfinite(loss):
                    nan_seen = True
                    print("non-finite loss, skipping step", file=sys.stderr)
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), max_grad)
                optimizer.step()
                global_step += 1
                running += float(loss.item()) * int(batch["gold_index"].size(0))
                seen += int(batch["gold_index"].size(0))
                if global_step % int(train_cfg.get("log_every", 5)) == 0:
                    rec = {"phase": "train", "step": global_step, "epoch": epoch + 1, "loss": float(loss.item())}
                    metrics_f.write(json.dumps(rec) + "\n")
                    metrics_f.flush()
                    pbar.set_postfix(loss=float(loss.item()))
                if global_step % ckpt_steps == 0:
                    _save_head(head, out_dir / f"step_{global_step}.safetensors")
                    _save_head(head, out_dir / "last.safetensors")
                if use_bea_gate and (time.time() - last_bea_wall) >= ckpt_seconds:
                    _save_head(head, out_dir / f"wall_{global_step}.safetensors")
                    bea = evaluate_bea_limit(
                        head,
                        config=cfg,
                        bea_dir=args.bea_dir,
                        limit=bea_limit,
                        device=device,
                        scaler=scaler,
                        cache_dir=cache_dir,
                    )
                    last_bea_wall = time.time()
                    rec = {"phase": "bea1k", "step": global_step, **{k: v for k, v in bea.items() if k != "ledger"}}
                    metrics_f.write(json.dumps(rec) + "\n")
                    metrics_f.flush()
                    overall = float(bea["overall_accuracy"])
                    cond = float(bea["conditional_accuracy"])
                    print(
                        f"BEA-{bea_limit} step={global_step} overall={overall:.4f} cond={cond:.4f}",
                        flush=True,
                    )
                    if best_bea_overall is None:
                        best_bea_overall, best_bea_cond = overall, cond
                        _save_head(head, out_dir / "best_bea.safetensors")
                    else:
                        d_over = overall - best_bea_overall
                        d_cond = cond - best_bea_cond
                        if overall > best_bea_overall:
                            best_bea_overall = overall
                            _save_head(head, out_dir / "best_bea.safetensors")
                        if cond > best_bea_cond:
                            best_bea_cond = cond
                        if d_over <= bea_delta and d_cond <= bea_delta:
                            stop_reason = (
                                f"bea-gate: overall {d_over:+.4f} cond {d_cond:+.4f} "
                                f"not greater than {bea_delta:.2f}"
                            )
                            bea_gate_fired = True
                            break
                if max_steps is not None and global_step >= max_steps:
                    stop_reason = f"max-steps {max_steps}"
                    break
            val = eval_loader(head, valid_loader, device, scaler)
            val_rec = {"phase": "valid", "step": global_step, "epoch": epoch + 1, **val}
            metrics_f.write(json.dumps(val_rec) + "\n")
            metrics_f.flush()
            print(f"epoch {epoch+1} val_cond={val['conditional_accuracy']:.4f} loss={val['loss']:.4f}")
            if val["conditional_accuracy"] > best_val_acc or (
                math.isclose(val["conditional_accuracy"], max(best_val_acc, 0.0)) and val["loss"] < best_val_loss
            ):
                best_val_acc = val["conditional_accuracy"]
                best_val_loss = val["loss"]
                epochs_without_improve = 0
                _save_head(head, out_dir / "best.safetensors")
            else:
                epochs_without_improve += 1
            _save_head(head, out_dir / "last.safetensors")
            if bea_gate_fired or (max_steps is not None and global_step >= max_steps):
                break
            if epochs_without_improve >= patience:
                stop_reason = f"patience {patience} on d-pair conditional accuracy"
                break
        else:
            stop_reason = stop_reason or "max-epochs"
    finally:
        metrics_f.close()

    # Always evaluate the last epoch / current weights.
    last_val = eval_loader(head, valid_loader, device, scaler)
    _save_head(head, out_dir / "last.safetensors")
    best_path = out_dir / "best.safetensors"
    portable = out_dir / "head.safetensors"
    src = best_path if best_path.is_file() else out_dir / "last.safetensors"
    if src.is_file():
        portable.write_bytes(src.read_bytes())

    _plot(metrics_path, plot_path)
    summary = {
        **meta,
        "stop_reason": stop_reason,
        "nan_seen": nan_seen,
        "steps": global_step,
        "best_val_conditional_accuracy": best_val_acc,
        "best_val_loss": best_val_loss if best_val_loss < float("inf") else last_val["loss"],
        "last_val": last_val,
        "h0_dpair": {k: h0[k] for k in h0 if k != "ledger"},
        "best_bea1k_overall": best_bea_overall,
        "best_bea1k_conditional": best_bea_cond,
        "bea_gate_fired": bea_gate_fired,
        "duration_sec": time.time() - started,
        "output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "h0_dpair"}, indent=2))
    if nan_seen:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
