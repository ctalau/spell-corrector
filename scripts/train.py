#!/usr/bin/env python3
"""Train the 28M byte-level spelling reranker."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safetensors.torch import save_file

from spelling_reranker.byte_encoding import special_tokens_map
from spelling_reranker.config import load_train_config, model_config_from_mapping
from spelling_reranker.dataset import SpellingParquetDataset, make_collate
from spelling_reranker.hunspell import collect_hunspell_metadata
from spelling_reranker.model import ByteSpellingReranker, count_parameters, masked_cross_entropy, topk_accuracy
from spelling_reranker.seed import DEFAULT_SEED, seed_everything


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def _cosine_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model: ByteSpellingReranker, loader: DataLoader, device: torch.device, amp_dtype: torch.dtype | None) -> dict:
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top3 = 0.0
    n = 0
    gold_hist: Counter[int] = Counter()
    correct_by_gold: Counter[int] = Counter()
    for batch in loader:
        batch = _move(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
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
        total_top1 += float(topk_accuracy(logits, batch["gold_index"], k=1).item()) * bsz
        total_top3 += float(topk_accuracy(logits, batch["gold_index"], k=3).item()) * bsz
        n += bsz
        pred = logits.argmax(dim=-1)
        for gold, ok in zip(batch["gold_index"].tolist(), (pred == batch["gold_index"]).tolist()):
            gold_hist[int(gold)] += 1
            if ok:
                correct_by_gold[int(gold)] += 1
    model.train()
    acc_by_gold = {
        str(i): (correct_by_gold[i] / gold_hist[i] if gold_hist[i] else None) for i in range(10)
    }
    return {
        "loss": total_loss / max(1, n),
        "acc_top1": total_top1 / max(1, n),
        "acc_top3": total_top3 / max(1, n),
        "n": n,
        "acc_by_gold_index": acc_by_gold,
    }


def _save_checkpoint(model: ByteSpellingReranker, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train_full.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_train_config(args.config)
    seed = int(cfg.get("seed", DEFAULT_SEED))
    seed_everything(seed)

    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    out_cfg = cfg["output"]
    model_cfg = model_config_from_mapping(cfg.get("model"))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    want_bf16 = str(train_cfg.get("dtype", "bf16")).lower() == "bf16"
    use_amp = device.type == "cuda" and want_bf16 and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else None

    train_ds = SpellingParquetDataset(
        data_cfg["train"],
        max_examples=data_cfg.get("max_train_examples"),
        max_seq_len=model_cfg.max_seq_len,
    )
    valid_ds = SpellingParquetDataset(
        data_cfg["validation"],
        max_examples=data_cfg.get("max_val_examples"),
        max_seq_len=model_cfg.max_seq_len,
    )

    microbatch = int(train_cfg.get("microbatch", 128))
    accum = int(train_cfg.get("grad_accumulation", 2))
    effective = int(train_cfg.get("effective_batch_size", microbatch * accum))
    if microbatch * accum != effective:
        print(
            f"warning: microbatch {microbatch} * accum {accum} != effective {effective}",
            file=sys.stderr,
        )

    collate = make_collate(model_cfg.max_seq_len)
    pin = bool(train_cfg.get("pin_memory", True)) and device.type == "cuda"
    workers = int(train_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_ds,
        batch_size=microbatch,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
        collate_fn=collate,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=microbatch,
        shuffle=False,
        num_workers=max(0, workers // 2),
        pin_memory=pin,
        collate_fn=collate,
    )

    model = ByteSpellingReranker(model_cfg).to(device)
    n_params = count_parameters(model)
    print(f"trainable parameters: {n_params:,}")
    if bool(train_cfg.get("compile", False)):
        model = torch.compile(model)  # type: ignore[assignment]

    betas = tuple(train_cfg.get("betas", (0.9, 0.95)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        betas=betas,
        eps=float(train_cfg.get("eps", 1e-8)),
        weight_decay=float(train_cfg.get("weight_decay", 0.10)),
    )

    epochs = int(train_cfg.get("epochs", 3))
    steps_per_epoch = math.ceil(len(train_ds) / max(1, microbatch))
    updates_per_epoch = math.ceil(steps_per_epoch / max(1, accum))
    max_steps = train_cfg.get("max_steps")
    total_updates = int(max_steps) if max_steps else updates_per_epoch * epochs
    warmup = max(1, int(total_updates * float(train_cfg.get("warmup_ratio", 0.05))))
    base_lr = float(train_cfg["learning_rate"])
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))

    model_dir = Path(out_cfg.get("model_dir", "artifacts/model"))
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(out_cfg.get("metrics_jsonl", "artifacts/train_metrics.jsonl"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(out_cfg.get("summary_json", "artifacts/train_summary.json"))

    (model_dir / "config.json").write_text(json.dumps(model_cfg.to_dict(), indent=2) + "\n", encoding="utf-8")
    (model_dir / "special_tokens.json").write_text(
        json.dumps(special_tokens_map(), indent=2) + "\n", encoding="utf-8"
    )
    (model_dir / "hunspell_metadata.json").write_text(
        json.dumps(collect_hunspell_metadata(), indent=2) + "\n", encoding="utf-8"
    )

    started = time.time()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    best_val_loss = float("inf")
    best_val_acc = -1.0
    last_metrics: dict = {}
    nan_seen = False
    model.train()

    metrics_f = metrics_path.open("w", encoding="utf-8")
    try:
        for epoch in range(epochs):
            epoch_start = time.time()
            running = 0.0
            seen = 0
            pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}")
            for micro_i, batch in enumerate(pbar, start=1):
                batch = _move(batch, device)
                step_t0 = time.time()
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    logits = model(
                        batch["token_ids"],
                        batch["attention_mask"],
                        batch["typo_mask"],
                        batch["candidate_masks"],
                        batch["candidate_valid"],
                    )
                    loss = masked_cross_entropy(logits, batch["gold_index"]) / accum
                if not torch.isfinite(loss):
                    nan_seen = True
                    print("non-finite loss, skipping step", file=sys.stderr)
                    optimizer.zero_grad(set_to_none=True)
                    continue
                loss.backward()
                running += float(loss.item()) * accum * int(batch["gold_index"].size(0))
                seen += int(batch["gold_index"].size(0))

                if micro_i % accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    lr = _cosine_lr(global_step, total_updates, warmup, base_lr)
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    dt = max(1e-6, time.time() - step_t0)
                    rec = {
                        "phase": "train",
                        "step": global_step,
                        "epoch": epoch + 1,
                        "loss": float(loss.item()) * accum,
                        "lr": lr,
                        "examples_per_sec": int(batch["gold_index"].size(0)) * accum / dt,
                        "sequences_per_sec": int(batch["token_ids"].size(0)) * accum / dt,
                        "approx_bytes_per_sec": float(batch["attention_mask"].sum().item()) * accum / dt,
                    }
                    if device.type == "cuda":
                        rec["gpu_mem_allocated"] = int(torch.cuda.memory_allocated())
                        rec["gpu_mem_reserved"] = int(torch.cuda.memory_reserved())
                    if global_step % int(train_cfg.get("log_every", 20)) == 0:
                        metrics_f.write(json.dumps(rec) + "\n")
                        metrics_f.flush()
                        pbar.set_postfix(loss=rec["loss"], lr=f"{lr:.2e}")

                    if global_step % int(train_cfg.get("eval_every", 200)) == 0:
                        val = evaluate(model, valid_loader, device, amp_dtype)
                        val_rec = {"phase": "valid", "step": global_step, **val}
                        metrics_f.write(json.dumps(val_rec) + "\n")
                        metrics_f.flush()
                        last_metrics = val
                        if val["loss"] < best_val_loss:
                            best_val_loss = val["loss"]
                            _save_checkpoint(model, model_dir / "best_val_loss.safetensors")
                        if val["acc_top1"] > best_val_acc:
                            best_val_acc = val["acc_top1"]
                            _save_checkpoint(model, model_dir / "best_val_acc.safetensors")

                    if global_step >= total_updates:
                        break
            _save_checkpoint(model, model_dir / "last.safetensors")
            print(f"epoch {epoch+1} duration_sec={time.time()-epoch_start:.1f}")
            if global_step >= total_updates:
                break
    finally:
        metrics_f.close()

    if last_metrics:
        val = last_metrics
    else:
        val = evaluate(model, valid_loader, device, amp_dtype)
        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            _save_checkpoint(model, model_dir / "best_val_loss.safetensors")
        if val["acc_top1"] > best_val_acc:
            best_val_acc = val["acc_top1"]
            _save_checkpoint(model, model_dir / "best_val_acc.safetensors")

    # Prefer best-val-acc weights as the portable model.safetensors
    best_acc_path = model_dir / "best_val_acc.safetensors"
    last_path = model_dir / "last.safetensors"
    portable = model_dir / "model.safetensors"
    src = best_acc_path if best_acc_path.is_file() else last_path
    if src.is_file():
        portable.write_bytes(src.read_bytes())

    duration = time.time() - started
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    peak_vram = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
    train_hash = None
    valid_hash = None
    try:
        manifest = json.loads((ROOT / "data" / "processed" / "manifest.json").read_text(encoding="utf-8"))
        train_hash = manifest.get("files", {}).get("train", {}).get("sha256")
        valid_hash = manifest.get("files", {}).get("validation", {}).get("sha256")
    except OSError:
        pass

    training_manifest = {
        "git_commit": _git_sha(),
        "seed": seed,
        "dataset_hashes": {"train": train_hash, "validation": valid_hash},
        "train_examples": len(train_ds),
        "validation_examples": len(valid_ds),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu_name,
        "device": str(device),
        "amp_bf16": use_amp,
        "total_trainable_parameters": n_params,
        "microbatch": microbatch,
        "grad_accumulation": accum,
        "effective_batch_size": microbatch * accum,
        "optimizer_steps": global_step,
        "training_duration_sec": duration,
        "best_validation_loss": best_val_loss if best_val_loss < float("inf") else val["loss"],
        "best_validation_acc_top1": best_val_acc if best_val_acc >= 0 else val["acc_top1"],
        "last_validation": val,
        "nan_seen": nan_seen,
        "hostname": platform.node(),
        "config": str(args.config),
    }
    (model_dir / "training_manifest.json").write_text(
        json.dumps(training_manifest, indent=2) + "\n", encoding="utf-8"
    )
    summary_path.write_text(json.dumps(training_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(training_manifest, indent=2))
    return 0


if __name__ == "__main__":
    # Allow DataLoader workers to find the package when run as a script.
    os.environ.setdefault("PYTHONPATH", str(ROOT))
    raise SystemExit(main())
