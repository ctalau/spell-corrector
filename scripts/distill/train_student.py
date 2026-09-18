#!/usr/bin/env python3
"""M7: distil the 2B Q4 corrector into the 0.8B Q4 student.

Student: `Qwen/Qwen3.5-0.8B` in 4-bit NF4 with a fresh LoRA (QLoRA).
Teacher: top-K logits precomputed by `dump_teacher_logits.py` from the 2B Q4
milestone-6 model, so the teacher is never resident during training.

Loss per answer token:

    L = (1 - alpha) * CE(gold) + alpha * T^2 * KL(teacher_topK || student)

The teacher distribution is renormalized over its own top-K support at
temperature T; the student's log-probabilities are gathered at those same ids.

Throughput comes from: precomputed teacher logits, 4-bit NF4 weights, bf16
compute, flash-attention-2 when importable (SDPA otherwise), Liger fused
RMSNorm/SwiGLU/RoPE when it patches the architecture, paged 8-bit AdamW, TF32
matmuls, length-bucketed batches with dynamic padding, and no gradient
checkpointing unless the OOM ladder turns it on.

Progress is written continuously to `--out-dir`: STATUS, metrics.jsonl, and
best/ (highest dev accuracy adapter so far).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    apply_chat,
    encode_example,
    load_jsonl,
    normalize_prediction,
    tokenizer_fingerprint,
    write_jsonl,
)


def log_status(out_dir: Path, text: str) -> None:
    (out_dir / "STATUS").write_text(text + "\n", encoding="utf-8")
    print(f"STATUS {text}", flush=True)


def log_metric(out_dir: Path, payload: dict) -> None:
    with (out_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    print("METRIC " + json.dumps(payload), flush=True)


def probe_kernels() -> dict:
    info = {}
    for name in ("flash_attn", "liger_kernel", "causal_conv1d", "fla"):
        try:
            mod = __import__(name)
            info[name] = str(getattr(mod, "__version__", "ok"))
        except Exception as exc:  # noqa: BLE001
            info[name] = f"missing ({type(exc).__name__})"
    return info


def build_batches(lengths: list[int], micro: int, seed: int, bucket: int = 64) -> list[list[int]]:
    """Length-bucketed batches: sort inside a shuffled window, shuffle batches."""
    rng = random.Random(seed)
    order = list(range(len(lengths)))
    rng.shuffle(order)
    window = max(micro, micro * bucket)
    batches: list[list[int]] = []
    for start in range(0, len(order), window):
        chunk = sorted(order[start : start + window], key=lambda i: lengths[i])
        for b in range(0, len(chunk), micro):
            batches.append(chunk[b : b + micro])
    rng.shuffle(batches)
    return batches


def load_student(args, torch):
    """4-bit NF4 student. Qwen3.5 ships as a conditional-generation (VLM) class,
    so the image-text loader is tried before the causal-LM one -- the same order
    the M4-M6 milestone scripts settled on."""
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        BitsAndBytesConfig,
    )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    attn = "sdpa"
    try:
        import flash_attn  # noqa: F401

        attn = "flash_attention_2"
    except Exception:  # noqa: BLE001
        pass

    last_err = None
    model = None
    for loader in (AutoModelForImageTextToText, AutoModelForCausalLM):
        for attn_try in ([attn, "sdpa"] if attn != "sdpa" else ["sdpa"]):
            try:
                model = loader.from_pretrained(
                    args.base_model,
                    revision=args.revision or None,
                    trust_remote_code=True,
                    quantization_config=bnb,
                    device_map={"": 0},
                    torch_dtype=torch.bfloat16,
                    attn_implementation=attn_try,
                )
                attn = attn_try
                print(f"loaded student via {loader.__name__} attn={attn_try}", flush=True)
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                print(f"loader {loader.__name__}/{attn_try} failed: {exc}", flush=True)
        if model is not None:
            break
    if model is None:
        raise SystemExit(f"could not load student: {last_err}")

    liger_note = "not attempted"
    if args.liger:
        try:
            from liger_kernel.transformers import _apply_liger_kernel_to_instance

            _apply_liger_kernel_to_instance(model=model)
            liger_note = "applied"
        except Exception as exc:  # noqa: BLE001
            liger_note = f"skipped ({type(exc).__name__}: {exc})"
    return model, attn, liger_note


def attach_lora(model, args, torch):
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=args.gradient_checkpointing
    )
    if args.resume_adapter:
        model = PeftModel.from_pretrained(model, str(args.resume_adapter), is_trainable=True)
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        return model, f"resumed from {args.resume_adapter}"
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_targets.split(","),
        bias="none",
    )
    return get_peft_model(model, cfg), "fresh LoRA"


def evaluate(model, tok_left, rows, torch, batch_size=48, max_new_tokens=6, limit=0):
    """Greedy batched generation; returns (exact, casefold, predictions)."""
    rows = rows[:limit] if limit else rows
    model.eval()
    prev_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = True  # KV cache off during training, on to generate
    preds = []
    exact = casefold = 0
    pad_id = tok_left.pad_token_id
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            texts = [apply_chat(tok_left, r["user_text"]) for r in chunk]
            enc = tok_left(texts, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=pad_id,
            )
            gen = out[:, enc["input_ids"].shape[1] :]
            texts_out = tok_left.batch_decode(gen, skip_special_tokens=True)
            for r, raw in zip(chunk, texts_out):
                pred = normalize_prediction(raw)
                is_exact = pred == r["gold"]
                is_cf = pred.casefold() == r["gold"].casefold()
                exact += int(is_exact)
                casefold += int(is_cf)
                preds.append(
                    {
                        "error_index": r["error_index"],
                        "typo": r["typo"],
                        "gold": r["gold"],
                        "sentence": r["sentence"],
                        "raw_output": raw,
                        "pred": pred,
                        "exact": is_exact,
                        "casefold": is_cf,
                    }
                )
    model.config.use_cache = prev_cache
    model.train()
    n = max(1, len(rows))
    return exact / n, casefold / n, preds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-jsonl", type=Path, default=ROOT / "data/distill/train.jsonl")
    ap.add_argument("--dev-jsonl", type=Path, default=ROOT / "data/distill/dev.jsonl")
    ap.add_argument("--teacher-dir", type=Path, default=ROOT / "artifacts/distill/teacher")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/distill/student")
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--revision", default="2fc06364715b967f1860aea9cf38778875588b17")
    ap.add_argument("--resume-adapter", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = derive from epochs")
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--kd-alpha", type=float, default=0.5)
    ap.add_argument("--kd-temperature", type=float, default=2.0)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora-targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )
    ap.add_argument("--max-seq-len", type=int, default=256)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-limit", type=int, default=500, help="dev rows per mid-run eval")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--no-liger", dest="liger", action="store_false")
    ap.add_argument("--stop-at-acc", type=float, default=0.0, help="0 = run the whole schedule")
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    kernels = probe_kernels()
    log_status(args.out_dir, f"loading student {args.base_model}")
    print(f"kernels={json.dumps(kernels)}", flush=True)

    tok = AutoTokenizer.from_pretrained(
        args.base_model, revision=args.revision or None, trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok_left = AutoTokenizer.from_pretrained(
        args.base_model, revision=args.revision or None, trust_remote_code=True
    )
    if tok_left.pad_token is None:
        tok_left.pad_token = tok_left.eos_token
    tok_left.padding_side = "left"

    fingerprint = tokenizer_fingerprint(tok)
    teacher_meta = json.loads((args.teacher_dir / "meta.json").read_text(encoding="utf-8"))
    kd_enabled = teacher_meta["tokenizer_fingerprint"] == fingerprint
    if not kd_enabled:
        raise SystemExit(
            "teacher and student tokenizers differ "
            f"({teacher_meta['tokenizer_fingerprint']} vs {fingerprint}); "
            "token-level KD would be meaningless. Rebuild the teacher dump or "
            "switch to sequence-level distillation."
        )

    train_rows = load_jsonl(args.train_jsonl)[: teacher_meta["n_rows"]]
    dev_rows = load_jsonl(args.dev_jsonl)
    enc = [encode_example(tok, r, args.max_seq_len) for r in train_rows]
    lengths = [len(e["input_ids"]) for e in enc]

    t_ids = np.load(args.teacher_dir / "topk_ids.npy", mmap_mode="r")
    t_logits = np.load(args.teacher_dir / "topk_logits.npy", mmap_mode="r")
    t_alen = np.load(args.teacher_dir / "answer_len.npy")
    t_aids = np.load(args.teacher_dir / "answer_ids.npy")
    L_cap = int(t_ids.shape[1])

    usable = [i for i in range(len(train_rows)) if t_alen[i] > 0 and len(enc[i]["answer_ids"]) > 0]
    print(f"train_rows={len(train_rows)} usable={len(usable)} dev={len(dev_rows)}", flush=True)

    model, attn, liger_note = load_student(args, torch)
    model, lora_note = attach_lora(model, args, torch)
    model.config.use_cache = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"attn={attn} liger={liger_note} lora={lora_note} "
        f"trainable={trainable/1e6:.2f}M/{total/1e6:.1f}M",
        flush=True,
    )

    try:
        import bitsandbytes as bnb

        optimizer = bnb.optim.PagedAdamW8bit(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )
        optim_note = "bnb.PagedAdamW8bit"
    except Exception as exc:  # noqa: BLE001
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
            fused=True,
        )
        optim_note = f"torch fused AdamW ({type(exc).__name__})"

    micro = args.micro_batch
    batches = build_batches([lengths[i] for i in usable], micro, args.seed)
    batches = [[usable[j] for j in b] for b in batches]
    steps_per_epoch = max(1, len(batches) // args.grad_accum)
    total_steps = args.max_steps or int(steps_per_epoch * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )
    run_meta = {
        "base_model": args.base_model,
        "revision": args.revision,
        "teacher": {k: teacher_meta[k] for k in ("base_model", "adapter_dir", "topk", "teacher_forced_token_agreement")},
        "kd_alpha": args.kd_alpha,
        "kd_temperature": args.kd_temperature,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                  "targets": args.lora_targets},
        "micro_batch": micro,
        "grad_accum": args.grad_accum,
        "effective_batch": micro * args.grad_accum,
        "lr": args.lr,
        "epochs": args.epochs,
        "total_optimizer_steps": total_steps,
        "n_train": len(usable),
        "attn_implementation": attn,
        "liger": liger_note,
        "optimizer": optim_note,
        "gradient_checkpointing": args.gradient_checkpointing,
        "kernels": kernels,
        "trainable_params": trainable,
        "total_params": total,
        "seed": args.seed,
    }
    (args.out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(json.dumps(run_meta, indent=2), flush=True)

    pad_id = tok.pad_token_id
    T = args.kd_temperature
    alpha = args.kd_alpha
    best_acc = -1.0
    step = 0
    micro_step_count = 0
    t0 = time.time()
    loss_acc = ce_acc = kd_acc = 0.0
    loss_n = 0
    tokens_seen = 0
    model.train()
    stop = False

    def micro_step(batch_idx: list[int], scale: float) -> tuple[float, float, float, int]:
        """One forward/backward over `batch_idx`, gradients scaled by `scale`.

        On CUDA OOM the batch is split in half and retried, so a single long
        example cannot kill an hours-long run.
        """
        try:
            rows_enc = [enc[i] for i in batch_idx]
            width = max(len(e["input_ids"]) for e in rows_enc)
            b = len(batch_idx)
            ids = torch.full((b, width), pad_id, dtype=torch.long)
            mask = torch.zeros((b, width), dtype=torch.long)
            a_len = [
                min(int(t_alen[i]), len(rows_enc[k]["answer_ids"]), L_cap)
                for k, i in enumerate(batch_idx)
            ]
            Lb = max(1, max(a_len))
            pos = torch.zeros((b, Lb), dtype=torch.long)
            valid = torch.zeros((b, Lb), dtype=torch.bool)
            gold = torch.zeros((b, Lb), dtype=torch.long)
            tk_ids = torch.zeros((b, Lb, t_ids.shape[2]), dtype=torch.long)
            tk_log = torch.zeros((b, Lb, t_ids.shape[2]), dtype=torch.float32)
            for k, i in enumerate(batch_idx):
                e = rows_enc[k]
                n_tok = len(e["input_ids"])
                ids[k, :n_tok] = torch.tensor(e["input_ids"])
                mask[k, :n_tok] = 1
                al = a_len[k]
                if al == 0:
                    continue
                first = e["prompt_len"] - 1
                pos[k, :al] = torch.arange(first, first + al)
                valid[k, :al] = True
                gold[k, :al] = torch.tensor(np.asarray(t_aids[i, :al], dtype=np.int64))
                tk_ids[k, :al] = torch.tensor(np.asarray(t_ids[i, :al], dtype=np.int64))
                tk_log[k, :al] = torch.tensor(np.asarray(t_logits[i, :al], dtype=np.float32))

            dev = model.device
            logits = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits
            pos, valid = pos.to(dev), valid.to(dev)
            gold, tk_ids, tk_log = gold.to(dev), tk_ids.to(dev), tk_log.to(dev)

            sel = torch.gather(
                logits, 1, pos.unsqueeze(-1).expand(-1, -1, logits.size(-1))
            ).float()  # [B, Lb, V]
            n_valid = valid.sum().clamp(min=1)

            ce = -torch.log_softmax(sel, dim=-1).gather(-1, gold.unsqueeze(-1)).squeeze(-1)
            ce = (ce * valid).sum() / n_valid

            student_lp = torch.log_softmax(sel / T, dim=-1).gather(-1, tk_ids)
            teacher_p = torch.softmax(tk_log / T, dim=-1)
            kd = (teacher_p * (torch.log(teacher_p.clamp_min(1e-9)) - student_lp)).sum(-1)
            kd = (kd * valid).sum() / n_valid

            loss = (1.0 - alpha) * ce + alpha * (T * T) * kd
            (loss * scale).backward()
            return float(loss.detach()), float(ce.detach()), float(kd.detach()), int(mask.sum())
        except torch.cuda.OutOfMemoryError:
            if len(batch_idx) == 1:
                raise
            torch.cuda.empty_cache()
            half = len(batch_idx) // 2
            a = micro_step(batch_idx[:half], scale)
            c = micro_step(batch_idx[half:], scale)
            return (
                (a[0] + c[0]) / 2,
                (a[1] + c[1]) / 2,
                (a[2] + c[2]) / 2,
                a[3] + c[3],
            )

    epoch = 0
    while step < total_steps and not stop:
        epoch += 1
        if epoch > 1:
            batches = build_batches([lengths[i] for i in usable], micro, args.seed + epoch)
            batches = [[usable[j] for j in b] for b in batches]
        for batch_idx in batches:
            loss_v, ce_v, kd_v, n_tok = micro_step(batch_idx, 1.0 / args.grad_accum)
            loss_acc += loss_v
            ce_acc += ce_v
            kd_acc += kd_v
            loss_n += 1
            tokens_seen += n_tok
            micro_step_count += 1

            if micro_step_count % args.grad_accum:
                continue

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                rate = step / max(elapsed, 1e-9)
                log_metric(
                    args.out_dir,
                    {
                        "kind": "train",
                        "step": step,
                        "total_steps": total_steps,
                        "epoch": epoch,
                        "loss": round(loss_acc / loss_n, 5),
                        "ce": round(ce_acc / loss_n, 5),
                        "kd": round(kd_acc / loss_n, 5),
                        "lr": scheduler.get_last_lr()[0],
                        "tokens_per_s": round(tokens_seen / max(elapsed, 1e-6), 1),
                        "steps_per_s": round(rate, 4),
                        "elapsed_s": round(elapsed, 1),
                        "eta_min": round((total_steps - step) / max(rate, 1e-9) / 60, 1),
                        "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    },
                )
                loss_acc = ce_acc = kd_acc = 0.0
                loss_n = 0
                log_status(
                    args.out_dir,
                    f"training step {step}/{total_steps} "
                    f"({100 * step / total_steps:.1f}%) best_dev_casefold={best_acc:.4f}",
                )

            if step % args.eval_every == 0 or step >= total_steps:
                ex, cf, _ = evaluate(model, tok_left, dev_rows, torch, limit=args.eval_limit)
                log_metric(
                    args.out_dir,
                    {
                        "kind": "dev_eval",
                        "step": step,
                        "n": min(args.eval_limit or len(dev_rows), len(dev_rows)),
                        "acc_exact": round(ex, 4),
                        "acc_casefold": round(cf, 4),
                        "elapsed_s": round(time.time() - t0, 1),
                    },
                )
                if cf > best_acc:
                    best_acc = cf
                    model.save_pretrained(str(args.out_dir / "best"))
                    (args.out_dir / "best" / "best_meta.json").write_text(
                        json.dumps(
                            {"step": step, "dev_acc_exact": ex, "dev_acc_casefold": cf}, indent=2
                        ),
                        encoding="utf-8",
                    )
                model.save_pretrained(str(args.out_dir / "last"))
                log_status(
                    args.out_dir,
                    f"step {step}/{total_steps} dev_casefold={cf:.4f} best={best_acc:.4f}",
                )
                if args.stop_at_acc and cf >= args.stop_at_acc:
                    stop = True
                    break
            if step >= total_steps:
                stop = True
                break

    model.save_pretrained(str(args.out_dir / "last"))
    ex, cf, preds = evaluate(model, tok_left, dev_rows, torch, limit=0)
    write_jsonl(args.out_dir / "dev_predictions_last.jsonl", preds)
    summary = {
        "steps_done": step,
        "wall_minutes": round((time.time() - t0) / 60, 2),
        "best_dev_casefold": best_acc,
        "final_dev_acc_exact": ex,
        "final_dev_acc_casefold": cf,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        **run_meta,
    }
    (args.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_metric(args.out_dir, {"kind": "final", **{k: summary[k] for k in (
        "steps_done", "wall_minutes", "best_dev_casefold", "final_dev_acc_casefold")}})
    log_status(args.out_dir, f"training done: best dev casefold {best_acc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
