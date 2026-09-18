#!/usr/bin/env python3
"""QLoRA (4-bit NF4) fine-tune Qwen3.5-0.8B for direct spelling correction.

Prefer continue-from M4 LoRA adapters on a 4-bit base; fall back to fresh LoRA.
Speedups: larger micro-batch, optional torch.compile, optional flash kernels,
gradient checkpointing off by default (VRAM headroom from 4-bit).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def apply_chat(tokenizer, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return user_text + "\nAnswer:"


def probe_kernels() -> dict:
    info = {
        "causal_conv1d": False,
        "flash_linear_attention": False,
        "flash_attn": False,
        "notes": [],
    }
    try:
        import causal_conv1d  # noqa: F401

        info["causal_conv1d"] = True
        info["notes"].append(f"causal_conv1d={getattr(causal_conv1d, '__version__', '?')}")
    except Exception as e:  # noqa: BLE001
        info["notes"].append(f"causal_conv1d missing: {e}")
    try:
        import flash_linear_attention  # noqa: F401

        info["flash_linear_attention"] = True
        info["notes"].append("flash_linear_attention=ok")
    except Exception as e:  # noqa: BLE001
        info["notes"].append(f"flash_linear_attention missing: {e}")
    try:
        import flash_attn  # noqa: F401

        info["flash_attn"] = True
        info["notes"].append(f"flash_attn={getattr(flash_attn, '__version__', '?')}")
    except Exception as e:  # noqa: BLE001
        info["notes"].append(f"flash_attn missing: {e}")
    return info


def load_base_4bit(model_id: str, revision: str, qcfg: dict, dtype):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        BitsAndBytesConfig,
    )

    compute = str(qcfg.get("bnb_4bit_compute_dtype", "bfloat16")).lower()
    compute_dtype = torch.bfloat16 if compute in ("bf16", "bfloat16") else torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=bool(qcfg.get("load_in_4bit", True)),
        bnb_4bit_quant_type=str(qcfg.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(qcfg.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=compute_dtype,
    )
    rev_kwargs = {"revision": revision} if revision else {}
    last_err = None
    for loader in (AutoModelForImageTextToText, AutoModelForCausalLM):
        try:
            model = loader.from_pretrained(
                model_id,
                trust_remote_code=True,
                quantization_config=bnb,
                device_map={"": 0} if torch.cuda.is_available() else None,
                torch_dtype=dtype,
                **rev_kwargs,
            )
            print(f"loaded 4bit via {loader.__name__}", flush=True)
            return model
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"loader {loader.__name__} failed: {e}", flush=True)
    raise RuntimeError(f"failed to load 4-bit model: {last_err}")


def attach_lora(model, cfg: dict, lora_cfg: dict) -> tuple:
    """Return (model, mode_str, detail_dict). Prefer continue-from M4."""
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=bool((cfg.get("train") or {}).get("gradient_checkpointing", False))
    )

    continue_flag = bool(cfg.get("continue_from_m4", True))
    m4_dir = ROOT / cfg.get("m4_adapter_dir", "models/qwen35_0_8b_direct_lora")
    detail = {"continue_from_m4_requested": continue_flag, "m4_adapter_dir": str(m4_dir)}

    if continue_flag and m4_dir.is_dir() and (m4_dir / "adapter_config.json").is_file():
        try:
            print(f"trying continue-from-M4 adapters: {m4_dir}", flush=True)
            model = PeftModel.from_pretrained(model, str(m4_dir), is_trainable=True)
            # Ensure trainable
            for n, p in model.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True
            detail["continued_from_m4"] = True
            detail["fresh_lora"] = False
            print("continue_from_m4=SUCCESS", flush=True)
            return model, "continue_from_m4", detail
        except Exception as e:  # noqa: BLE001
            detail["continue_error"] = repr(e)
            detail["continue_traceback"] = traceback.format_exc()
            print(f"continue_from_m4 FAILED: {e}", flush=True)
            print("falling back to fresh LoRA on 4-bit base", flush=True)
            # Need a clean base again — caller should re-load if needed.
            # If PeftModel partially wrapped, try to get base.
            try:
                if hasattr(model, "get_base_model"):
                    model = model.get_base_model()
            except Exception:
                pass
            detail["continued_from_m4"] = False
            detail["fresh_lora"] = True
            # Re-prepare and attach fresh
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=bool(
                    (cfg.get("train") or {}).get("gradient_checkpointing", False)
                ),
            )
    else:
        detail["continued_from_m4"] = False
        detail["fresh_lora"] = True
        if continue_flag:
            detail["continue_skip_reason"] = "m4 adapter dir missing"

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(
            lora_cfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        ),
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    print("fresh_lora_on_4bit=True", flush=True)
    return model, "fresh_lora", detail


def build_datasets(train_rows, val_rows, tokenizer, max_len: int):
    from datasets import Dataset

    def encode_row(row: dict) -> dict:
        prompt = apply_chat(tokenizer, row["user_text"])
        answer = str(row.get("target") or row.get("gold") or "").strip()
        eos = tokenizer.eos_token or ""
        full = prompt + answer + eos
        tok_prompt = tokenizer(prompt, add_special_tokens=False)
        tok_full = tokenizer(
            full,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
        )
        input_ids = tok_full["input_ids"]
        labels = list(input_ids)
        prompt_len = min(len(tok_prompt["input_ids"]), len(labels))
        for i in range(prompt_len):
            labels[i] = -100
        if all(x == -100 for x in labels) and labels:
            labels[-1] = input_ids[-1]
        return {
            "input_ids": input_ids,
            "attention_mask": tok_full["attention_mask"],
            "labels": labels,
        }

    print("tokenizing...", flush=True)
    train_enc = [encode_row(r) for r in train_rows]
    val_enc = [encode_row(r) for r in val_rows] if val_rows else []
    train_ds = Dataset.from_list(train_enc)
    val_ds = Dataset.from_list(val_enc) if val_enc else None
    return train_ds, val_ds


class PadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features):
        import torch as T

        max_l = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad_n = max_l - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_id] * pad_n)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad_n)
            batch["labels"].append(f["labels"] + [-100] * pad_n)
        return {k: T.tensor(v) for k, v in batch.items()}


def make_training_args(tcfg, out_dir, train_ds, seed, max_steps, torch_mod):
    import inspect
    from transformers import TrainingArguments

    ta_kwargs = dict(
        output_dir=str(out_dir / "trainer_out"),
        per_device_train_batch_size=int(tcfg.get("per_device_train_batch_size", 8)),
        per_device_eval_batch_size=int(tcfg.get("per_device_eval_batch_size", 4)),
        gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 4)),
        learning_rate=float(tcfg.get("lr", 2e-4)),
        num_train_epochs=float(tcfg.get("epochs", 1)),
        max_steps=max_steps,
        logging_steps=int(tcfg.get("logging_steps", 20)),
        save_steps=int(tcfg.get("save_steps", 200)),
        save_total_limit=2,
        bf16=bool(tcfg.get("bf16", True)) and torch_mod.cuda.is_available(),
        fp16=False,
        report_to=[],
        seed=seed,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        optim="paged_adamw_8bit",
    )
    _ta_params = set(inspect.signature(TrainingArguments.__init__).parameters)
    # Drop optim if unsupported
    if "optim" not in _ta_params:
        ta_kwargs.pop("optim", None)
    warm_ratio = float(tcfg.get("warmup_ratio", 0.03))
    if "warmup_ratio" in _ta_params:
        ta_kwargs["warmup_ratio"] = warm_ratio
    elif "warmup_steps" in _ta_params:
        steps_per_epoch = max(
            1,
            len(train_ds)
            // (
                ta_kwargs["per_device_train_batch_size"]
                * ta_kwargs["gradient_accumulation_steps"]
            ),
        )
        ta_kwargs["warmup_steps"] = max(1, int(steps_per_epoch * warm_ratio))
    if "eval_strategy" in _ta_params:
        ta_kwargs["eval_strategy"] = "no"
    elif "evaluation_strategy" in _ta_params:
        ta_kwargs["evaluation_strategy"] = "no"
    if "gradient_checkpointing" in _ta_params:
        ta_kwargs["gradient_checkpointing"] = bool(tcfg.get("gradient_checkpointing", False))
    return TrainingArguments(**ta_kwargs)


def run_train(model, training_args, train_ds, tokenizer, resume):
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )
    t0 = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume)
    elapsed = time.time() - t0
    return trainer, train_result, elapsed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "milestone5_qlora.yaml")
    p.add_argument("--train-jsonl", type=Path, default=None)
    p.add_argument("--val-jsonl", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--force-fresh", action="store_true", help="Skip continue-from-M4")
    args = p.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cfg = load_yaml(args.config) if args.config.is_file() else {}
    if args.force_fresh:
        cfg["continue_from_m4"] = False
    seed = int(cfg.get("seed", 1337))
    random.seed(seed)

    train_path = Path(
        args.train_jsonl
        or (ROOT / cfg.get("train_data_dir", "data/direct_train") / "train.jsonl")
    )
    val_path = Path(
        args.val_jsonl or (ROOT / cfg.get("train_data_dir", "data/direct_train") / "val.jsonl")
    )
    out_dir = Path(
        args.output_dir or (ROOT / cfg.get("adapter_dir", "models/qwen35_0_8b_direct_qlora"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(train_path)
    val_rows = load_jsonl(val_path) if val_path.is_file() else []
    print(f"train={len(train_rows)} val={len(val_rows)} seed={seed}", flush=True)

    kernel_info = probe_kernels()
    print(f"kernels={json.dumps(kernel_info)}", flush=True)

    model_id = cfg.get("base_model", "Qwen/Qwen3.5-0.8B")
    revision = cfg.get("revision", "2fc06364715b967f1860aea9cf38778875588b17")
    lora_cfg = cfg.get("lora") or {}
    tcfg = dict(cfg.get("train") or {})
    qcfg = cfg.get("quantization") or {}

    import torch
    from transformers import AutoTokenizer

    print(f"cuda={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    rev_kwargs = {"revision": revision} if revision else {}
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, **rev_kwargs
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = (
        torch.bfloat16
        if (tcfg.get("bf16", True) and torch.cuda.is_available())
        else torch.float16
    )

    max_len = int(tcfg.get("max_seq_length", 256))
    train_ds, _val_ds = build_datasets(train_rows, val_rows, tokenizer, max_len)
    max_steps = args.max_steps if args.max_steps is not None else int(tcfg.get("max_steps", -1))

    # Batch fallback ladder: try configured, then smaller micro-batch, then grad ckpt
    batch_ladder = [
        {
            "per_device_train_batch_size": int(tcfg.get("per_device_train_batch_size", 8)),
            "gradient_accumulation_steps": int(tcfg.get("gradient_accumulation_steps", 4)),
            "gradient_checkpointing": bool(tcfg.get("gradient_checkpointing", False)),
        },
        {
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 8,
            "gradient_checkpointing": bool(tcfg.get("gradient_checkpointing", False)),
        },
        {
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 8,
            "gradient_checkpointing": True,
        },
        {
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 16,
            "gradient_checkpointing": True,
        },
    ]
    # Deduplicate
    seen = set()
    ladder = []
    for b in batch_ladder:
        key = (
            b["per_device_train_batch_size"],
            b["gradient_accumulation_steps"],
            b["gradient_checkpointing"],
        )
        if key not in seen:
            seen.add(key)
            ladder.append(b)

    want_compile = bool(tcfg.get("torch_compile", True)) and not args.no_compile
    last_err = None
    trainer = None
    train_result = None
    elapsed = None
    mode = None
    attach_detail = {}
    used_batch = None
    compile_status = "skipped"

    for attempt, batch_cfg in enumerate(ladder):
        print(f"\n=== attempt {attempt+1}/{len(ladder)} batch_cfg={batch_cfg} ===", flush=True)
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = load_base_4bit(model_id, revision, qcfg, dtype)
            if hasattr(model, "config"):
                model.config.use_cache = False

            # Update tcfg for this attempt
            attempt_tcfg = dict(tcfg)
            attempt_tcfg.update(batch_cfg)

            # prepare_model_for_kbit uses gradient_checkpointing from tcfg at attach time
            cfg_attempt = dict(cfg)
            cfg_attempt["train"] = attempt_tcfg
            model, mode, attach_detail = attach_lora(model, cfg_attempt, lora_cfg)

            if batch_cfg["gradient_checkpointing"]:
                try:
                    model.enable_input_require_grads()
                except Exception:
                    pass
                try:
                    model.gradient_checkpointing_enable()
                except Exception:
                    pass
                print("gradient_checkpointing=ON", flush=True)
            else:
                print("gradient_checkpointing=OFF", flush=True)

            model.print_trainable_parameters()

            # transformers/PEFT refuse torch.compile on quantized models during fine-tune
            if want_compile and hasattr(torch, "compile"):
                compile_status = "skipped_quantized_peft_incompatible"
                print(
                    "torch.compile=SKIPPED (PEFT+bitsandbytes QLoRA incompatible with Trainer)",
                    flush=True,
                )
            else:
                compile_status = "disabled"

            training_args = make_training_args(
                attempt_tcfg, out_dir, train_ds, seed, max_steps, torch
            )
            print(
                f"eff_batch={training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}",
                flush=True,
            )
            trainer, train_result, elapsed = run_train(
                model, training_args, train_ds, tokenizer, args.resume
            )
            used_batch = {
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "gradient_checkpointing": batch_cfg["gradient_checkpointing"],
                "effective_batch": training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps,
                "lr": training_args.learning_rate,
                "optim": getattr(training_args, "optim", None),
            }
            last_err = None
            break
        except RuntimeError as e:
            last_err = e
            msg = str(e).lower()
            print(f"attempt failed: {e}", flush=True)
            if "out of memory" in msg or "cuda" in msg and "memory" in msg:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            # Non-OOM: if continue-from failed mid-train oddly, try fresh once
            raise
        except Exception as e:
            last_err = e
            print(f"attempt failed (non-OOM): {e}", flush=True)
            traceback.print_exc()
            msg = str(e).lower()
            # Compile+QLoRA conflict: disable compile and retry same batch/mode
            if "torch.compile" in msg or "compiled model" in msg:
                print("disabling torch.compile and retrying...", flush=True)
                want_compile = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            # If continue-from attach somehow poisoned the model, force fresh
            if mode == "continue_from_m4" and "peft" in msg and attempt == 0:
                print("retrying with --force-fresh path next attempt...", flush=True)
                cfg["continue_from_m4"] = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise

    if trainer is None or train_result is None:
        raise RuntimeError(f"all train attempts failed; last_err={last_err}")

    loss_logs = [
        {"step": int(x["step"]), "loss": float(x["loss"])}
        for x in (trainer.state.log_history or [])
        if "loss" in x
    ]
    (out_dir / "loss_curve.json").write_text(json.dumps(loss_logs, indent=2) + "\n")

    # Unwrap compile if needed for save
    save_model = trainer.model
    if hasattr(save_model, "_orig_mod"):
        save_model = save_model._orig_mod
    save_model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    peak_vram_gb = None
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

    meta = {
        "task": "direct_correct_qlora",
        "mode": mode,
        "attach_detail": attach_detail,
        "continued_from_m4": bool(attach_detail.get("continued_from_m4")),
        "fresh_lora_on_4bit": bool(attach_detail.get("fresh_lora")),
        "base_model": model_id,
        "revision": revision,
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": qcfg.get("bnb_4bit_quant_type", "nf4"),
            "bnb_4bit_use_double_quant": qcfg.get("bnb_4bit_use_double_quant", True),
            "bnb_4bit_compute_dtype": qcfg.get("bnb_4bit_compute_dtype", "bfloat16"),
        },
        "adapter_dir": str(out_dir),
        "seed": seed,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "lora": {
            "r": int(lora_cfg.get("r", 16)),
            "alpha": int(lora_cfg.get("alpha", 32)),
            "dropout": float(lora_cfg.get("dropout", 0.05)),
            "target_modules": list(
                lora_cfg.get(
                    "target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                )
            ),
        },
        "train": {
            **(used_batch or {}),
            "epochs": float(tcfg.get("epochs", 1)),
            "max_steps": max_steps,
            "max_seq_length": max_len,
            "torch_compile": compile_status,
        },
        "kernels": kernel_info,
        "train_loss": float(train_result.training_loss) if train_result else None,
        "n_loss_logs": len(loss_logs),
        "first_logged_loss": loss_logs[0]["loss"] if loss_logs else None,
        "last_logged_loss": loss_logs[-1]["loss"] if loss_logs else None,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "bitsandbytes": getattr(__import__("bitsandbytes"), "__version__", "?"),
        "peft": __import__("peft").__version__,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)
    print(f"DONE adapters -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
