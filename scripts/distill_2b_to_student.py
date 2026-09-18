#!/usr/bin/env python3
"""Milestone 7: distill Qwen3.5-2B (teacher, M6 QLoRA, 91% Acc@1) into a fresh
Qwen3.5-0.8B (student) QLoRA, both 4-bit NF4, using BEA-60K itself as the
distillation corpus.

Explicit, user-approved exception to the byte-level reranker's locked-benchmark
rule (see CLAUDE.md "Milestone 7" section): a fixed seed-1337 100-example
holdout (`data/distill/holdout_100.json`) is carved out of BEA-60K and never
touched by teacher labeling or student training; every other BEA-60K word
error is fair game. This mirrors the precedent already set by milestones
3-6 (`max_bea_typos: 20000` in their configs) -- this run simply lifts the cap
and uses teacher-generated labels instead of raw gold.

Subcommands:
  prepare-data    Extract BEA-60K word errors, split off the 100-sample
                  holdout (seed 1337), write the training pool.
  label-teacher   Load the 2B teacher (4-bit NF4 + M6 LoRA) and batch-generate
                  corrections over the training pool -> distillation targets.
  train           QLoRA fine-tune the 0.8B student on teacher labels, with a
                  round-by-round (per-epoch) accuracy gate against the 100
                  holdout; stops as soon as Acc@1 > 90% or a round budget is
                  exhausted.
  eval            Standalone Acc@1 eval of any adapter dir against any holdout.

Optimized-kernel stack (probed, not assumed): flash-attn if importable,
bf16 compute, 4-bit NF4 double-quant, paged_adamw_8bit, torch.compile probed
but PEFT+bitsandbytes generally rejects it (matches M4-M6 findings).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_PROMPT = ROOT / "prompts" / "direct_correct_v1.txt"
DEFAULT_BEA_DIR = ROOT / "data" / "bea60k"
DEFAULT_OUT_DIR = ROOT / "data" / "distill"
DEFAULT_HOLDOUT_N = 100
DEFAULT_SEED = 1337

TEACHER_BASE = "Qwen/Qwen3.5-2B"
TEACHER_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
TEACHER_ADAPTER_DIR = ROOT / "artifacts" / "spell_slm_m6" / "qwen35_2b_direct_qlora"

STUDENT_BASE = "Qwen/Qwen3.5-0.8B"
STUDENT_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_prompt(template: str, sentence: str) -> str:
    return template.replace("{{SENTENCE}}", sentence)


def apply_chat(tokenizer, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return user_text + "\nAnswer:"


def probe_kernels() -> dict:
    info = {"flash_attn": False, "notes": []}
    try:
        import flash_attn  # noqa: F401

        info["flash_attn"] = True
        info["notes"].append(f"flash_attn={getattr(flash_attn, '__version__', '?')}")
    except Exception as e:  # noqa: BLE001
        info["notes"].append(f"flash_attn missing: {e}")
    return info


def attn_impl_kwargs(kernel_info: dict) -> dict:
    return {"attn_implementation": "flash_attention_2"} if kernel_info.get("flash_attn") else {}


# ---------------------------------------------------------------------------
# prepare-data
# ---------------------------------------------------------------------------


def build_bea_examples(bea_dir: Path) -> list[dict]:
    from spelling_reranker.bea60k import extract_word_errors, load_bea_pairs

    pairs = load_bea_pairs(bea_dir)
    errors = extract_word_errors(pairs)
    rows = []
    for i, e in enumerate(errors):
        sentence = f"{e['context_before']}<TYPO>{e['typo']}</TYPO>{e['context_after']}"
        rows.append(
            {
                "error_index": i,
                "typo": e["typo"],
                "gold": e["gold"],
                "sentence": sentence,
                "noisy_sentence": e["noisy_sentence"],
                "clean_sentence": e["clean_sentence"],
            }
        )
    return rows


def split_holdout(rows: list[dict], seed: int, n_holdout: int) -> tuple[list[dict], list[dict]]:
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    holdout_idx = set(order[:n_holdout])
    holdout = [rows[i] for i in order[:n_holdout]]
    pool = [rows[i] for i in range(len(rows)) if i not in holdout_idx]
    return holdout, pool


def cmd_prepare_data(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    rows = build_bea_examples(args.bea_dir)
    print(f"extracted {len(rows)} BEA-60K word errors from {args.bea_dir}", flush=True)
    holdout, pool = split_holdout(rows, args.seed, args.holdout_n)
    if args.max_train is not None:
        pool = pool[: args.max_train]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "holdout_100.json").write_text(json.dumps(holdout, indent=2) + "\n")
    write_jsonl(args.out_dir / "train_pool.jsonl", pool)
    meta = {
        "seed": args.seed,
        "n_total": len(rows),
        "n_holdout": len(holdout),
        "n_pool": len(pool),
        "bea_dir": str(args.bea_dir),
    }
    (args.out_dir / "prepare_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)
    return 0


# ---------------------------------------------------------------------------
# label-teacher
# ---------------------------------------------------------------------------


def load_4bit_with_adapter(model_id: str, revision: str, adapter_dir: Path | None, dtype, kernel_info: dict):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    rev_kwargs = {"revision": revision} if revision else {}
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb,
        device_map={"": 0} if torch.cuda.is_available() else None,
        torch_dtype=dtype,
        **attn_impl_kwargs(kernel_info),
        **rev_kwargs,
    )
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    return model


def batch_generate(model, tokenizer, prompts: list[str], batch_size: int, max_new_tokens: int) -> list[str]:
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    outputs: list[str] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
        if torch.cuda.is_available():
            enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = gen[:, enc["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        outputs.extend(t.strip().split()[0].strip(" \"'.,;:") if t.strip() else "" for t in texts)
        if (start // batch_size) % 20 == 0:
            print(f"  labeled {start + len(chunk)}/{len(prompts)}", flush=True)
    return outputs


def cmd_label_teacher(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoTokenizer

    kernel_info = probe_kernels()
    print(f"kernels={json.dumps(kernel_info)}", flush=True)
    print(f"cuda={torch.cuda.is_available()}", flush=True)

    template = args.prompt.read_text(encoding="utf-8")
    pool = load_jsonl(args.train_pool)
    print(f"pool={len(pool)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_BASE, trust_remote_code=True, revision=TEACHER_REVISION)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    t0 = time.time()
    model = load_4bit_with_adapter(TEACHER_BASE, TEACHER_REVISION, args.adapter_dir, dtype, kernel_info)
    print(f"teacher loaded in {time.time() - t0:.1f}s", flush=True)

    prompts = [apply_chat(tokenizer, render_prompt(template, row["sentence"])) for row in pool]
    t0 = time.time()
    preds = batch_generate(model, tokenizer, prompts, args.batch_size, args.max_new_tokens)
    elapsed = time.time() - t0
    print(f"labeled {len(preds)} examples in {elapsed:.1f}s ({len(preds) / max(elapsed, 1e-6):.1f}/s)", flush=True)

    labeled = []
    agree = 0
    for row, pred in zip(pool, preds):
        target = pred or row["gold"]
        if target.casefold() == row["gold"].casefold():
            agree += 1
        labeled.append(
            {
                "user_text": render_prompt(template, row["sentence"]),
                "target": target,
                "gold": row["gold"],
                "typo": row["typo"],
                "teacher_gold_agreement": target.casefold() == row["gold"].casefold(),
            }
        )
    write_jsonl(args.out, labeled)
    meta = {
        "n_labeled": len(labeled),
        "teacher_base": TEACHER_BASE,
        "teacher_revision": TEACHER_REVISION,
        "teacher_adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "teacher_gold_agreement_rate": agree / max(len(labeled), 1),
        "elapsed_s": elapsed,
        "kernels": kernel_info,
    }
    (args.out.parent / "label_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)
    return 0


# ---------------------------------------------------------------------------
# eval (shared by train's accuracy gate and standalone `eval` subcommand)
# ---------------------------------------------------------------------------


def eval_accuracy(model, tokenizer, template: str, holdout: list[dict], batch_size: int) -> dict:
    prompts = [apply_chat(tokenizer, render_prompt(template, row["sentence"])) for row in holdout]
    preds = batch_generate(model, tokenizer, prompts, batch_size, max_new_tokens=5)
    predictions = []
    n_exact = 0
    n_casefold = 0
    for row, pred in zip(holdout, preds):
        exact = pred == row["gold"]
        casefold = pred.casefold() == row["gold"].casefold()
        n_exact += int(exact)
        n_casefold += int(casefold)
        predictions.append({**row, "pred": pred, "exact": exact, "casefold": casefold})
    n = len(holdout)
    return {
        "n": n,
        "acc1_exact": n_exact / n,
        "acc1_casefold": n_casefold / n,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# train (QLoRA student, per-epoch accuracy gate)
# ---------------------------------------------------------------------------


def build_datasets(train_rows, tokenizer, max_len: int):
    from datasets import Dataset

    def encode_row(row: dict) -> dict:
        prompt = apply_chat(tokenizer, row["user_text"])
        answer = str(row["target"]).strip()
        eos = tokenizer.eos_token or ""
        full = prompt + answer + eos
        tok_prompt = tokenizer(prompt, add_special_tokens=False)
        tok_full = tokenizer(full, add_special_tokens=False, truncation=True, max_length=max_len)
        input_ids = tok_full["input_ids"]
        labels = list(input_ids)
        prompt_len = min(len(tok_prompt["input_ids"]), len(labels))
        for i in range(prompt_len):
            labels[i] = -100
        if all(x == -100 for x in labels) and labels:
            labels[-1] = input_ids[-1]
        return {"input_ids": input_ids, "attention_mask": tok_full["attention_mask"], "labels": labels}

    print("tokenizing training rows...", flush=True)
    return Dataset.from_list([encode_row(r) for r in train_rows])


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


def attach_fresh_lora(model, lora_cfg: dict, gradient_checkpointing: bool):
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(
            lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        ),
        bias="none",
    )
    return get_peft_model(model, peft_config)


def cmd_train(args: argparse.Namespace) -> int:
    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    random.seed(args.seed)
    kernel_info = probe_kernels()
    print(f"kernels={json.dumps(kernel_info)}", flush=True)
    print(f"cuda={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    train_rows = load_jsonl(args.train_labeled)
    holdout = json.loads(args.holdout.read_text())
    template = args.prompt.read_text(encoding="utf-8")
    print(f"train={len(train_rows)} holdout={len(holdout)} seed={args.seed}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_BASE, trust_remote_code=True, revision=STUDENT_REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    train_ds = build_datasets(train_rows, tokenizer, args.max_seq_length)

    model = load_4bit_with_adapter(STUDENT_BASE, STUDENT_REVISION, None, dtype, kernel_info)
    model.config.use_cache = False
    model = attach_fresh_lora(model, {"r": args.lora_r, "alpha": args.lora_alpha}, args.gradient_checkpointing)
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.out_dir / "distill_progress.jsonl"
    progress_path.write_text("")

    steps_per_epoch = max(1, len(train_ds) // (args.per_device_batch * args.grad_accum))
    checkpoint = None
    all_loss_logs: list[dict] = []
    best_acc = -1.0
    best_round = -1
    t_start = time.time()

    for round_i in range(1, args.max_rounds + 1):
        print(f"\n=== round {round_i}/{args.max_rounds} (1 epoch, ~{steps_per_epoch} steps) ===", flush=True)
        training_args = TrainingArguments(
            output_dir=str(args.out_dir / "trainer_out"),
            per_device_train_batch_size=args.per_device_batch,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            num_train_epochs=1.0,
            logging_steps=20,
            save_steps=max(1, steps_per_epoch),
            save_total_limit=2,
            bf16=torch.cuda.is_available(),
            fp16=False,
            report_to=[],
            seed=args.seed,
            remove_unused_columns=False,
            dataloader_num_workers=2,
            optim="paged_adamw_8bit",
            warmup_ratio=0.03 if round_i == 1 else 0.0,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            data_collator=PadCollator(tokenizer.pad_token_id),
        )
        t0 = time.time()
        trainer.train(resume_from_checkpoint=checkpoint)
        round_elapsed = time.time() - t0
        checkpoint = None  # each round is its own fresh Trainer walking the same LoRA weights forward

        round_loss_logs = [
            {"round": round_i, "step": int(x["step"]), "loss": float(x["loss"])}
            for x in (trainer.state.log_history or [])
            if "loss" in x
        ]
        all_loss_logs.extend(round_loss_logs)

        print(f"round {round_i} eval on {len(holdout)} holdout examples...", flush=True)
        model.eval()
        model.config.use_cache = True
        tokenizer.padding_side = "left"
        result = eval_accuracy(model, tokenizer, template, holdout, args.eval_batch_size)
        model.config.use_cache = False
        model.train()
        tokenizer.padding_side = "right"

        entry = {
            "round": round_i,
            "elapsed_s": round_elapsed,
            "cumulative_elapsed_s": time.time() - t_start,
            "train_loss": float(trainer.state.log_history[-1].get("loss")) if trainer.state.log_history else None,
            "acc1_exact": result["acc1_exact"],
            "acc1_casefold": result["acc1_casefold"],
            "n_holdout": result["n"],
        }
        print(f"PROGRESS {json.dumps(entry)}", flush=True)
        with progress_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

        if result["acc1_exact"] > best_acc:
            best_acc = result["acc1_exact"]
            best_round = round_i
            save_model = trainer.model
            if hasattr(save_model, "_orig_mod"):
                save_model = save_model._orig_mod
            save_model.save_pretrained(str(args.out_dir))
            tokenizer.save_pretrained(str(args.out_dir))
            (args.out_dir / "predictions_holdout.jsonl").write_text(
                "\n".join(json.dumps(p) for p in result["predictions"]) + "\n"
            )

        if result["acc1_exact"] > args.target_acc:
            print(f"TARGET REACHED: acc1_exact={result['acc1_exact']:.3f} > {args.target_acc} at round {round_i}", flush=True)
            break
    else:
        print(f"round budget exhausted; best acc1_exact={best_acc:.3f} at round {best_round}", flush=True)

    (args.out_dir / "loss_curve.json").write_text(json.dumps(all_loss_logs, indent=2) + "\n")
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None
    meta = {
        "task": "distill_2b_to_0p8b_qlora",
        "teacher_base": TEACHER_BASE,
        "teacher_adapter_dir": str(TEACHER_ADAPTER_DIR),
        "student_base": STUDENT_BASE,
        "student_revision": STUDENT_REVISION,
        "seed": args.seed,
        "n_train": len(train_rows),
        "n_holdout": len(holdout),
        "target_acc": args.target_acc,
        "best_round": best_round,
        "best_acc1_exact": best_acc,
        "rounds_run": round_i,
        "max_rounds": args.max_rounds,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha},
        "train": {
            "per_device_train_batch_size": args.per_device_batch,
            "gradient_accumulation_steps": args.grad_accum,
            "effective_batch": args.per_device_batch * args.grad_accum,
            "lr": args.lr,
            "gradient_checkpointing": args.gradient_checkpointing,
            "max_seq_length": args.max_seq_length,
        },
        "kernels": kernel_info,
        "peak_vram_gb": peak_vram_gb,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "elapsed_s": time.time() - t_start,
    }
    (args.out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)
    print(f"DONE best adapters -> {args.out_dir} (acc1_exact={best_acc:.3f})", flush=True)
    return 0 if best_acc > args.target_acc else 2


# ---------------------------------------------------------------------------
# standalone eval
# ---------------------------------------------------------------------------


def cmd_eval(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoTokenizer

    kernel_info = probe_kernels()
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_BASE, trust_remote_code=True, revision=STUDENT_REVISION)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = load_4bit_with_adapter(STUDENT_BASE, STUDENT_REVISION, args.adapter_dir, dtype, kernel_info)
    template = args.prompt.read_text(encoding="utf-8")
    holdout = json.loads(args.holdout.read_text())
    result = eval_accuracy(model, tokenizer, template, holdout, args.eval_batch_size)
    metrics = {
        "n": result["n"],
        "acc1_exact": result["acc1_exact"],
        "acc1_casefold": result["acc1_casefold"],
        "adapter_dir": str(args.adapter_dir),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2) + "\n")
    (args.out.parent / "predictions.jsonl").write_text(
        "\n".join(json.dumps(p) for p in result["predictions"]) + "\n"
    )
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("prepare-data")
    pd.add_argument("--bea-dir", type=Path, default=DEFAULT_BEA_DIR)
    pd.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    pd.add_argument("--seed", type=int, default=DEFAULT_SEED)
    pd.add_argument("--holdout-n", type=int, default=DEFAULT_HOLDOUT_N)
    pd.add_argument("--max-train", type=int, default=None)
    pd.set_defaults(func=cmd_prepare_data)

    lt = sub.add_parser("label-teacher")
    lt.add_argument("--train-pool", type=Path, default=DEFAULT_OUT_DIR / "train_pool.jsonl")
    lt.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR / "train_labeled.jsonl")
    lt.add_argument("--adapter-dir", type=Path, default=TEACHER_ADAPTER_DIR)
    lt.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    lt.add_argument("--batch-size", type=int, default=32)
    lt.add_argument("--max-new-tokens", type=int, default=5)
    lt.set_defaults(func=cmd_label_teacher)

    tr = sub.add_parser("train")
    tr.add_argument("--train-labeled", type=Path, default=DEFAULT_OUT_DIR / "train_labeled.jsonl")
    tr.add_argument("--holdout", type=Path, default=DEFAULT_OUT_DIR / "holdout_100.json")
    tr.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    tr.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "spell_slm_m7" / "qwen35_0_8b_distill_qlora")
    tr.add_argument("--seed", type=int, default=DEFAULT_SEED)
    tr.add_argument("--target-acc", type=float, default=0.90)
    tr.add_argument("--max-rounds", type=int, default=4)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--per-device-batch", type=int, default=8)
    tr.add_argument("--grad-accum", type=int, default=4)
    tr.add_argument("--eval-batch-size", type=int, default=16)
    tr.add_argument("--max-seq-length", type=int, default=256)
    tr.add_argument("--lora-r", type=int, default=16)
    tr.add_argument("--lora-alpha", type=int, default=32)
    tr.add_argument("--gradient-checkpointing", action="store_true", default=False)
    tr.set_defaults(func=cmd_train)

    ev = sub.add_parser("eval")
    ev.add_argument("--adapter-dir", type=Path, required=True)
    ev.add_argument("--holdout", type=Path, default=DEFAULT_OUT_DIR / "holdout_100.json")
    ev.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    ev.add_argument("--eval-batch-size", type=int, default=16)
    ev.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR / "eval_metrics.json")
    ev.set_defaults(func=cmd_eval)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
