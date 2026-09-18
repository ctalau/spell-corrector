#!/usr/bin/env python3
"""Teacher pass for M7: dump top-K logits of the 2B Q4 corrector.

The teacher is `Qwen/Qwen3.5-2B` loaded in 4-bit NF4 with the milestone-6 QLoRA
adapters — the exact configuration that scored 91% Acc@1 on the frozen BEA-100.

For every training row it runs one teacher-forced forward pass over
`prompt + gold answer` and stores, for each answer position, the top-K token ids
and their logits. The student then trains against those distributions without
the teacher ever being resident again: distillation costs one teacher epoch,
not one teacher forward per student step.

Outputs (numpy memmaps, `--out-dir`):
    topk_ids.npy   int32   [N, L, K]
    topk_logits.npy float16 [N, L, K]
    answer_ids.npy int32   [N, L]
    answer_len.npy int32   [N]
    prompt_len.npy int32   [N]
    meta.json              tokenizer fingerprint, teacher-forced agreement, config
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import encode_example, load_jsonl, tokenizer_fingerprint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-jsonl", type=Path, default=ROOT / "data/distill/train.jsonl")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/distill/teacher")
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--revision", default="15852e8c16360a2fea060d615a32b45270f8a8fc")
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--max-answer-tokens", type=int, default=10)
    ap.add_argument("--max-seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    rows = load_jsonl(args.train_jsonl)
    if args.limit:
        rows = rows[: args.limit]
    print(f"teacher rows={len(rows)}", flush=True)

    tok = AutoTokenizer.from_pretrained(
        args.base_model, revision=args.revision, trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    fingerprint = tokenizer_fingerprint(tok)
    print(f"tokenizer_fingerprint={fingerprint}", flush=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    attn = "sdpa"
    try:
        import flash_attn  # noqa: F401

        attn = "flash_attention_2"
    except Exception:  # noqa: BLE001
        pass

    # Qwen3.5 ships as a conditional-generation (VLM) class; the image-text
    # loader is tried first, as in the M4-M6 milestone scripts.
    model = None
    last_err = None
    for loader in (AutoModelForImageTextToText, AutoModelForCausalLM):
        for attn_try in ([attn, "sdpa"] if attn != "sdpa" else ["sdpa"]):
            try:
                model = loader.from_pretrained(
                    args.base_model,
                    revision=args.revision,
                    trust_remote_code=True,
                    quantization_config=bnb,
                    device_map={"": 0},
                    torch_dtype=torch.bfloat16,
                    attn_implementation=attn_try,
                )
                attn = attn_try
                print(f"loaded teacher via {loader.__name__} attn={attn_try}", flush=True)
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                print(f"loader {loader.__name__}/{attn_try} failed: {exc}", flush=True)
        if model is not None:
            break
    if model is None:
        raise SystemExit(f"could not load teacher: {last_err}")
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(args.adapter_dir))
    model.eval()

    enc = [encode_example(tok, r, args.max_seq_len) for r in rows]
    n = len(enc)
    L, K = args.max_answer_tokens, args.topk
    args.out_dir.mkdir(parents=True, exist_ok=True)

    topk_ids = np.lib.format.open_memmap(
        args.out_dir / "topk_ids.npy", mode="w+", dtype=np.int32, shape=(n, L, K)
    )
    topk_logits = np.lib.format.open_memmap(
        args.out_dir / "topk_logits.npy", mode="w+", dtype=np.float16, shape=(n, L, K)
    )
    answer_ids = np.zeros((n, L), dtype=np.int32)
    answer_len = np.zeros((n,), dtype=np.int32)
    prompt_len = np.zeros((n,), dtype=np.int32)

    order = sorted(range(n), key=lambda i: len(enc[i]["input_ids"]))  # pack similar lengths
    pad_id = tok.pad_token_id
    agree_tokens = agree_total = 0
    exact_rows = 0
    t0 = time.time()
    done = 0
    with torch.inference_mode():
        for start in range(0, n, args.batch_size):
            chunk = order[start : start + args.batch_size]
            batch = [enc[i] for i in chunk]
            width = max(len(b["input_ids"]) for b in batch)
            ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for j, b in enumerate(batch):
                ids[j, : len(b["input_ids"])] = torch.tensor(b["input_ids"])
                mask[j, : len(b["input_ids"])] = 1
            logits = model(
                input_ids=ids.to(model.device), attention_mask=mask.to(model.device)
            ).logits.float()

            for j, (i, b) in enumerate(zip(chunk, batch)):
                a_ids = b["answer_ids"][:L]
                p_len = b["prompt_len"]
                if not a_ids:
                    continue
                pos = torch.arange(p_len - 1, p_len - 1 + len(a_ids), device=logits.device)
                sel = logits[j].index_select(0, pos)  # [len(a_ids), V]
                vals, idx = torch.topk(sel, K, dim=-1)
                topk_ids[i, : len(a_ids)] = idx.cpu().numpy().astype(np.int32)
                topk_logits[i, : len(a_ids)] = vals.cpu().numpy().astype(np.float16)
                answer_ids[i, : len(a_ids)] = np.array(a_ids, dtype=np.int32)
                answer_len[i] = len(a_ids)
                prompt_len[i] = p_len
                hits = (idx[:, 0].cpu() == torch.tensor(a_ids)).sum().item()
                agree_tokens += hits
                agree_total += len(a_ids)
                exact_rows += int(hits == len(a_ids))
                done += 1

            if (start // args.batch_size) % 25 == 0:
                seen = min(start + args.batch_size, n)
                rate = seen / max(1e-6, time.time() - t0)
                print(
                    f"teacher {seen}/{n} rows  {rate:.1f} rows/s  "
                    f"eta={(n - seen) / max(rate, 1e-6) / 60:.1f} min  "
                    f"token_agree={agree_tokens / max(1, agree_total):.4f}",
                    flush=True,
                )

    topk_ids.flush()
    topk_logits.flush()
    np.save(args.out_dir / "answer_ids.npy", answer_ids)
    np.save(args.out_dir / "answer_len.npy", answer_len)
    np.save(args.out_dir / "prompt_len.npy", prompt_len)
    meta = {
        "n_rows": n,
        "rows_with_answer": done,
        "topk": K,
        "max_answer_tokens": L,
        "max_seq_len": args.max_seq_len,
        "base_model": args.base_model,
        "revision": args.revision,
        "adapter_dir": str(args.adapter_dir),
        "quantization": "nf4 double-quant, bf16 compute",
        "attn_implementation": attn,
        "tokenizer_fingerprint": fingerprint,
        "teacher_forced_token_agreement": agree_tokens / max(1, agree_total),
        "teacher_forced_exact_rows": exact_rows / max(1, done),
        "wall_seconds": time.time() - t0,
        "train_jsonl": str(args.train_jsonl),
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
