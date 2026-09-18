#!/usr/bin/env python3
"""Score a direct spelling corrector (HF + PEFT, 4-bit NF4) on a BEA split.

Accuracy comes from batched greedy generation; latency comes from a separate
single-request pass over the first `--latency-n` rows, because a batched number
is not the number a server would show.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import apply_chat, load_jsonl, normalize_prediction, write_jsonl  # noqa: E402


def load_model(base_model: str, revision: str | None, adapter: Path | None, torch):
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
    model = None
    last_err = None
    for loader in (AutoModelForImageTextToText, AutoModelForCausalLM):
        for attn_try in ([attn, "sdpa"] if attn != "sdpa" else ["sdpa"]):
            try:
                model = loader.from_pretrained(
                    base_model,
                    revision=revision or None,
                    trust_remote_code=True,
                    quantization_config=bnb,
                    device_map={"": 0},
                    torch_dtype=torch.bfloat16,
                    attn_implementation=attn_try,
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        if model is not None:
            break
    if model is None:
        raise SystemExit(f"could not load {base_model}: {last_err}")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    model.config.use_cache = True
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--revision", default="")
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--out-metrics", type=Path, required=True)
    ap.add_argument("--out-predictions", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--max-new-tokens", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--latency-n", type=int, default=100)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    rows = load_jsonl(args.split)
    if args.limit:
        rows = rows[: args.limit]
    tok = AutoTokenizer.from_pretrained(
        args.base_model, revision=args.revision or None, trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = load_model(args.base_model, args.revision, args.adapter, torch)

    preds = []
    exact = casefold = 0
    t0 = time.time()
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start : start + args.batch_size]
            enc = tok(
                [apply_chat(tok, r["user_text"]) for r in chunk],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
            gen = tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            for r, raw in zip(chunk, gen):
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
            if start % (args.batch_size * 10) == 0:
                print(f"eval {start + len(chunk)}/{len(rows)}", flush=True)
    wall = time.time() - t0

    latencies = []
    if args.latency_n:
        with torch.inference_mode():
            for r in rows[: args.latency_n]:
                enc = tok(
                    apply_chat(tok, r["user_text"]), return_tensors="pt", add_special_tokens=False
                )
                enc = {k: v.to(model.device) for k, v in enc.items()}
                t1 = time.time()
                model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
                latencies.append(time.time() - t1)

    n = max(1, len(rows))
    metrics = {
        "label": args.label or args.split.stem,
        "split": str(args.split),
        "n": len(rows),
        "acc@1_exact": exact / n,
        "acc@1_casefold": casefold / n,
        "base_model": args.base_model,
        "revision": args.revision,
        "adapter": str(args.adapter) if args.adapter else None,
        "quantization": "nf4 double-quant, bf16 compute",
        "decode": {"greedy": True, "max_new_tokens": args.max_new_tokens},
        "batched_wall_seconds": round(wall, 2),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if latencies:
        ordered = sorted(latencies)
        metrics["latency_s_batch1"] = {
            "n": len(ordered),
            "p50": statistics.median(ordered),
            "p90": ordered[int(0.9 * (len(ordered) - 1))],
            "mean": sum(ordered) / len(ordered),
        }
    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.out_predictions:
        write_jsonl(args.out_predictions, preds)
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
