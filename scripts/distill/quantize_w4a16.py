#!/usr/bin/env python3
"""Quantize the merged 0.8B corrector to a vLLM-native weight-only format.

Why not reuse the deployed Q4_K_M GGUF: that file exists for llama.cpp on a
CPU, and vLLM's GGUF path is a compatibility shim, not a fast one. On an
Ampere card the format that actually has optimized kernels is compressed-tensors
W4A16 (group 128), which vLLM dispatches to Marlin. This produces that.

Calibration uses the same synthetic `<TYPO>` prompts as the throughput
benchmark -- `data/wikipedia_misspellings.txt`, never BEA-60K, which is locked.

The model is a hybrid: three linear-attention layers for every full-attention
one, plus a multi-token-prediction head. `--ignore` therefore defaults to
leaving the MTP head, the vision tower and the linear-attention projections in
fp16, and quantizing the MLPs and the full-attention projections -- where the
weights and the GEMM time actually are. Pass `--ignore` to move that line.

    python scripts/distill/quantize_w4a16.py --model /workspace/models/fp16 \\
        --output /workspace/models/w4a16 --scheme W4A16 --samples 256
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_IGNORE = [
    "lm_head",
    "re:.*mtp.*",
    "re:.*visual.*",
    "re:.*vision.*",
    "re:.*linear_attn.*",
]


def build_calibration_texts(tokenizer, count: int) -> list[str]:
    """Chat-templated corrector prompts, the shape the served traffic has."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_bench_prompts", ROOT / "scripts/bench_spell_throughput.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sentences = module.build_sentences(count, seed=7)
    texts = []
    for sentence in sentences:
        messages = [{"role": "user", "content": module.build_user_prompt(sentence)}]
        texts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        )
    return texts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="merged fp16 checkpoint directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scheme", default="W4A16", help="compressed-tensors scheme, e.g. W4A16 / W8A8")
    parser.add_argument("--algorithm", choices=("gptq", "awq", "rtn"), default="gptq")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--ignore", action="append", default=None, help="module pattern to leave in fp16")
    parser.add_argument("--dump-modules", type=Path, default=None)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ignore = args.ignore if args.ignore else DEFAULT_IGNORE
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map="cuda:0")

    if args.dump_modules:
        names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
        args.dump_modules.parent.mkdir(parents=True, exist_ok=True)
        args.dump_modules.write_text(json.dumps(names, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {len(names)} Linear module names to {args.dump_modules}")

    texts = build_calibration_texts(tokenizer, args.samples)

    encoded = [
        tokenizer(t, truncation=True, max_length=args.max_seq_len, add_special_tokens=False)
        for t in texts
    ]
    rows = [{"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]} for e in encoded]
    try:  # `datasets` is llmcompressor's expected input, but it is a heavy
        from datasets import Dataset  # dependency and this needs one column of ints

        dataset = Dataset.from_list(rows)
    except ImportError:
        print("datasets is not installed; passing the calibration rows directly")
        dataset = rows

    from llmcompressor import oneshot

    if args.algorithm == "gptq":
        from llmcompressor.modifiers.quantization import GPTQModifier

        modifier = GPTQModifier(targets="Linear", scheme=args.scheme, ignore=ignore)
    elif args.algorithm == "awq":
        from llmcompressor.modifiers.awq import AWQModifier

        modifier = AWQModifier(targets="Linear", scheme=args.scheme, ignore=ignore)
    else:
        from llmcompressor.modifiers.quantization import QuantizationModifier

        modifier = QuantizationModifier(targets="Linear", scheme=args.scheme, ignore=ignore)

    oneshot(
        model=model,
        dataset=dataset,
        recipe=modifier,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=len(dataset),
    )

    output = Path(args.output)
    model.save_pretrained(str(output), save_compressed=True)
    tokenizer.save_pretrained(str(output))
    total = sum(p.stat().st_size for p in output.rglob("*.safetensors"))
    print(json.dumps({"output": str(output), "scheme": args.scheme, "algorithm": args.algorithm,
                      "ignore": ignore, "weights_bytes": total}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
