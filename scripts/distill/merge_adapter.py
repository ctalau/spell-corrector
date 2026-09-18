#!/usr/bin/env python3
"""Merge the student's LoRA into an fp16 base and write a GGUF-ready HF dir.

Two Qwen3.5 specifics are handled here rather than after conversion:

* the multi-token-prediction head (`mtp.*`, `mtp_num_hidden_layers: 1`) is
  dropped. llama.cpp does not convert those tensors but still counts the layer,
  which is what produced M5/M6's `block_count=25` against 24 real blocks and
  made the GGUF unloadable until its metadata was rewritten.
* the vision tower is dropped: this is a text-only corrector.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--revision", default="")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--keep-mtp", action="store_true")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    model = None
    last_err = None
    for loader in (AutoModelForImageTextToText, AutoModelForCausalLM):
        try:
            model = loader.from_pretrained(
                args.base_model,
                revision=args.revision or None,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="cpu",
            )
            print(f"loaded base via {loader.__name__}", flush=True)
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"loader {loader.__name__} failed: {exc}", flush=True)
    if model is None:
        raise SystemExit(f"could not load base: {last_err}")

    model = PeftModel.from_pretrained(model, str(args.adapter))
    model = model.merge_and_unload()
    print("adapters merged", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out_dir), safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(
        args.base_model, revision=args.revision or None, trust_remote_code=True
    )
    tok.save_pretrained(str(args.out_dir))

    if not args.keep_mtp:
        strip_mtp(args.out_dir)
    print(f"merged model written to {args.out_dir}", flush=True)
    return 0


def strip_mtp(out_dir: Path) -> None:
    """Remove the MTP head from both the weights and the config."""
    import json as _json

    from safetensors.torch import load_file, save_file

    cfg_path = out_dir / "config.json"
    cfg = _json.loads(cfg_path.read_text(encoding="utf-8"))
    for holder in (cfg, cfg.get("text_config") or {}):
        holder.pop("mtp_num_hidden_layers", None)
        holder.pop("mtp_use_dedicated_embeddings", None)
        holder.pop("num_nextn_predict_layers", None)
    cfg_path.write_text(_json.dumps(cfg, indent=2), encoding="utf-8")

    index_path = out_dir / "model.safetensors.index.json"
    shards = []
    if index_path.is_file():
        index = _json.loads(index_path.read_text(encoding="utf-8"))
        shards = sorted({v for v in index["weight_map"].values()})
    else:
        shards = [p.name for p in out_dir.glob("*.safetensors")]

    removed = []
    total = 0
    new_map: dict[str, str] = {}
    for shard in shards:
        path = out_dir / shard
        tensors = load_file(str(path))
        keep = {}
        for name, tensor in tensors.items():
            if name.startswith("mtp.") or ".mtp." in name:
                removed.append(name)
                continue
            keep[name] = tensor
            new_map[name] = shard
            total += tensor.numel() * tensor.element_size()
        save_file(keep, str(path), metadata={"format": "pt"})
    if index_path.is_file():
        index_path.write_text(
            _json.dumps({"metadata": {"total_size": total}, "weight_map": new_map}, indent=2),
            encoding="utf-8",
        )
    print(f"stripped {len(removed)} MTP tensors", flush=True)
    (out_dir / "mtp_strip.json").write_text(
        json.dumps({"removed_tensors": removed}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
