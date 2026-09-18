#!/usr/bin/env python3
"""Verify (and if needed repair) the block_count / MTP metadata of a GGUF.

M5 and M6 both hit the same Qwen3.5 quirk: conversion writes
`block_count = 25` and a `nextn_predict_layers` key while only `blk.0`..`blk.23`
exist, and llama.cpp then refuses the file. `merge_adapter.py` removes the MTP
head before conversion so this should now be clean; this script proves it, and
`--fix` rewrites the metadata in place (new file) when it is not.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def read(path: Path):
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    blocks = set()
    for tensor in reader.tensors:
        m = re.match(r"blk\.(\d+)\.", tensor.name)
        if m:
            blocks.add(int(m.group(1)))
    fields = {}
    for name, field in reader.fields.items():
        try:
            fields[name] = field.contents()
        except Exception:  # noqa: BLE001
            fields[name] = "<unreadable>"
    return reader, blocks, fields


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--fix", type=Path, default=None, help="write a repaired copy here")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    reader, blocks, fields = read(args.gguf)
    n_blocks = (max(blocks) + 1) if blocks else 0
    arch = fields.get("general.architecture")
    block_key = f"{arch}.block_count"
    recurrent_key = f"{arch}.attention.recurrent_layers"
    nextn_key = f"{arch}.nextn_predict_layers"
    declared = fields.get(block_key)
    recurrent = fields.get(recurrent_key)
    report = {
        "file": str(args.gguf),
        "architecture": arch,
        "tensor_blocks": n_blocks,
        "declared_block_count": declared,
        "recurrent_layers_len": len(recurrent) if isinstance(recurrent, list) else None,
        "has_nextn_predict_layers": nextn_key in fields,
    }
    consistent = (
        declared == n_blocks
        and not report["has_nextn_predict_layers"]
        and (report["recurrent_layers_len"] in (None, n_blocks))
    )
    report["consistent"] = bool(consistent)
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if consistent or not args.fix:
        return 0 if consistent else 2

    # ---- repair: copy every field and tensor, with the three fixes applied --
    import gguf
    from gguf import GGUFReader, GGUFValueType, GGUFWriter

    writer = GGUFWriter(str(args.fix), arch)
    for name, field in reader.fields.items():
        if name in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"):
            continue
        if name == nextn_key:
            continue
        value = field.contents()
        if name == block_key:
            value = n_blocks
        if name == recurrent_key and isinstance(value, list):
            value = value[:n_blocks]
        types = list(field.types)
        if types and types[0] == GGUFValueType.ARRAY:
            writer.add_array(name, value)
        else:
            writer.add_key_value(name, value, types[0])
    for tensor in reader.tensors:
        writer.add_tensor_info(
            tensor.name,
            list(tensor.data.shape),
            tensor.data.dtype,
            tensor.data.nbytes,
            tensor.tensor_type,
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data)
    writer.close()
    print(f"repaired copy written to {args.fix}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
