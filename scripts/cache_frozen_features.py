#!/usr/bin/env python3
"""Deterministic frozen-encoder feature cache (200k train + D-pair).

Caches only pooled x, t, c_i plus metadata/scalars. Interaction features are
rebuilt at train time. Large shards stay out of git; the manifest records the
cache key (encoder/tokenizer revision, data hash, serialization, max length,
pooling, precision, code SHA).
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spelling_reranker.candidates import FROZEN_CANDIDATE_SLOTS, first_ten_pool, gold_index
from spelling_reranker.config import load_yaml
from spelling_reranker.device import describe_cuda, select_training_device
from spelling_reranker.hunspell import default_engine
from spelling_reranker.frozen_encoder import (
    DEFAULT_ENCODER_ID,
    DEFAULT_ENCODER_REVISION,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_MAX_LENGTH,
    POOLING_NAME,
    SERIALIZATION_VERSION,
    FrozenEncoder,
    cache_key,
    dpair_split,
    git_sha,
    nfc_pair,
    preferred_encoder_dtype,
    row_first_ten_candidates,
    select_subset_by_example_id,
    sha256_file,
    sha256_text,
)
from spelling_reranker.seed import DEFAULT_SEED, seed_everything


def _rss_bytes() -> int:
    # ru_maxrss is kilobytes on Linux.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _hidden_size(encoder: FrozenEncoder) -> int:
    cfg = getattr(encoder.encoder, "config", None)
    if cfg is not None and getattr(cfg, "hidden_size", None):
        return int(cfg.hidden_size)
    return DEFAULT_HIDDEN_SIZE


class DummyBackbone(torch.nn.Module):
    """CPU-safe stand-in for tests and --dummy-encoder smoke runs."""

    def __init__(self, hidden: int = DEFAULT_HIDDEN_SIZE, vocab: int = 512) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.norm = torch.nn.LayerNorm(hidden)
        self.config = type("Cfg", (), {"hidden_size": hidden})()

    def forward(self, input_ids, attention_mask=None, **kwargs):
        clamped = input_ids.clamp(min=0, max=self.embed.num_embeddings - 1)
        hidden = self.norm(self.embed(clamped))
        return type("Out", (), {"last_hidden_state": hidden})()


class DummyTokenizer:
    """Character-level tokenizer with BERT-like specials. Tests only."""

    cls_token = "[CLS]"
    sep_token = "[SEP]"
    pad_token = "[PAD]"
    cls_token_id = 1
    sep_token_id = 2
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [3 + (ord(ch) % 200) for ch in text]
        if add_special_tokens:
            return [self.cls_token_id] + ids + [self.sep_token_id]
        return ids

    def __call__(
        self,
        text: str,
        return_offsets_mapping: bool = False,
        add_special_tokens: bool = True,
        truncation: bool = False,
        padding: bool = False,
    ) -> dict:
        body = [(3 + (ord(ch) % 200), (i, i + 1)) for i, ch in enumerate(text)]
        ids = [tok for tok, _ in body]
        offsets = [off for _, off in body]
        if add_special_tokens:
            ids = [self.cls_token_id] + ids + [self.sep_token_id]
            offsets = [(0, 0)] + offsets + [(0, 0)]
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "offset_mapping": offsets,
        }


def records_from_frame(frame: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    for _, row in frame.iterrows():
        cands = row_first_ten_candidates(row)
        gold = str(row["gold"])
        gi = gold_index(cands, gold)
        records.append(
            {
                "example_id": str(row["example_id"]),
                "source_document_id": str(row["source_document_id"]) if "source_document_id" in row else "",
                "context_before": str(row["context_before"]),
                "typo": str(row["typo"]),
                "context_after": str(row["context_after"]),
                "gold": gold,
                "candidates": cands,
                "gold_index": -1 if gi is None else int(gi),
                "solvable": gi is not None,
                "pair": nfc_pair(str(row["typo"]), gold),
            }
        )
    return records


def prepare_splits(cfg: dict, *, smoke_examples: int | None = None) -> dict[str, pd.DataFrame | dict]:
    data_cfg = cfg["data"]
    train_path = Path(data_cfg["train"])
    valid_path = Path(data_cfg["validation"])
    if not train_path.is_file():
        raise SystemExit(f"missing {train_path}; run scripts/build_training_data.py first")
    if not valid_path.is_file():
        raise SystemExit(f"missing {valid_path}; run scripts/build_training_data.py first")
    train = pd.read_parquet(train_path)
    valid = pd.read_parquet(valid_path)
    n_train = int(data_cfg.get("train_subset", 200_000))
    n_dpair = int(data_cfg.get("dpair_target", 20_000))
    if smoke_examples is not None:
        n_train = min(n_train, int(smoke_examples))
        n_dpair = min(n_dpair, max(1, int(smoke_examples) // 5))
    train_sub = select_subset_by_example_id(train, n_train)
    dpair, dpair_report = dpair_split(train_sub, valid, n_dpair)
    return {
        "train": train_sub,
        "dpair": dpair,
        "dpair_report": dpair_report,
        "train_hash": sha256_file(train_path),
        "valid_hash": sha256_file(valid_path),
        "train_ids_hash": sha256_text("\n".join(train_sub["example_id"].astype(str).tolist())),
        "dpair_ids_hash": sha256_text("\n".join(dpair["example_id"].astype(str).tolist())),
    }


def write_shard(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def extract_records(
    records: list[dict],
    encoder: FrozenEncoder,
    *,
    batch_size: int,
    hidden: int,
) -> dict[str, np.ndarray]:
    n = len(records)
    c = FROZEN_CANDIDATE_SLOTS
    x = np.zeros((n, hidden), dtype=np.float16)
    t = np.zeros((n, hidden), dtype=np.float16)
    cand = np.zeros((n, c, hidden), dtype=np.float16)
    scalars = np.zeros((n, c, 4), dtype=np.float32)
    gold = np.full(n, -1, dtype=np.int16)
    valid = np.zeros((n, c), dtype=np.uint8)
    solvable = np.zeros(n, dtype=np.uint8)
    failed = np.zeros(n, dtype=np.uint8)
    truncated = np.zeros(n, dtype=np.uint8)
    example_ids = np.array([r["example_id"] for r in records], dtype=object)
    docs = np.array([r.get("source_document_id", "") for r in records], dtype=object)

    for start in tqdm(range(0, n, batch_size), desc="extract", leave=False):
        chunk = records[start : start + batch_size]
        serialized = [
            encoder.serialize(
                rec["context_before"],
                rec["typo"],
                rec["context_after"],
                rec["candidates"],
                gold_index_value=rec["gold_index"] if rec["gold_index"] >= 0 else None,
            )
            for rec in chunk
        ]
        context, typo, candidates, extras = encoder.extract_from_serialized(serialized)
        end = start + len(chunk)
        x[start:end] = context.cpu().numpy().astype(np.float16, copy=False)
        t[start:end] = typo.cpu().numpy().astype(np.float16, copy=False)
        cand[start:end] = candidates.cpu().numpy().astype(np.float16, copy=False)
        scalars[start:end] = extras["scalars"].cpu().numpy().astype(np.float32, copy=False)
        gold[start:end] = extras["gold_index"].cpu().numpy().astype(np.int16, copy=False)
        valid[start:end] = extras["candidate_valid"].cpu().numpy().astype(np.uint8, copy=False)
        failed[start:end] = extras["serialization_failed"].cpu().numpy().astype(np.uint8, copy=False)
        truncated[start:end] = extras["truncated_context"].cpu().numpy().astype(np.uint8, copy=False)
        for i, rec in enumerate(chunk):
            solvable[start + i] = np.uint8(int(rec["solvable"] and not serialized[i].serialization_failed))
            if rec["gold_index"] >= 0:
                gold[start + i] = rec["gold_index"]
    return {
        "x": x,
        "t": t,
        "c": cand,
        "scalars": scalars,
        "gold_index": gold,
        "valid": valid,
        "solvable": solvable,
        "serialization_failed": failed,
        "truncated_context": truncated,
        "example_id": example_ids,
        "source_document_id": docs,
    }


def shard_arrays(payload: dict[str, np.ndarray], shard_size: int, out_dir: Path, split: str) -> list[dict]:
    n = int(payload["x"].shape[0])
    meta = []
    shard_i = 0
    for start in range(0, max(n, 1) if n else 0, shard_size):
        end = min(n, start + shard_size)
        shard = {k: v[start:end] for k, v in payload.items()}
        path = out_dir / split / f"shard_{shard_i:05d}.npz"
        write_shard(path, shard)
        meta.append({"path": str(path), "n": int(end - start), "sha256": sha256_file(path)})
        shard_i += 1
    return meta


def load_split_cache(cache_dir: Path, split: str) -> dict[str, np.ndarray]:
    shard_dir = cache_dir / split
    paths = sorted(shard_dir.glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no shards in {shard_dir}")
    parts: list[dict[str, np.ndarray]] = [dict(np.load(p, allow_pickle=True)) for p in paths]
    keys = parts[0].keys()
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}


def load_bea_pairs(bea_dir: Path) -> list[tuple[str, str]]:
    clean_path = bea_dir / "test.bea60k"
    noise_path = bea_dir / "test.bea60k.noise"
    if not clean_path.is_file() or not noise_path.is_file():
        raise FileNotFoundError(f"BEA files missing in {bea_dir}. Run scripts/download_bea60k.py")
    cleans = clean_path.read_text(encoding="utf-8", errors="replace").splitlines()
    noises = noise_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(cleans) != len(noises):
        raise ValueError(f"line count mismatch: clean={len(cleans)} noise={len(noises)}")
    return list(zip(noises, cleans))


def align_errors(noisy: str, clean: str) -> list[dict]:
    n_toks = noisy.split()
    c_toks = clean.split()
    matcher = SequenceMatcher(a=n_toks, b=c_toks, autojunk=False)
    errors: list[dict] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        n_span = n_toks[i1:i2]
        c_span = c_toks[j1:j2]
        if len(n_span) == 1 and len(c_span) == 1:
            left = " ".join(n_toks[:i1])
            right = " ".join(n_toks[i2:])
            if left:
                left += " "
            if right:
                right = " " + right
            errors.append(
                {
                    "typo": n_span[0],
                    "gold": c_span[0],
                    "context_before": left,
                    "context_after": right,
                    "noisy_sentence": noisy,
                    "clean_sentence": clean,
                }
            )
    return errors


def bea_records(bea_dir: Path, limit: int) -> list[dict]:
    pairs = load_bea_pairs(bea_dir)
    errors: list[dict] = []
    for noisy, clean in pairs:
        errors.extend(align_errors(noisy, clean))

    def _key(err: dict) -> str:
        raw = f"{err['typo']}\t{err['gold']}\t{err['context_before']}\t{err['context_after']}"
        return sha256_text(raw)

    ordered = sorted(errors, key=_key)[: int(limit)]
    engine = default_engine()
    records: list[dict] = []
    for i, err in enumerate(ordered):
        raw = engine.suggest_raw(err["typo"])
        pool = first_ten_pool(raw)
        padded: list[str | None] = list(pool) + [None] * (FROZEN_CANDIDATE_SLOTS - len(pool))
        gi = gold_index(padded, err["gold"])
        records.append(
            {
                "example_id": f"bea:{i:05d}:{_key(err)[:12]}",
                "source_document_id": f"bea:{sha256_text(err.get('noisy_sentence', ''))[:12]}",
                "context_before": err["context_before"],
                "typo": err["typo"],
                "context_after": err["context_after"],
                "gold": err["gold"],
                "candidates": padded[:FROZEN_CANDIDATE_SLOTS],
                "gold_index": -1 if gi is None else int(gi),
                "solvable": gi is not None,
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train_frozen_modernbert.yaml")
    parser.add_argument("--smoke-examples", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--dummy-encoder", action="store_true")
    parser.add_argument("--bea-dir", type=Path, default=ROOT / "data" / "bea60k")
    parser.add_argument("--bea-limit", type=int, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", DEFAULT_SEED)))
    enc_cfg = cfg.get("encoder", {})
    cache_cfg = cfg.get("cache", {})
    cache_dir = Path(args.cache_dir or cache_cfg.get("dir", "artifacts/frozen_cache"))
    shard_size = int(cache_cfg.get("shard_size", 10_000))
    extract_bs = int(enc_cfg.get("extract_batch_size", 16))
    max_length = int(enc_cfg.get("max_length", DEFAULT_MAX_LENGTH))
    model_id = str(enc_cfg.get("model_id", DEFAULT_ENCODER_ID))
    revision = str(enc_cfg.get("revision", DEFAULT_ENCODER_REVISION))

    if args.dummy_encoder or args.allow_cpu:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        device = select_training_device(args.device)
    print(f"device={device}{describe_cuda(device)}")

    splits = prepare_splits(cfg, smoke_examples=args.smoke_examples)
    train_frame: pd.DataFrame = splits["train"]  # type: ignore[assignment]
    dpair_frame: pd.DataFrame = splits["dpair"]  # type: ignore[assignment]
    print(f"train subset {len(train_frame):,}  d-pair {len(dpair_frame):,}  report={splits['dpair_report']}")

    if args.dummy_encoder:
        tokenizer = DummyTokenizer()
        backbone = DummyBackbone()
        encoder = FrozenEncoder(
            tokenizer=tokenizer,
            encoder=backbone,
            model_id="dummy",
            revision="dummy",
            device=device,
            dtype=torch.float32,
            max_length=max_length,
        )
        tokenizer_rev = "dummy"
        encoder_id = "dummy"
        encoder_rev = "dummy"
    else:
        dtype = preferred_encoder_dtype(device)
        encoder = FrozenEncoder(
            model_id=model_id,
            revision=revision,
            device=device,
            dtype=dtype,
            max_length=max_length,
        )
        tokenizer_rev = revision
        encoder_id = model_id
        encoder_rev = revision

    hidden = _hidden_size(encoder)
    code_sha = sha256_file(ROOT / "spelling_reranker" / "frozen_encoder.py")
    data_hash = sha256_text(
        json.dumps(
            {
                "train_parquet": splits["train_hash"],
                "valid_parquet": splits["valid_hash"],
                "train_ids": splits["train_ids_hash"],
                "dpair_ids": splits["dpair_ids_hash"],
                "smoke": args.smoke_examples,
            },
            sort_keys=True,
        )
    )
    key = cache_key(
        encoder_id=encoder_id,
        encoder_revision=encoder_rev,
        tokenizer_revision=tokenizer_rev,
        data_hash=data_hash,
        max_length=max_length,
        pooling=str(enc_cfg.get("pooling", POOLING_NAME)),
        precision=str(cache_cfg.get("precision", "fp16")),
        serialization=SERIALIZATION_VERSION,
        code_sha=code_sha,
        extra={"git_sha": git_sha(ROOT), "first_ten_policy": enc_cfg.get("first_ten_policy", "raw_slice")},
    )
    existing = cache_dir / "manifest.json"
    if existing.is_file():
        prev = json.loads(existing.read_text(encoding="utf-8"))
        if prev.get("cache_key", {}).get("key_sha256") == key["key_sha256"]:
            print(f"cache key matches {existing}; skipping extraction")
            return 0
        print("cache key changed; rebuilding")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    report: dict = {"splits": {}, "dpair_report": splits["dpair_report"]}

    for name, frame in (("train", train_frame), ("dpair", dpair_frame)):
        recs = records_from_frame(frame)
        t0 = time.time()
        payload = extract_records(recs, encoder, batch_size=extract_bs, hidden=hidden)
        shards = shard_arrays(payload, shard_size, cache_dir, name)
        dt = max(1e-6, time.time() - t0)
        report["splits"][name] = {
            "n": len(recs),
            "n_solvable": int(payload["solvable"].sum()),
            "n_failed": int(payload["serialization_failed"].sum()),
            "n_truncated": int(payload["truncated_context"].sum()),
            "examples_per_sec": len(recs) / dt,
            "shards": shards,
        }
        print(json.dumps(report["splits"][name], indent=2))

    if args.bea_limit:
        recs = bea_records(args.bea_dir, args.bea_limit)
        t0 = time.time()
        payload = extract_records(recs, encoder, batch_size=extract_bs, hidden=hidden)
        shards = shard_arrays(payload, shard_size, cache_dir, "bea")
        dt = max(1e-6, time.time() - t0)
        report["splits"]["bea"] = {
            "n": len(recs),
            "n_solvable": int(payload["solvable"].sum()),
            "n_failed": int(payload["serialization_failed"].sum()),
            "examples_per_sec": len(recs) / dt,
            "shards": shards,
            "limit": args.bea_limit,
        }

    peak_vram = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
    summary = {
        "cache_key": key,
        "cache_dir": str(cache_dir),
        "device": str(device),
        "hidden_size": hidden,
        "extract_batch_size": extract_bs,
        "duration_sec": time.time() - started,
        "peak_rss_bytes": _rss_bytes(),
        "peak_vram_bytes": peak_vram,
        "smoke_examples": args.smoke_examples,
        "dummy_encoder": bool(args.dummy_encoder),
        **report,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("cache_key", "duration_sec", "peak_rss_bytes", "peak_vram_bytes")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
