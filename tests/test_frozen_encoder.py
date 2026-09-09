"""Frozen ModernBERT selector: freeze/eval, first-ten policy, overfit, splits, cache key."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cache_frozen_features import DummyBackbone, DummyTokenizer
from evaluate_frozen_selector import score_bundle
from spelling_reranker.byte_encoding import nfc
from spelling_reranker.candidates import FROZEN_CANDIDATE_SLOTS, first_ten_pool, gold_index
from spelling_reranker.frozen_encoder import (
    MLP_INPUT_DIM,
    FrozenEncoder,
    FrozenSelector,
    SelectorHead,
    assemble_selector_features,
    cache_key,
    dpair_split,
    encoder_param_digest,
    encoder_requires_grad,
    load_tokenizer,
    mask_invalid_logits,
    pair_document_overlap,
    predict_index_from_logits,
    row_first_ten_candidates,
    select_subset_by_example_id,
    serialize_frozen_example,
)
from spelling_reranker.model import count_parameters
from spelling_reranker.seed import seed_everything


def _frozen(max_length: int = 128) -> FrozenEncoder:
    return FrozenEncoder(
        tokenizer=DummyTokenizer(),
        encoder=DummyBackbone(),
        model_id="dummy",
        revision="dummy",
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_length=max_length,
    )


def test_mlp_parameter_count_is_about_one_million() -> None:
    n = count_parameters(SelectorHead("mlp", dropout=0.0))
    assert MLP_INPUT_DIM == 3854
    assert 1_000_000 <= n <= 1_010_000, n


def test_encoder_parameters_are_frozen_and_have_no_grads() -> None:
    backbone = _frozen()
    head = SelectorHead("mlp", dropout=0.0)
    model = FrozenSelector(backbone, head)
    digest = encoder_param_digest(backbone.encoder)
    assert encoder_requires_grad(backbone.encoder) is False
    model.train()
    assert backbone.encoder.training is False
    examples = []
    gold = []
    for i in range(4):
        cands = ["quick", "quirk", "quack", "quit"] + [None] * 6
        examples.append(
            backbone.serialize("left context ", "quik", " right", cands, gold_index_value=0)
        )
        gold.append(0)
    logits = model(examples)
    loss = F.cross_entropy(logits, torch.tensor(gold))
    loss.backward()
    for parameter in backbone.encoder.parameters():
        assert parameter.grad is None
        assert parameter.requires_grad is False
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    head_ids = {id(p) for p in head.parameters()}
    enc_ids = {id(p) for p in backbone.encoder.parameters()}
    assert opt_ids == head_ids
    assert opt_ids.isdisjoint(enc_ids)
    opt.step()
    assert encoder_param_digest(backbone.encoder) == digest
    assert backbone.encoder.training is False


def test_encoder_stays_eval_while_head_trains() -> None:
    backbone = _frozen()
    head = SelectorHead("linear", dropout=0.0)
    model = FrozenSelector(backbone, head)
    model.train(True)
    assert head.training is True
    assert backbone.encoder.training is False
    model.eval()
    model.train()
    assert backbone.encoder.training is False


def test_mlp_overfits_32_solvable_examples() -> None:
    seed_everything(1337)
    n, n_cand, hidden = 32, FROZEN_CANDIDATE_SLOTS, 768
    context = torch.randn(n, hidden)
    typo = torch.randn(n, hidden)
    candidates = torch.randn(n, n_cand, hidden) * 0.01
    gold = torch.arange(n) % 8
    for i in range(n):
        candidates[i, int(gold[i])] = 3.0
        candidates[i, int(gold[i]), i % hidden] = 8.0
    scalars = torch.zeros(n, n_cand, 4)
    valid = torch.ones(n, n_cand, dtype=torch.int64)
    valid[:, 8:] = 0
    head = SelectorHead("mlp", dropout=0.0)
    opt = torch.optim.AdamW(head.parameters(), lr=8e-3)
    last = None
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        features = assemble_selector_features(context, typo, candidates, scalars)
        logits = mask_invalid_logits(head(features), valid)
        loss = F.cross_entropy(logits, gold)
        loss.backward()
        opt.step()
        last = logits
    acc = float((last.argmax(dim=-1) == gold).float().mean())
    assert acc >= 0.95, acc


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_extract_skips_without_cuda_or_runs_on_gpu() -> None:
    device = torch.device("cuda")
    backbone = FrozenEncoder(
        tokenizer=DummyTokenizer(),
        encoder=DummyBackbone().to(device),
        model_id="dummy",
        revision="dummy",
        device=device,
        dtype=torch.float32,
        max_length=64,
    )
    ex = backbone.serialize("a ", "teh", " cat", ["the", "tea"] + [None] * 8, gold_index_value=0)
    x, t, c, extras = backbone.extract_from_serialized([ex])
    assert x.device.type == "cuda"
    assert extras["candidate_valid"][0, 0] == 1


def test_padded_candidates_never_win() -> None:
    logits = torch.tensor([[0.1, 0.2, 50.0, -1.0]])
    valid = torch.tensor([[1, 1, 0, 0]])
    masked = mask_invalid_logits(logits, valid)
    pred = predict_index_from_logits(masked, valid)
    assert int(pred.item()) != 2
    assert int(pred.item()) in (0, 1)
    features = torch.zeros(1, FROZEN_CANDIDATE_SLOTS, MLP_INPUT_DIM)
    features[0, 9] = 100.0
    valid10 = torch.zeros(1, FROZEN_CANDIDATE_SLOTS, dtype=torch.int64)
    valid10[0, :3] = 1
    head = SelectorHead("linear", dropout=0.0)
    with torch.no_grad():
        head.net.weight.zero_()
        head.net.bias.zero_()
        # Give slot 9 a huge incoming feature; the mask must still block it.
        logits = mask_invalid_logits(head(features), valid10)
    assert int(logits.argmax(dim=-1).item()) != 9


def test_first_ten_cannot_backfill_rank_eleven() -> None:
    raw = [
        "the",
        "teh",
        "the",  # duplicate of rank 0; a naive limit=10 filter would pull rank 11
        "tea",
        "ten",
        "tech",
        "then",
        "them",
        "they",
        "thaw",
        "GOLD",
        "gild",
    ]
    pool = first_ten_pool(raw)
    assert nfc("GOLD") not in [nfc(x) for x in pool]
    assert "GOLD" not in pool
    assert len(pool) <= 10
    # Length filtering of a first-ten item also must not backfill.
    long_item = "w" * 80
    raw2 = ["ok"] + [long_item] * 8 + ["also", "GOLD11"]
    pool2 = first_ten_pool(raw2, max_bytes=32)
    assert "GOLD11" not in pool2
    assert gold_index(pool2, "GOLD11") is None


def test_row_first_ten_never_reads_slot_eleven() -> None:
    row = {f"cand_{i}": f"c{i}" for i in range(16)}
    row["cand_10"] = "SHOULD_NOT_APPEAR"
    row["cand_11"] = "GOLD"
    cands = row_first_ten_candidates(row)
    assert len(cands) == 10
    assert "SHOULD_NOT_APPEAR" not in cands
    assert "GOLD" not in cands
    assert cands[-1] == "c9"


def test_pair_and_document_overlap_are_dropped_from_dpair() -> None:
    train = pd.DataFrame(
        {
            "example_id": ["a", "b"],
            "typo": ["teh", "recieve"],
            "gold": ["the", "receive"],
            "source_document_id": ["doc-1", "doc-2"],
        }
    )
    valid = pd.DataFrame(
        {
            "example_id": ["z", "y", "x", "w"],
            "typo": ["teh", "seperate", "occured", "teh"],
            "gold": ["the", "separate", "occurred", "the"],
            "source_document_id": ["doc-9", "doc-8", "doc-1", "doc-7"],
        }
    )
    taken, report = dpair_split(train, valid, target=20)
    pairs = {(nfc(t), nfc(g)) for t, g in zip(taken["typo"], taken["gold"])}
    assert ("teh", "the") not in pairs
    assert report["pair_overlap_dropped"] >= 1
    assert report["document_overlap_dropped"] >= 1
    overlap = pair_document_overlap(train, taken)
    assert overlap["pair_overlap"] == 0
    assert overlap["document_overlap"] == 0
    subset = select_subset_by_example_id(valid, 2)
    assert list(subset["example_id"]) == ["w", "x"]  # lexicographic


def test_cache_key_changes_when_policy_changes() -> None:
    base = dict(
        encoder_id="answerdotai/ModernBERT-base",
        encoder_revision="8949b909ec900327062f0ebf497f51aef5e6f0c8",
        tokenizer_revision="8949b909ec900327062f0ebf497f51aef5e6f0c8",
        data_hash="abc",
        max_length=512,
        pooling="mean_last_layer",
        precision="fp16",
        code_sha="deadbeef",
    )
    k1 = cache_key(**base)
    k2 = cache_key(**{**base, "max_length": 256})
    k3 = cache_key(**{**base, "pooling": "cls"})
    k4 = cache_key(**{**base, "data_hash": "abcd"})
    k5 = cache_key(**base)
    assert k1["key_sha256"] == k5["key_sha256"]
    assert k1["key_sha256"] != k2["key_sha256"]
    assert k1["key_sha256"] != k3["key_sha256"]
    assert k1["key_sha256"] != k4["key_sha256"]


def test_modernbert_tokenizer_does_not_add_specials() -> None:
    tok = load_tokenizer()
    n_before = len(tok)
    ex = serialize_frozen_example(
        "see the ",
        "the",
        " cat",
        ["the", "teh"] + [None] * 8,
        tokenizer=tok,
        gold_index_value=0,
    )
    assert len(tok) == n_before
    assert not ex.serialization_failed
    assert ex.typo_token_indices
    assert ex.candidate_token_indices[0]
    assert set(ex.typo_token_indices).isdisjoint(set(ex.candidate_token_indices[0]))
    # Tokens covering the typo should decode to the typo, not the context "the".
    typo_ids = [ex.input_ids[i] for i in ex.typo_token_indices]
    piece = tok.decode(typo_ids)
    assert "the" in piece.lower()
    tok = DummyTokenizer()
    ex = serialize_frozen_example(
        "see the ",
        "the",
        " cat",
        ["the", "teh"] + [None] * 8,
        tokenizer=tok,
        gold_index_value=0,
    )
    assert not ex.serialization_failed
    typo_chars = "".join(ex.text[s:e] for s, e in (ex.offset_mapping[i] for i in ex.typo_token_indices))
    assert "the" in typo_chars
    cand0 = "".join(ex.text[s:e] for s, e in (ex.offset_mapping[i] for i in ex.candidate_token_indices[0]))
    assert cand0 == "the"
    # Context "the" and candidate "the" occupy different token indices.
    assert set(ex.typo_token_indices).isdisjoint(ex.candidate_token_indices[0])
    ctx_tokens = set(ex.context_token_indices)
    assert ctx_tokens.isdisjoint(ex.typo_token_indices)


def test_unicode_and_punctuation_offsets() -> None:
    tok = DummyTokenizer()
    ex = serialize_frozen_example(
        "naïve ",
        "café",
        " shop.",
        ["café", "cafe"] + [None] * 8,
        tokenizer=tok,
        gold_index_value=0,
    )
    recovered = "".join(ex.text[s:e] for s, e in (ex.offset_mapping[i] for i in ex.typo_token_indices))
    assert nfc(recovered) == nfc("café")


def test_serialization_failures_stay_in_overall_denominator() -> None:
    n, hidden, n_cand = 5, 768, FROZEN_CANDIDATE_SLOTS
    bundle = {
        "x": np.zeros((n, hidden), np.float16),
        "t": np.zeros((n, hidden), np.float16),
        "c": np.zeros((n, n_cand, hidden), np.float16),
        "scalars": np.zeros((n, n_cand, 4), np.float32),
        "gold_index": np.array([0, 1, -1, 0, 2], dtype=np.int16),
        "valid": np.ones((n, n_cand), np.uint8),
        "solvable": np.array([1, 1, 0, 1, 1], np.uint8),
        "serialization_failed": np.array([0, 0, 0, 1, 0], np.uint8),
        "truncated_context": np.zeros(n, np.uint8),
        "example_id": np.array([f"e{i}" for i in range(n)], dtype=object),
        "source_document_id": np.array(["d"] * n, dtype=object),
    }
    # Hunspell-first baseline: pred=0. Correct when gold_index==0 and solvable.
    metrics = score_bundle(None, bundle, device=torch.device("cpu"), hunspell_baseline=True)
    assert metrics["n"] == 5
    assert metrics["serialization_failures"] == 1
    # Failed row still in the denominator.
    assert metrics["n"] == 5
    # gold 0,0,-1,0,2 → hunspell-first hits rows 0 and 3 (failed still predicts 0).
    assert metrics["overall_accuracy"] == pytest.approx(2 / 5)


def test_context_truncation_keeps_typo_and_candidates() -> None:
    tok = DummyTokenizer()
    left = "L" * 400
    right = "R" * 400
    ex = serialize_frozen_example(
        left, "typo", right, ["alpha", "beta"] + [None] * 8, tokenizer=tok, max_length=80
    )
    assert not ex.serialization_failed
    assert ex.truncated_context
    recovered_typo = "".join(ex.text[s:e] for s, e in (ex.offset_mapping[i] for i in ex.typo_token_indices))
    assert recovered_typo == "typo"
    cand0 = "".join(ex.text[s:e] for s, e in (ex.offset_mapping[i] for i in ex.candidate_token_indices[0]))
    assert cand0 == "alpha"
    assert ex.seq_len <= 80


def test_pathological_candidate_segment_fails_closed() -> None:
    tok = DummyTokenizer()
    monster = "x" * 500
    ex = serialize_frozen_example(
        "left ", "t", " right", [monster] + [None] * 9, tokenizer=tok, max_length=32
    )
    assert ex.serialization_failed
    pred = predict_index_from_logits(
        torch.zeros(1, FROZEN_CANDIDATE_SLOTS),
        torch.ones(1, FROZEN_CANDIDATE_SLOTS, dtype=torch.int64),
        torch.tensor([1]),
    )
    assert int(pred.item()) == 0


def _tiny_parquet_frame(
    n: int,
    prefix: str,
    *,
    gold_index: int = 0,
    doc: str | None = None,
    pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    default_pairs = [("teh", "the"), ("recieve", "receive")]
    pairs = pairs or default_pairs
    rows = []
    for i in range(n):
        typo, gold = pairs[i % len(pairs)]
        row = {
            "example_id": f"{prefix}-{i:04d}",
            "source": "fixture",
            "context_before": "the ",
            "typo": typo,
            "context_after": " cat",
            "gold": gold,
            "gold_index": gold_index,
            "corruption_type": "x",
            "source_document_id": doc or f"doc-{prefix}-{i // 5}",
            "original_sentence_hash": f"h{prefix}{i}",
        }
        for c in range(16):
            if c == 0:
                row[f"cand_{c}"] = gold
            elif c < 4:
                row[f"cand_{c}"] = f"alt{c}"
            else:
                row[f"cand_{c}"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def test_dummy_cache_and_head_train_smoke(tmp_path) -> None:
    train = _tiny_parquet_frame(40, "tr")
    valid = _tiny_parquet_frame(
        20, "va", doc="held-out", pairs=[("seperate", "separate"), ("occured", "occurred")]
    )
    train_path = tmp_path / "train.parquet"
    valid_path = tmp_path / "valid.parquet"
    train.to_parquet(train_path)
    valid.to_parquet(valid_path)
    cfg = tmp_path / "cfg.yaml"
    cache_dir = tmp_path / "cache"
    cfg.write_text(
        "\n".join(
            [
                "seed: 1337",
                "frozen: true",
                "encoder:",
                "  model_id: dummy",
                "  revision: dummy",
                "  max_length: 256",
                "  hidden_size: 768",
                "  extract_batch_size: 8",
                "  pooling: mean_last_layer",
                "  first_ten_policy: raw_slice",
                "data:",
                f"  train: {train_path}",
                f"  validation: {valid_path}",
                "  train_subset: 40",
                "  dpair_target: 20",
                "cache:",
                f"  dir: {cache_dir}",
                "  precision: fp16",
                "  shard_size: 16",
                "training:",
                "  learning_rate: 0.01",
                "  weight_decay: 0.01",
                "  batch_size: 8",
                "  epochs: 1",
                "  patience: 2",
                "  dropout: 0.0",
            ]
        )
        + "\n"
    )
    cache_script = ROOT / "scripts" / "cache_frozen_features.py"
    train_script = ROOT / "scripts" / "train_frozen_selector.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(cache_script),
            "--config",
            str(cfg),
            "--dummy-encoder",
            "--allow-cpu",
            "--smoke-examples",
            "40",
            "--cache-dir",
            str(cache_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    key_a = manifest["cache_key"]["key_sha256"]
    proc2 = subprocess.run(
        [
            sys.executable,
            str(train_script),
            "--config",
            str(cfg),
            "--arm",
            "mlp",
            "--allow-cpu",
            "--no-bea-gate",
            "--max-steps",
            "3",
            "--cache-dir",
            str(cache_dir),
            "--output-dir",
            str(tmp_path / "head"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc2.returncode == 0, proc2.stdout + proc2.stderr
    assert (tmp_path / "head" / "head.safetensors").is_file()
    # Policy change invalidates the cache key.
    other = cache_key(
        encoder_id="dummy",
        encoder_revision="dummy",
        tokenizer_revision="dummy",
        data_hash="changed",
        max_length=64,
    )
    assert other["key_sha256"] != key_a

