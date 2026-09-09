"""Frozen ModernBERT encoder and Hunspell first-ten selector heads.

The encoder is never trained: every parameter stays frozen, the module stays in
eval mode, and pooled features are detached before they reach a selector.
Serialization uses the tokenizer's existing separator token and offset maps —
no new special embeddings.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from spelling_reranker.byte_encoding import nfc
from spelling_reranker.candidates import FROZEN_CANDIDATE_SLOTS, first_ten_pool, gold_index
from spelling_reranker.dataset import _as_optional_str
from spelling_reranker.serialization import _truncate_context

DEFAULT_ENCODER_ID = "answerdotai/ModernBERT-base"
#: Hugging Face commit resolved from answerdotai/ModernBERT-base HEAD on 2026-09-09.
DEFAULT_ENCODER_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
DEFAULT_MAX_LENGTH = 512
DEFAULT_HIDDEN_SIZE = 768
SERIALIZATION_VERSION = "frozen-modernbert-v1"
POOLING_NAME = "mean_last_layer"
N_SPELLING_FEATURES = 4
ENCODER_FEATURE_BLOCKS = 5  # c_i, t, x, c_i*t, |c_i-t|

MLP_INPUT_DIM = DEFAULT_HIDDEN_SIZE * ENCODER_FEATURE_BLOCKS + FROZEN_CANDIDATE_SLOTS + N_SPELLING_FEATURES
SCALAR_INPUT_DIM = FROZEN_CANDIDATE_SLOTS + N_SPELLING_FEATURES


def selector_input_dim(hidden_size: int = DEFAULT_HIDDEN_SIZE, n_candidates: int = FROZEN_CANDIDATE_SLOTS) -> int:
    return hidden_size * ENCODER_FEATURE_BLOCKS + n_candidates + N_SPELLING_FEATURES


class FrozenSerializationError(ValueError):
    """Required typo/candidate segments do not fit in max_length."""


@dataclass
class FrozenSerializedExample:
    input_ids: list[int]
    attention_mask: list[int]
    offset_mapping: list[tuple[int, int]]
    typo_token_indices: list[int]
    context_token_indices: list[int]
    candidate_token_indices: list[list[int]]
    candidate_valid: list[bool]
    candidates: list[str | None]
    typo: str
    gold_index: int | None = None
    truncated_context: bool = False
    serialization_failed: bool = False
    scalars: list[list[float]] = field(default_factory=list)
    text: str = ""
    seq_len: int = 0

    def __post_init__(self) -> None:
        self.seq_len = len(self.input_ids)


@dataclass
class ScalarScaler:
    """Train-only numeric scaling for the two continuous spelling features."""

    length_mean: float = 0.0
    length_std: float = 1.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ScalarScaler":
        if not data:
            return cls()
        return cls(
            length_mean=float(data.get("length_mean", 0.0)),
            length_std=float(data.get("length_std", 1.0) or 1.0),
        )

    def apply_numpy(self, scalars: np.ndarray) -> np.ndarray:
        out = np.array(scalars, dtype=np.float32, copy=True)
        std = self.length_std if self.length_std > 1e-6 else 1.0
        out[..., 1] = (out[..., 1] - self.length_mean) / std
        return out


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def spelling_features(typo: str, candidate: str | None) -> list[float]:
    """Four gold-free spelling features. Missing candidates are zeros."""
    if candidate is None or candidate == "":
        return [0.0, 0.0, 0.0, 0.0]
    t = nfc(typo)
    c = nfc(candidate)
    denom = max(len(t), len(c), 1)
    edit = levenshtein(t, c) / denom
    length_delta = float(len(c) - len(t))
    exact = 1.0 if c == t else 0.0
    casefold = 1.0 if c.casefold() == t.casefold() else 0.0
    return [edit, length_delta, exact, casefold]


def row_first_ten_candidates(row: Any) -> list[str | None]:
    """First ten stored Hunspell slots. Never backfills from cand_10+."""
    get = row.get if hasattr(row, "get") else lambda key, default=None: row[key] if key in row else default
    out: list[str | None] = []
    for i in range(FROZEN_CANDIDATE_SLOTS):
        out.append(_as_optional_str(get(f"cand_{i}")))
    return out


def _join_segments(parts: Sequence[str], sep: str) -> tuple[str, list[tuple[int, int]]]:
    chunks: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    for i, part in enumerate(parts):
        if i:
            chunks.append(sep)
            pos += len(sep)
        start = pos
        chunks.append(part)
        pos += len(part)
        spans.append((start, pos))
    return "".join(chunks), spans


def _token_indices_for_span(
    offset_mapping: Sequence[tuple[int, int]],
    char_start: int,
    char_end: int,
) -> list[int]:
    """Map a character span onto tokenizer offsets. Special tokens are (0, 0)."""
    if char_end <= char_start:
        return []
    indices: list[int] = []
    for i, (start, end) in enumerate(offset_mapping):
        if end <= start:
            continue
        if start < char_end and end > char_start:
            indices.append(i)
    return indices


def _encode_with_offsets(tokenizer, text: str) -> dict[str, Any]:
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    ids = list(encoded["input_ids"])
    offsets = [tuple(o) for o in encoded["offset_mapping"]]
    return {
        "input_ids": ids,
        "attention_mask": list(encoded.get("attention_mask") or [1] * len(ids)),
        "offset_mapping": offsets,
    }


def serialize_frozen_example(
    context_before: str,
    typo: str,
    context_after: str,
    candidates: Sequence[str | None],
    *,
    tokenizer,
    max_length: int = DEFAULT_MAX_LENGTH,
    gold_index_value: int | None = None,
) -> FrozenSerializedExample:
    """One sequence: left, marked typo, right, then first-ten candidate spans.

    Reserves typo + candidate tokens first and truncates context symmetrically.
    A candidate is never silently truncated; overflow is a serialization failure.
    """
    sep = tokenizer.sep_token
    if not sep:
        raise FrozenSerializationError("tokenizer has no sep_token; refusing to add new specials")

    typo_n = nfc(typo)
    left = nfc(context_before)
    right = nfc(context_after)
    padded: list[str | None] = list(candidates)[:FROZEN_CANDIDATE_SLOTS]
    if len(padded) < FROZEN_CANDIDATE_SLOTS:
        padded = padded + [None] * (FROZEN_CANDIDATE_SLOTS - len(padded))
    cand_text = [nfc(c) if c else "" for c in padded]
    valid = [bool(c) for c in cand_text]
    scalars = [spelling_features(typo_n, c if c else None) for c in cand_text]

    def _parts(left_s: str, right_s: str) -> list[str]:
        return [left_s, typo_n, right_s, *cand_text]

    def _build(left_s: str, right_s: str) -> tuple[str, list[tuple[int, int]]]:
        return _join_segments(_parts(left_s, right_s), sep)

    empty_text, _ = _build("", "")
    empty_enc = _encode_with_offsets(tokenizer, empty_text)
    if len(empty_enc["input_ids"]) > max_length:
        return _failed_example(typo_n, padded, valid, scalars, gold_index_value)

    full_text, _ = _build(left, right)
    full_enc = _encode_with_offsets(tokenizer, full_text)
    truncated = False
    if len(full_enc["input_ids"]) > max_length:
        truncated = True
        lo, hi = 0, len(left) + len(right)
        best_left, best_right = "", ""
        while lo <= hi:
            mid = (lo + hi) // 2
            l_chars, r_chars, _ = _truncate_context(list(left), list(right), mid)
            cand_l, cand_r = "".join(l_chars), "".join(r_chars)
            text_m, _ = _build(cand_l, cand_r)
            n_tok = len(_encode_with_offsets(tokenizer, text_m)["input_ids"])
            if n_tok <= max_length:
                best_left, best_right = cand_l, cand_r
                lo = mid + 1
            else:
                hi = mid - 1
        left, right = best_left, best_right
        full_text, _ = _build(left, right)
        full_enc = _encode_with_offsets(tokenizer, full_text)
        if len(full_enc["input_ids"]) > max_length:
            return _failed_example(typo_n, padded, valid, scalars, gold_index_value)

    text, spans = _build(left, right)
    encoded = full_enc if text == full_text else _encode_with_offsets(tokenizer, text)
    ids = encoded["input_ids"]
    if len(ids) > max_length:
        return _failed_example(typo_n, padded, valid, scalars, gold_index_value)

    offsets = encoded["offset_mapping"]
    left_span, typo_span, right_span = spans[0], spans[1], spans[2]
    cand_spans = spans[3 : 3 + FROZEN_CANDIDATE_SLOTS]
    typo_idx = _token_indices_for_span(offsets, *typo_span)
    context_idx = _token_indices_for_span(offsets, *left_span) + _token_indices_for_span(offsets, *right_span)
    cand_idx = [_token_indices_for_span(offsets, *span) for span in cand_spans]
    for i, (is_valid, positions) in enumerate(zip(valid, cand_idx)):
        if is_valid and not positions:
            # A required candidate produced no tokens and was not reserved — fail.
            return _failed_example(typo_n, padded, valid, scalars, gold_index_value)

    return FrozenSerializedExample(
        input_ids=ids,
        attention_mask=encoded["attention_mask"],
        offset_mapping=offsets,
        typo_token_indices=typo_idx,
        context_token_indices=context_idx,
        candidate_token_indices=cand_idx,
        candidate_valid=valid,
        candidates=list(padded),
        typo=typo_n,
        gold_index=gold_index_value,
        truncated_context=truncated,
        serialization_failed=False,
        scalars=scalars,
        text=text,
    )


def _failed_example(
    typo: str,
    padded: list[str | None],
    valid: list[bool],
    scalars: list[list[float]],
    gold_index_value: int | None,
) -> FrozenSerializedExample:
    return FrozenSerializedExample(
        input_ids=[],
        attention_mask=[],
        offset_mapping=[],
        typo_token_indices=[],
        context_token_indices=[],
        candidate_token_indices=[[] for _ in range(FROZEN_CANDIDATE_SLOTS)],
        candidate_valid=valid,
        candidates=list(padded),
        typo=typo,
        gold_index=gold_index_value,
        truncated_context=False,
        serialization_failed=True,
        scalars=scalars,
        text="",
    )


def pad_frozen_batch(examples: Sequence[FrozenSerializedExample], *, pad_id: int) -> dict[str, np.ndarray]:
    if not examples:
        raise ValueError("empty batch")
    lengths = [max(ex.seq_len, 1) if not ex.serialization_failed else 1 for ex in examples]
    length = int(max(lengths))
    batch = len(examples)
    n_cand = FROZEN_CANDIDATE_SLOTS
    token_ids = np.full((batch, length), pad_id, dtype=np.int64)
    attention = np.zeros((batch, length), dtype=np.int64)
    typo_mask = np.zeros((batch, length), dtype=np.int8)
    context_mask = np.zeros((batch, length), dtype=np.int8)
    cand_masks = np.zeros((batch, n_cand, length), dtype=np.int8)
    cand_valid = np.zeros((batch, n_cand), dtype=np.int8)
    gold = np.full(batch, -1, dtype=np.int64)
    failed = np.zeros(batch, dtype=np.int8)
    truncated = np.zeros(batch, dtype=np.int8)
    scalars = np.zeros((batch, n_cand, N_SPELLING_FEATURES), dtype=np.float32)

    for i, ex in enumerate(examples):
        failed[i] = int(ex.serialization_failed)
        truncated[i] = int(ex.truncated_context)
        if ex.gold_index is not None:
            gold[i] = int(ex.gold_index)
        cand_valid[i, : len(ex.candidate_valid)] = np.asarray(ex.candidate_valid, dtype=np.int8)
        if ex.scalars:
            scalars[i, : len(ex.scalars)] = np.asarray(ex.scalars, dtype=np.float32)
        if ex.serialization_failed or not ex.input_ids:
            continue
        n = min(ex.seq_len, length)
        token_ids[i, :n] = ex.input_ids[:n]
        attention[i, :n] = ex.attention_mask[:n] if ex.attention_mask else 1
        for pos in ex.typo_token_indices:
            if 0 <= pos < n:
                typo_mask[i, pos] = 1
        for pos in ex.context_token_indices:
            if 0 <= pos < n:
                context_mask[i, pos] = 1
        for ci, positions in enumerate(ex.candidate_token_indices):
            for pos in positions:
                if 0 <= pos < n:
                    cand_masks[i, ci, pos] = 1
    return {
        "input_ids": token_ids,
        "attention_mask": attention,
        "typo_mask": typo_mask,
        "context_mask": context_mask,
        "candidate_masks": cand_masks,
        "candidate_valid": cand_valid,
        "gold_index": gold,
        "serialization_failed": failed,
        "truncated_context": truncated,
        "scalars": scalars,
    }


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(hidden.dtype).unsqueeze(-1)
    denom = weights.sum(dim=-2)
    summed = (hidden * weights).sum(dim=-2)
    zero = torch.zeros_like(summed)
    return torch.where(denom > 0, summed / denom.clamp(min=1e-6), zero)


def pool_hidden_states(
    hidden: torch.Tensor,
    typo_mask: torch.Tensor,
    context_mask: torch.Tensor,
    candidate_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean-pool last-layer states into x (context), t (typo), c_i (candidates)."""
    context = masked_mean(hidden, context_mask)
    typo = masked_mean(hidden, typo_mask)
    weights = candidate_masks.to(hidden.dtype)
    denom = weights.sum(dim=2).clamp(min=1e-6).unsqueeze(-1)
    candidates = torch.bmm(weights, hidden) / denom
    empty = candidate_masks.sum(dim=2) == 0
    candidates = candidates.masked_fill(empty.unsqueeze(-1), 0.0)
    return context, typo, candidates


def assemble_selector_features(
    context: torch.Tensor,
    typo: torch.Tensor,
    candidates: torch.Tensor,
    scalars: torch.Tensor,
    *,
    include_encoder: bool = True,
    scaler: ScalarScaler | None = None,
) -> torch.Tensor:
    """Rebuild the per-candidate feature vector. Interactions are not cached."""
    batch, n_cand, hidden = candidates.shape
    if scaler is not None:
        std = scaler.length_std if scaler.length_std > 1e-6 else 1.0
        scaled = scalars.clone()
        scaled[..., 1] = (scaled[..., 1] - scaler.length_mean) / std
    else:
        scaled = scalars
    rank = torch.eye(n_cand, device=candidates.device, dtype=scaled.dtype)
    rank = rank.unsqueeze(0).expand(batch, n_cand, n_cand)
    if not include_encoder:
        return torch.cat([rank, scaled], dim=-1)
    typo_exp = typo.unsqueeze(1).expand_as(candidates)
    ctx_exp = context.unsqueeze(1).expand_as(candidates)
    return torch.cat(
        [
            candidates,
            typo_exp,
            ctx_exp,
            candidates * typo_exp,
            (candidates - typo_exp).abs(),
            rank,
            scaled,
        ],
        dim=-1,
    )


def mask_invalid_logits(logits: torch.Tensor, candidate_valid: torch.Tensor) -> torch.Tensor:
    invalid = candidate_valid == 0
    return logits.masked_fill(invalid, torch.finfo(logits.dtype).min)


def freeze_encoder(module: nn.Module) -> None:
    """Freeze every encoder parameter, including embeddings and norms."""
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


def encoder_requires_grad(module: nn.Module) -> bool:
    return any(bool(p.requires_grad) for p in module.parameters())


def encoder_param_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().float().numpy().tobytes())
    return digest.hexdigest()


def preferred_encoder_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def load_tokenizer(model_id: str = DEFAULT_ENCODER_ID, revision: str = DEFAULT_ENCODER_REVISION):
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    # Existing separators only — never add untrained specials.
    return tokenizer


def load_frozen_backbone(
    model_id: str = DEFAULT_ENCODER_ID,
    revision: str = DEFAULT_ENCODER_REVISION,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    device = device or torch.device("cpu")
    dtype = dtype or preferred_encoder_dtype(device)
    encoder = AutoModel.from_pretrained(model_id, revision=revision, torch_dtype=dtype)
    encoder.to(device)
    freeze_encoder(encoder)
    encoder.eval()
    return encoder


class FrozenEncoder(nn.Module):
    """Tokenizer + frozen backbone. Encoder remains eval even if .train() is called."""

    def __init__(
        self,
        *,
        tokenizer=None,
        encoder: nn.Module | None = None,
        model_id: str = DEFAULT_ENCODER_ID,
        revision: str = DEFAULT_ENCODER_REVISION,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.revision = revision
        self.max_length = int(max_length)
        self.device = device or torch.device("cpu")
        self.compute_dtype = dtype or preferred_encoder_dtype(self.device)
        self.tokenizer = tokenizer if tokenizer is not None else load_tokenizer(model_id, revision)
        if encoder is None:
            encoder = load_frozen_backbone(
                model_id, revision, device=self.device, dtype=self.compute_dtype
            )
        else:
            encoder.to(self.device)
            freeze_encoder(encoder)
        self.encoder = encoder
        freeze_encoder(self.encoder)

    def train(self, mode: bool = True) -> "FrozenEncoder":
        super().train(mode)
        self.encoder.eval()
        freeze_encoder(self.encoder)
        return self

    def eval(self) -> "FrozenEncoder":
        super().eval()
        self.encoder.eval()
        return self

    @torch.no_grad()
    def extract_from_serialized(
        self, examples: Sequence[FrozenSerializedExample]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        freeze_encoder(self.encoder)
        self.encoder.eval()
        pad_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)
        padded = pad_frozen_batch(examples, pad_id=pad_id)
        input_ids = torch.from_numpy(padded["input_ids"]).to(self.device)
        attention = torch.from_numpy(padded["attention_mask"]).to(self.device)
        typo_mask = torch.from_numpy(padded["typo_mask"]).to(self.device)
        context_mask = torch.from_numpy(padded["context_mask"]).to(self.device)
        cand_masks = torch.from_numpy(padded["candidate_masks"]).to(self.device)
        failed = torch.from_numpy(padded["serialization_failed"]).to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.compute_dtype,
            enabled=self.device.type == "cuda" and self.compute_dtype in (torch.float16, torch.bfloat16),
        ):
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention).last_hidden_state
        hidden = hidden.float()
        context, typo, candidates = pool_hidden_states(hidden, typo_mask, context_mask, cand_masks)
        zero = torch.zeros_like(context)
        fail_mask = failed.bool().unsqueeze(-1)
        context = torch.where(fail_mask, zero, context)
        typo = torch.where(fail_mask, zero, typo)
        candidates = torch.where(fail_mask.unsqueeze(1), torch.zeros_like(candidates), candidates)
        extras = {
            "candidate_valid": torch.from_numpy(padded["candidate_valid"]).to(self.device),
            "gold_index": torch.from_numpy(padded["gold_index"]).to(self.device),
            "serialization_failed": failed,
            "truncated_context": torch.from_numpy(padded["truncated_context"]).to(self.device),
            "scalars": torch.from_numpy(padded["scalars"]).to(self.device),
        }
        return context.detach(), typo.detach(), candidates.detach(), extras

    def serialize(
        self,
        context_before: str,
        typo: str,
        context_after: str,
        candidates: Sequence[str | None],
        gold_index_value: int | None = None,
    ) -> FrozenSerializedExample:
        return serialize_frozen_example(
            context_before,
            typo,
            context_after,
            candidates,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            gold_index_value=gold_index_value,
        )


def _mlp(in_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(256, 64),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(64, 1),
    )


class SelectorHead(nn.Module):
    """H1 scalar MLP, H2 linear, or H3 encoder MLP. Shared across candidates."""

    def __init__(
        self,
        arm: str,
        *,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        n_candidates: int = FROZEN_CANDIDATE_SLOTS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if arm not in {"scalar", "linear", "mlp"}:
            raise ValueError(f"unknown selector arm {arm!r}")
        self.arm = arm
        self.hidden_size = int(hidden_size)
        self.n_candidates = int(n_candidates)
        self.encoder_dim = selector_input_dim(self.hidden_size, self.n_candidates)
        self.scalar_dim = self.n_candidates + N_SPELLING_FEATURES
        if arm == "scalar":
            self.in_dim = self.scalar_dim
            self.net = _mlp(self.in_dim, dropout)
        elif arm == "linear":
            self.in_dim = self.encoder_dim
            self.net = nn.Linear(self.in_dim, 1)
        else:
            self.in_dim = self.encoder_dim
            self.net = _mlp(self.in_dim, dropout)

    def uses_encoder_features(self) -> bool:
        return self.arm != "scalar"

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class FrozenSelector(nn.Module):
    """Head on top of a frozen encoder. `.train()` does not un-eval the encoder."""

    def __init__(self, encoder: FrozenEncoder, head: SelectorHead, scaler: ScalarScaler | None = None) -> None:
        super().__init__()
        self.backbone = encoder
        self.head = head
        self.scaler = scaler or ScalarScaler()
        freeze_encoder(self.backbone.encoder)

    def train(self, mode: bool = True) -> "FrozenSelector":
        super().train(mode)
        self.backbone.encoder.eval()
        freeze_encoder(self.backbone.encoder)
        self.head.train(mode)
        return self

    def forward_from_pooled(
        self,
        context: torch.Tensor,
        typo: torch.Tensor,
        candidates: torch.Tensor,
        scalars: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> torch.Tensor:
        features = assemble_selector_features(
            context,
            typo,
            candidates,
            scalars,
            include_encoder=self.head.uses_encoder_features(),
            scaler=self.scaler,
        )
        logits = self.head(features)
        return mask_invalid_logits(logits, candidate_valid)

    def forward(self, examples: Sequence[FrozenSerializedExample]) -> torch.Tensor:
        context, typo, candidates, extras = self.backbone.extract_from_serialized(examples)
        return self.forward_from_pooled(
            context, typo, candidates, extras["scalars"], extras["candidate_valid"]
        )


def predict_index_from_logits(
    logits: torch.Tensor,
    candidate_valid: torch.Tensor,
    serialization_failed: torch.Tensor | None = None,
) -> torch.Tensor:
    """Argmax over valid slots; serialization failures fall back to candidate 0."""
    masked = mask_invalid_logits(logits, candidate_valid)
    pred = masked.argmax(dim=-1)
    if serialization_failed is not None:
        pred = torch.where(serialization_failed.bool(), torch.zeros_like(pred), pred)
    return pred


def nfc_pair(typo: str, gold: str) -> tuple[str, str]:
    return nfc(str(typo)), nfc(str(gold))


def select_subset_by_example_id(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    ordered = frame.sort_values("example_id", kind="mergesort")
    return ordered.head(int(n)).reset_index(drop=True)


def dpair_split(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    target: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Held-out-article rows whose NFC (typo, gold) pairs are absent from train."""
    train_pairs = {nfc_pair(t, g) for t, g in zip(train_frame["typo"], train_frame["gold"])}
    val = valid_frame.copy()
    val_pairs = [nfc_pair(t, g) for t, g in zip(val["typo"], val["gold"])]
    val["_pair"] = val_pairs
    disjoint = val[~val["_pair"].isin(train_pairs)].copy()
    pair_overlap = int(len(val) - len(disjoint))
    doc_overlap = 0
    if "source_document_id" in train_frame.columns and "source_document_id" in disjoint.columns:
        train_docs = set(train_frame["source_document_id"].astype(str))
        hit = disjoint["source_document_id"].astype(str).isin(train_docs)
        doc_overlap = int(hit.sum())
        disjoint = disjoint.loc[~hit].copy()
    disjoint = disjoint.sort_values("example_id", kind="mergesort")
    taken = disjoint.head(int(target)).reset_index(drop=True)
    report = {
        "train_pairs": len(train_pairs),
        "valid_rows": int(len(val)),
        "pair_overlap_dropped": pair_overlap,
        "document_overlap_dropped": doc_overlap,
        "dpair_available": int(len(disjoint)),
        "dpair_taken": int(len(taken)),
        "dpair_shortfall": max(0, int(target) - int(len(taken))),
    }
    return taken.drop(columns=["_pair"], errors="ignore"), report


def pair_document_overlap(
    train_frame: pd.DataFrame, other_frame: pd.DataFrame
) -> dict[str, int]:
    train_pairs = {nfc_pair(t, g) for t, g in zip(train_frame["typo"], train_frame["gold"])}
    other_pairs = [nfc_pair(t, g) for t, g in zip(other_frame["typo"], other_frame["gold"])]
    pair_overlap = sum(1 for p in other_pairs if p in train_pairs)
    doc_overlap = 0
    if "source_document_id" in train_frame.columns and "source_document_id" in other_frame.columns:
        train_docs = set(train_frame["source_document_id"].astype(str))
        doc_overlap = int(other_frame["source_document_id"].astype(str).isin(train_docs).sum())
    return {"pair_overlap": int(pair_overlap), "document_overlap": doc_overlap}


def git_sha(root: Path | None = None) -> str | None:
    cwd = str(root) if root is not None else None
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()
    except Exception:
        return None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def cache_key(
    *,
    encoder_id: str,
    encoder_revision: str,
    tokenizer_revision: str,
    data_hash: str,
    max_length: int,
    pooling: str = POOLING_NAME,
    precision: str = "fp16",
    serialization: str = SERIALIZATION_VERSION,
    n_candidates: int = FROZEN_CANDIDATE_SLOTS,
    code_sha: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "encoder_id": encoder_id,
        "encoder_revision": encoder_revision,
        "tokenizer_revision": tokenizer_revision,
        "data_hash": data_hash,
        "serialization": serialization,
        "max_length": int(max_length),
        "pooling": pooling,
        "precision": precision,
        "n_candidates": int(n_candidates),
        "first_ten_policy": "raw_slice",
        "code_sha": code_sha,
    }
    if extra:
        payload.update(extra)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["key_sha256"] = sha256_text(canonical)
    return payload


def load_feature_shards(cache_dir: Path, split: str) -> dict[str, np.ndarray]:
    shard_dir = Path(cache_dir) / split
    paths = sorted(shard_dir.glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no shards in {shard_dir}")
    parts = [dict(np.load(path, allow_pickle=True)) for path in paths]
    keys = parts[0].keys()
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}


def fit_scalar_scaler(scalars: np.ndarray, valid: np.ndarray) -> ScalarScaler:
    """Fit length-delta z-score on training rows only, using valid candidates."""
    mask = valid.astype(bool)
    if scalars.ndim != 3:
        raise ValueError("scalars must be [N, C, 4]")
    lengths = scalars[..., 1][mask]
    if lengths.size == 0:
        return ScalarScaler()
    mean = float(lengths.mean())
    std = float(lengths.std())
    if std < 1e-6:
        std = 1.0
    return ScalarScaler(length_mean=mean, length_std=std)
