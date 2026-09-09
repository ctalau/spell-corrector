"""Serialize typo + context + Hunspell candidates into one byte sequence."""

from __future__ import annotations

from dataclasses import dataclass, field

from spelling_reranker.byte_encoding import (
    CAND_END_ID,
    CAND_IDS,
    CLS_ID,
    CTX_END_ID,
    CTX_START_ID,
    LANG_EN_ID,
    N_CANDIDATE_SLOTS,
    PAD_ID,
    TYPO_END_ID,
    TYPO_START_ID,
    nfc,
    text_to_byte_ids,
)

#: A theoretically full pool (16 x MAX_CANDIDATE_BYTES + 38 structural tokens)
#: would not fit here, but measured over 4,000 real BEA-60K errors the worst
#: case is 169 bytes -- Hunspell returns short lists of short words. The budget
#: is therefore set from observed data rather than the worst case, and
#: `predict_indices` degrades to candidate 0 if a freak input ever exceeds it.
DEFAULT_MAX_SEQ_LEN = 448
N_CANDIDATES = N_CANDIDATE_SLOTS
MAX_CANDIDATE_BYTES = 32


class PathologicalExampleError(ValueError):
    """Raised when candidates + typo cannot fit in max_seq_len."""


@dataclass
class SerializedExample:
    token_ids: list[int]
    typo_positions: list[int]
    candidate_positions: list[list[int]]
    candidate_valid: list[bool]
    gold_index: int | None = None
    truncated_context: bool = False
    seq_len: int = 0

    def __post_init__(self) -> None:
        self.seq_len = len(self.token_ids)


@dataclass
class SpanMasks:
    token_ids: list[int] = field(default_factory=list)
    typo_mask: list[int] = field(default_factory=list)
    candidate_masks: list[list[int]] = field(default_factory=list)


def _truncate_context(
    left: list[int],
    right: list[int],
    budget: int,
) -> tuple[list[int], list[int], bool]:
    """Keep approximately equal left/right contextual bytes around the typo."""
    if budget <= 0:
        return [], [], bool(left or right)
    if len(left) + len(right) <= budget:
        return left, right, False

    left_budget = budget // 2
    right_budget = budget - left_budget
    if len(left) < left_budget:
        right_budget += left_budget - len(left)
        left_budget = len(left)
    if len(right) < right_budget:
        left_budget = min(len(left), left_budget + (right_budget - len(right)))
        right_budget = len(right)

    new_left = left[-left_budget:] if left_budget < len(left) else left
    new_right = right[:right_budget] if right_budget < len(right) else right
    truncated = len(new_left) < len(left) or len(new_right) < len(right)
    return new_left, new_right, truncated


def serialize_example(
    context_before: str,
    typo: str,
    context_after: str,
    candidates: list[str | None],
    *,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    gold_index: int | None = None,
) -> SerializedExample:
    """Build one sequence that scores all Hunspell candidates.

    Never truncates the typo or any candidate string. Context is truncated
    symmetrically around the typo when the sequence would exceed max_seq_len.
    """
    if len(candidates) > N_CANDIDATES:
        candidates = candidates[:N_CANDIDATES]
    padded: list[str | None] = list(candidates) + [None] * (N_CANDIDATES - len(candidates))

    typo_bytes = text_to_byte_ids(nfc(typo))
    cand_bytes: list[list[int]] = []
    candidate_valid: list[bool] = []
    for cand in padded:
        if cand is None or cand == "":
            cand_bytes.append([])
            candidate_valid.append(False)
            continue
        encoded = text_to_byte_ids(nfc(cand))
        cand_bytes.append(encoded)
        candidate_valid.append(True)

    n_special = 6 + 2 * N_CANDIDATES
    reserved = n_special + len(typo_bytes) + sum(len(c) for c in cand_bytes)
    if reserved > max_seq_len:
        raise PathologicalExampleError(
            f"typo + candidates require {reserved} tokens > max_seq_len={max_seq_len}"
        )

    left = text_to_byte_ids(nfc(context_before))
    right = text_to_byte_ids(nfc(context_after))
    left, right, truncated = _truncate_context(left, right, max_seq_len - reserved)

    token_ids: list[int] = [CLS_ID, LANG_EN_ID, CTX_START_ID]
    token_ids.extend(left)
    token_ids.append(TYPO_START_ID)
    typo_start = len(token_ids)
    token_ids.extend(typo_bytes)
    typo_positions = list(range(typo_start, typo_start + len(typo_bytes)))
    token_ids.append(TYPO_END_ID)
    token_ids.extend(right)
    token_ids.append(CTX_END_ID)

    candidate_positions: list[list[int]] = []
    for idx in range(N_CANDIDATES):
        token_ids.append(CAND_IDS[idx])
        start = len(token_ids)
        token_ids.extend(cand_bytes[idx])
        candidate_positions.append(list(range(start, start + len(cand_bytes[idx]))))
        token_ids.append(CAND_END_ID)

    if len(token_ids) > max_seq_len:
        raise PathologicalExampleError(
            f"serialized length {len(token_ids)} exceeds max_seq_len={max_seq_len}"
        )

    return SerializedExample(
        token_ids=token_ids,
        typo_positions=typo_positions,
        candidate_positions=candidate_positions,
        candidate_valid=candidate_valid,
        gold_index=gold_index,
        truncated_context=truncated,
    )


def pad_batch(
    examples: list[SerializedExample],
    *,
    max_seq_len: int | None = None,
) -> dict:
    """Pad serialized examples into batched numpy arrays.

    Built with numpy rather than nested Python lists: the candidate mask alone
    is batch x 16 x seq_len entries (~900k for a 128-example batch), and
    materialising that as Python ints made collation, not the GPU, the
    bottleneck.
    """
    if not examples:
        raise ValueError("empty batch")
    import numpy as np

    length = max(ex.seq_len for ex in examples)
    if max_seq_len is not None:
        length = min(max(length, 1), max(max_seq_len, 1)) if length > max_seq_len else length
    batch = len(examples)

    token_ids = np.full((batch, length), PAD_ID, dtype=np.int64)
    attention = np.zeros((batch, length), dtype=np.int64)
    typo_mask = np.zeros((batch, length), dtype=np.int8)
    cand_masks = np.zeros((batch, N_CANDIDATES, length), dtype=np.int8)
    cand_valid = np.zeros((batch, N_CANDIDATES), dtype=np.int8)
    gold = np.empty(batch, dtype=np.int64)

    for i, ex in enumerate(examples):
        n = ex.seq_len
        token_ids[i, :n] = ex.token_ids
        attention[i, :n] = 1
        if ex.typo_positions:
            typo_mask[i, ex.typo_positions] = 1
        for ci, positions in enumerate(ex.candidate_positions):
            if positions:
                cand_masks[i, ci, positions] = 1
        cand_valid[i, : len(ex.candidate_valid)] = np.asarray(ex.candidate_valid, dtype=np.int8)
        gold[i] = -1 if ex.gold_index is None else int(ex.gold_index)

    return {
        "token_ids": token_ids,
        "attention_mask": attention,
        "typo_mask": typo_mask,
        "candidate_masks": cand_masks,
        "candidate_valid": cand_valid,
        "gold_index": gold,
    }


def locate_spans(example: SerializedExample) -> dict:
    """Return typo/candidate byte spans for tests."""
    return {
        "typo_positions": list(example.typo_positions),
        "candidate_positions": [list(p) for p in example.candidate_positions],
        "candidate_valid": list(example.candidate_valid),
        "token_ids": list(example.token_ids),
    }
