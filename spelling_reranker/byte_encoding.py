"""Byte-level vocabulary and lossless UTF-8 encoding."""

from __future__ import annotations

import unicodedata
from typing import Iterable, Mapping

BYTE_VOCAB = 256

#: Number of candidate slots the model scores in a single forward pass.
#: Widened from 10 to 16 so the pool can hold Hunspell's suggestions *and*
#: the Aspell suggestions Hunspell misses (see reports/EXPERIMENT.md).
N_CANDIDATE_SLOTS = 16

PAD_ID = 256
CLS_ID = 257
LANG_EN_ID = 258
CTX_START_ID = 259
CTX_END_ID = 260
TYPO_START_ID = 261
TYPO_END_ID = 262
CAND_0_ID = 263
CAND_IDS = tuple(range(CAND_0_ID, CAND_0_ID + N_CANDIDATE_SLOTS))
CAND_END_ID = CAND_IDS[-1] + 1
#: Used only by the auxiliary masked-byte objective during training. Inference
#: never emits it.
MASK_ID = CAND_END_ID + 1
VOCAB_SIZE = MASK_ID + 1

SPECIAL_TOKEN_NAMES = (
    "PAD",
    "CLS",
    "LANG_EN",
    "CTX_START",
    "CTX_END",
    "TYPO_START",
    "TYPO_END",
    *[f"CAND_{i}" for i in range(N_CANDIDATE_SLOTS)],
    "CAND_END",
    "MASK",
)


def nfc(text: str) -> str:
    """Normalize Unicode to NFC. Does not lowercase."""
    return unicodedata.normalize("NFC", text)


def text_to_byte_ids(text: str) -> list[int]:
    """Encode text as raw UTF-8 byte IDs in 0..255."""
    return list(text.encode("utf-8"))


def byte_ids_to_text(ids: Iterable[int]) -> str:
    """Decode a sequence of UTF-8 byte IDs back to text."""
    return bytes(int(i) for i in ids).decode("utf-8")


def special_tokens_map() -> dict[str, int]:
    mapping: dict[str, int] = {
        "PAD": PAD_ID,
        "CLS": CLS_ID,
        "LANG_EN": LANG_EN_ID,
        "CTX_START": CTX_START_ID,
        "CTX_END": CTX_END_ID,
        "TYPO_START": TYPO_START_ID,
        "TYPO_END": TYPO_END_ID,
    }
    for i, token_id in enumerate(CAND_IDS):
        mapping[f"CAND_{i}"] = token_id
    mapping["CAND_END"] = CAND_END_ID
    mapping["MASK"] = MASK_ID
    mapping["BYTE_VOCAB"] = BYTE_VOCAB
    mapping["VOCAB_SIZE"] = VOCAB_SIZE
    return mapping


def is_special_id(token_id: int) -> bool:
    return int(token_id) >= BYTE_VOCAB


def assert_vocab_complete() -> None:
    names = special_tokens_map()
    expected = {
        "PAD": 256,
        "CLS": 257,
        "LANG_EN": 258,
        "CTX_START": 259,
        "CTX_END": 260,
        "TYPO_START": 261,
        "TYPO_END": 262,
        "CAND_0": 263,
        f"CAND_{N_CANDIDATE_SLOTS - 1}": 263 + N_CANDIDATE_SLOTS - 1,
        "CAND_END": 263 + N_CANDIDATE_SLOTS,
        "MASK": 264 + N_CANDIDATE_SLOTS,
        "VOCAB_SIZE": 265 + N_CANDIDATE_SLOTS,
    }
    for key, value in expected.items():
        if names[key] != value:
            raise RuntimeError(f"special token {key} expected {value}, got {names[key]}")
    # IDs must be contiguous and unique.
    ids = [v for k, v in names.items() if k not in ("BYTE_VOCAB", "VOCAB_SIZE")]
    if sorted(ids) != list(range(BYTE_VOCAB, VOCAB_SIZE)):
        raise RuntimeError("special token IDs are not a contiguous block")


def dump_special_tokens() -> Mapping[str, int]:
    assert_vocab_complete()
    return special_tokens_map()
