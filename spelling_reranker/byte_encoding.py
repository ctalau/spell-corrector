"""Byte-level vocabulary and lossless UTF-8 encoding."""

from __future__ import annotations

import unicodedata
from typing import Iterable, Mapping

BYTE_VOCAB = 256

PAD_ID = 256
CLS_ID = 257
LANG_EN_ID = 258
CTX_START_ID = 259
CTX_END_ID = 260
TYPO_START_ID = 261
TYPO_END_ID = 262
CAND_0_ID = 263
CAND_IDS = tuple(range(CAND_0_ID, CAND_0_ID + 10))
CAND_END_ID = 273
VOCAB_SIZE = 274

SPECIAL_TOKEN_NAMES = (
    "PAD",
    "CLS",
    "LANG_EN",
    "CTX_START",
    "CTX_END",
    "TYPO_START",
    "TYPO_END",
    "CAND_0",
    "CAND_1",
    "CAND_2",
    "CAND_3",
    "CAND_4",
    "CAND_5",
    "CAND_6",
    "CAND_7",
    "CAND_8",
    "CAND_9",
    "CAND_END",
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
    return {
        "PAD": PAD_ID,
        "CLS": CLS_ID,
        "LANG_EN": LANG_EN_ID,
        "CTX_START": CTX_START_ID,
        "CTX_END": CTX_END_ID,
        "TYPO_START": TYPO_START_ID,
        "TYPO_END": TYPO_END_ID,
        "CAND_0": CAND_IDS[0],
        "CAND_1": CAND_IDS[1],
        "CAND_2": CAND_IDS[2],
        "CAND_3": CAND_IDS[3],
        "CAND_4": CAND_IDS[4],
        "CAND_5": CAND_IDS[5],
        "CAND_6": CAND_IDS[6],
        "CAND_7": CAND_IDS[7],
        "CAND_8": CAND_IDS[8],
        "CAND_9": CAND_IDS[9],
        "CAND_END": CAND_END_ID,
        "BYTE_VOCAB": BYTE_VOCAB,
        "VOCAB_SIZE": VOCAB_SIZE,
    }


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
        "CAND_9": 272,
        "CAND_END": 273,
        "VOCAB_SIZE": 274,
    }
    for key, value in expected.items():
        if names[key] != value:
            raise RuntimeError(f"special token {key} expected {value}, got {names[key]}")


def dump_special_tokens() -> Mapping[str, int]:
    assert_vocab_complete()
    return special_tokens_map()
