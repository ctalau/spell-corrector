"""1. BYTE ROUNDTRIP TEST — UTF-8 text -> byte IDs -> text is lossless."""

from __future__ import annotations

from spelling_reranker.byte_encoding import (
    BYTE_VOCAB,
    N_CANDIDATE_SLOTS,
    VOCAB_SIZE,
    assert_vocab_complete,
    byte_ids_to_text,
    nfc,
    special_tokens_map,
    text_to_byte_ids,
)


def test_utf8_roundtrip_ascii() -> None:
    text = "The quick brown fox jumps over the lazy dog."
    assert byte_ids_to_text(text_to_byte_ids(text)) == text


def test_utf8_roundtrip_unicode() -> None:
    samples = [
        "café",
        "naïve",
        "東京",
        "emoji 💡",
        "Ångström",
        "e\u0301 vs é",
        "crème brûlée — “quotes”",
    ]
    for text in samples:
        ids = text_to_byte_ids(text)
        assert all(0 <= i <= 255 for i in ids)
        assert byte_ids_to_text(ids) == text


def test_nfc_does_not_lowercase() -> None:
    assert nfc("Café") == "Café"
    assert nfc("USA") == "USA"


def test_vocab_size_and_specials() -> None:
    tokens = special_tokens_map()
    # 256 raw bytes + 7 structural specials + one token per candidate slot
    # + CAND_END + MASK.
    expected_vocab = BYTE_VOCAB + 9 + N_CANDIDATE_SLOTS
    assert VOCAB_SIZE == expected_vocab
    assert tokens["VOCAB_SIZE"] == expected_vocab
    assert tokens["CAND_0"] == 263
    assert tokens[f"CAND_{N_CANDIDATE_SLOTS - 1}"] == 263 + N_CANDIDATE_SLOTS - 1
    assert tokens["CAND_END"] == 263 + N_CANDIDATE_SLOTS
    assert tokens["MASK"] == 264 + N_CANDIDATE_SLOTS


def test_special_token_ids_are_a_contiguous_block() -> None:
    assert_vocab_complete()
