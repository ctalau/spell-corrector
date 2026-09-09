"""1. BYTE ROUNDTRIP TEST — UTF-8 text -> byte IDs -> text is lossless."""

from __future__ import annotations

from spelling_reranker.byte_encoding import (
    VOCAB_SIZE,
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
    assert tokens["VOCAB_SIZE"] == 274
    assert VOCAB_SIZE == 274
    assert tokens["CAND_0"] == 263
    assert tokens["CAND_9"] == 272
    assert tokens["CAND_END"] == 273
