"""2. INPUT SERIALIZATION  3. CANDIDATE LABEL  4. PADDING."""

from __future__ import annotations

import torch

from spelling_reranker.byte_encoding import (
    CAND_END_ID,
    CAND_IDS,
    CLS_ID,
    TYPO_END_ID,
    TYPO_START_ID,
    byte_ids_to_text,
    nfc,
)
from spelling_reranker.hunspell import HunspellEngine
from spelling_reranker.model import ByteSpellingReranker, ModelConfig
from spelling_reranker.serialization import (
    serialize_example,
)
from spelling_reranker.dataset import collate_examples


def _example(n_cands: int = 10):
    cands = [f"cand{i}" for i in range(n_cands)]
    return serialize_example(
        "The left ",
        "quik",
        " brown fox.",
        cands,
        gold_index=0,
    )


def test_typo_and_candidate_spans_located() -> None:
    ex = _example()
    typo = byte_ids_to_text(ex.token_ids[i] for i in ex.typo_positions)
    assert typo == "quik"
    assert ex.token_ids[ex.typo_positions[0] - 1] == TYPO_START_ID
    assert ex.token_ids[ex.typo_positions[-1] + 1] == TYPO_END_ID
    assert ex.token_ids[0] == CLS_ID
    for i, positions in enumerate(ex.candidate_positions):
        recovered = byte_ids_to_text(ex.token_ids[p] for p in positions)
        assert recovered == f"cand{i}"
        assert ex.token_ids[positions[0] - 1] == CAND_IDS[i]
        assert ex.token_ids[positions[-1] + 1] == CAND_END_ID


def test_context_truncation_keeps_typo_and_candidates() -> None:
    left = "L" * 400
    right = "R" * 400
    cands = ["alpha", "beta"]
    ex = serialize_example(left, "typo", right, cands, max_seq_len=128)
    assert byte_ids_to_text(ex.token_ids[i] for i in ex.typo_positions) == "typo"
    assert byte_ids_to_text(ex.token_ids[i] for i in ex.candidate_positions[0]) == "alpha"
    assert byte_ids_to_text(ex.token_ids[i] for i in ex.candidate_positions[1]) == "beta"
    assert ex.truncated_context
    assert ex.seq_len <= 128


def test_gold_index_points_to_exact_candidate() -> None:
    engine = HunspellEngine()
    typo = "teh"
    gold = "the"
    cands = engine.candidates(typo)
    idx = engine.gold_index(cands, gold)
    assert idx is not None
    assert nfc(cands[idx]) == nfc(gold)
    ex = serialize_example(" ", typo, " ", cands, gold_index=idx)
    recovered = byte_ids_to_text(ex.token_ids[i] for i in ex.candidate_positions[idx])
    assert nfc(recovered) == nfc(gold)


def test_padding_masks_missing_candidates() -> None:
    cands = ["the", "eh", "tech"]
    ex = serialize_example("See ", "teh", ".", cands, gold_index=0)
    assert ex.candidate_valid == [True, True, True, False, False, False, False, False, False, False]
    for i in range(3, 10):
        assert ex.candidate_positions[i] == []

    model = ByteSpellingReranker(ModelConfig(n_layers=1, max_seq_len=64))
    model.eval()
    batch = collate_examples([ex])
    with torch.no_grad():
        logits = model(
            batch["token_ids"],
            batch["attention_mask"],
            batch["typo_mask"],
            batch["candidate_masks"],
            batch["candidate_valid"],
        )
    assert logits.shape == (1, 10)
    assert torch.isneginf(logits[0, 3:]).all() or (logits[0, 3:] <= torch.finfo(logits.dtype).min / 2).all()
    probs = torch.softmax(logits[0, :3], dim=0)
    assert torch.isfinite(probs).all()
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
