"""Candidate pool construction. Hunspell is the only suggestion source.

Hunspell's own suggestion list is used in full rather than truncated at 10.
Most typos get far fewer than 10 suggestions, so the extra slots usually cost
nothing at all, and when Hunspell does return a long list the correction is
occasionally past rank 9 -- cheap insurance rather than a large win.
`scripts/calibrate_typo_model.py` reports both figures against a public
misspelling list.

Slot order is Hunspell's own order, so slot 0 is always Hunspell's top-1 and the
CAND_i token continues to carry Hunspell's ranking as a prior.
"""

from __future__ import annotations

from typing import Sequence

from spelling_reranker.byte_encoding import N_CANDIDATE_SLOTS, nfc

#: Frozen-encoder track uses Hunspell's raw first ten suggestions. Slice the
#: raw list to this width *before* dedup or length filtering so a discarded
#: duplicate in ranks 0-9 cannot pull in raw rank 11.
FROZEN_CANDIDATE_SLOTS = 10


def build_pool(
    hunspell_suggestions: Sequence[str],
    *,
    limit: int = N_CANDIDATE_SLOTS,
    max_bytes: int | None = None,
) -> list[str]:
    """Deduplicated, NFC-normalized Hunspell pool, at most `limit` long."""
    pool: list[str] = []
    seen: set[str] = set()
    for word in hunspell_suggestions:
        if len(pool) >= limit:
            break
        normalized = nfc(word)
        if not normalized or normalized in seen:
            continue
        if max_bytes is not None and len(normalized.encode("utf-8")) > max_bytes:
            continue
        seen.add(normalized)
        pool.append(normalized)
    return pool


def first_ten_pool(
    hunspell_suggestions: Sequence[str],
    *,
    max_bytes: int | None = None,
) -> list[str]:
    """Top-10 policy: slice raw suggestions first, then filter.

    ``build_pool(suggestions, limit=10)`` can backfill with raw ranks beyond 10
    after deduplication or length filtering. This function must not.
    """
    sliced = list(hunspell_suggestions)[:FROZEN_CANDIDATE_SLOTS]
    return build_pool(sliced, limit=FROZEN_CANDIDATE_SLOTS, max_bytes=max_bytes)


def gold_index(candidates: Sequence[str | None], gold: str) -> int | None:
    """Index of `gold` in the pool under NFC equality, or None."""
    target = nfc(gold)
    for idx, cand in enumerate(candidates):
        if cand is not None and nfc(cand) == target:
            return idx
    return None


def pad_pool(candidates: Sequence[str], limit: int = N_CANDIDATE_SLOTS) -> list[str | None]:
    """Pad a pool out to `limit` slots with None."""
    pool: list[str | None] = list(candidates)[:limit]
    return pool + [None] * (limit - len(pool))
