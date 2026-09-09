"""Parquet dataset + byte-level DataLoader collation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from spelling_reranker.serialization import (
    DEFAULT_MAX_SEQ_LEN,
    N_CANDIDATES,
    PathologicalExampleError,
    SerializedExample,
    pad_batch,
    serialize_example,
)

CAND_COLUMNS = [f"cand_{i}" for i in range(N_CANDIDATES)]
REQUIRED_COLUMNS = [
    "example_id",
    "source",
    "context_before",
    "typo",
    "context_after",
    "gold",
    *CAND_COLUMNS,
    "gold_index",
]


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value)
    return None if text == "" or text == "None" or text == "<NA>" else text


def row_to_candidates(row: pd.Series) -> list[str | None]:
    return [_as_optional_str(row.get(col)) for col in CAND_COLUMNS]


def serialize_row(row: pd.Series, max_seq_len: int = DEFAULT_MAX_SEQ_LEN) -> SerializedExample:
    return serialize_example(
        str(row["context_before"]),
        str(row["typo"]),
        str(row["context_after"]),
        row_to_candidates(row),
        max_seq_len=max_seq_len,
        gold_index=int(row["gold_index"]),
    )


class SpellingParquetDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        *,
        max_examples: int | None = None,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    ) -> None:
        self.path = Path(path)
        table = pq.read_table(self.path)
        frame = table.to_pandas()
        if max_examples is not None:
            frame = frame.iloc[: int(max_examples)].copy()
        self.frame = frame.reset_index(drop=True)
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> SerializedExample:
        row = self.frame.iloc[index]
        try:
            return serialize_row(row, max_seq_len=self.max_seq_len)
        except PathologicalExampleError as exc:
            raise RuntimeError(f"pathological example {row.get('example_id')}: {exc}") from exc


def collate_examples(
    examples: list[SerializedExample],
    *,
    max_seq_len: int | None = None,
) -> dict[str, torch.Tensor]:
    padded = pad_batch(examples, max_seq_len=max_seq_len)
    return {
        "token_ids": torch.tensor(padded["token_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(padded["attention_mask"], dtype=torch.long),
        "typo_mask": torch.tensor(padded["typo_mask"], dtype=torch.long),
        "candidate_masks": torch.tensor(padded["candidate_masks"], dtype=torch.long),
        "candidate_valid": torch.tensor(padded["candidate_valid"], dtype=torch.long),
        "gold_index": torch.tensor(padded["gold_index"], dtype=torch.long),
    }


def make_collate(max_seq_len: int | None = None):
    def _collate(examples: list[SerializedExample]) -> dict[str, torch.Tensor]:
        return collate_examples(examples, max_seq_len=max_seq_len)

    return _collate
