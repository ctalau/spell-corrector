"""Parquet dataset + byte-level DataLoader collation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler

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
        self._lengths: np.ndarray | None = None

    @property
    def lengths(self) -> np.ndarray:
        """Approximate serialized length per row, for length-bucketed batching."""
        if self._lengths is None:
            frame = self.frame

            def _blen(series) -> np.ndarray:
                return series.fillna("").astype(str).str.len().to_numpy()

            total = _blen(frame["context_before"]) + _blen(frame["typo"]) + _blen(frame["context_after"])
            for col in CAND_COLUMNS:
                total = total + _blen(frame[col]) + 2
            total = total + 6
            self._lengths = np.minimum(total, self.max_seq_len).astype(np.int32)
        return self._lengths

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
        "token_ids": torch.from_numpy(padded["token_ids"]),
        "attention_mask": torch.from_numpy(padded["attention_mask"]),
        "typo_mask": torch.from_numpy(padded["typo_mask"]),
        "candidate_masks": torch.from_numpy(padded["candidate_masks"]),
        "candidate_valid": torch.from_numpy(padded["candidate_valid"]),
        "gold_index": torch.from_numpy(padded["gold_index"]),
    }


def make_collate(max_seq_len: int | None = None):
    def _collate(examples: list[SerializedExample]) -> dict[str, torch.Tensor]:
        return collate_examples(examples, max_seq_len=max_seq_len)

    return _collate


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Batch indices of similar serialized length together.

    Sequences here vary from ~120 to 448 bytes, and a batch is padded to its
    longest member, so random batching wastes a large fraction of every forward
    pass on padding. Shuffling, then sorting inside a large window, then
    shuffling the resulting batches keeps the sampling close to random while
    cutting padded tokens substantially.
    """

    def __init__(
        self,
        lengths: "np.ndarray",
        batch_size: int,
        *,
        shuffle: bool = True,
        window_batches: int = 64,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.lengths = lengths
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.window = max(1, int(window_batches)) * self.batch_size
        self.seed = int(seed)
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        n = len(self.lengths)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.lengths)
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(n) if self.shuffle else np.arange(n)
        batches: list[list[int]] = []
        for start in range(0, n, self.window):
            window = order[start : start + self.window]
            window = window[np.argsort(self.lengths[window], kind="stable")]
            for b_start in range(0, len(window), self.batch_size):
                batch = window[b_start : b_start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append([int(i) for i in batch])
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)
