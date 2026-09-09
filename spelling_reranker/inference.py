"""Load a trained reranker and pick one Hunspell candidate."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from spelling_reranker.dataset import collate_examples
from spelling_reranker.model import ByteSpellingReranker, ModelConfig
from spelling_reranker.serialization import PathologicalExampleError, serialize_example


def load_model_dir(model_dir: str | Path, device: torch.device | None = None) -> ByteSpellingReranker:
    model_dir = Path(model_dir)
    device = device or torch.device("cpu")
    config_path = model_dir / "config.json"
    weights = model_dir / "model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    cfg = ModelConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    model = ByteSpellingReranker(cfg)
    if weights.is_file():
        state = load_file(str(weights), device=str(device))
        model.load_state_dict(state)
    else:
        pt_path = model_dir / "model.pt"
        if not pt_path.is_file():
            raise FileNotFoundError(f"no weights in {model_dir}")
        state = torch.load(pt_path, map_location=device)
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_index(
    model: ByteSpellingReranker,
    context_before: str,
    typo: str,
    context_after: str,
    candidates: list[str | None],
    device: torch.device | None = None,
) -> int:
    device = device or next(model.parameters()).device
    serialized = serialize_example(
        context_before,
        typo,
        context_after,
        candidates,
        max_seq_len=model.cfg.max_seq_len,
    )
    batch = collate_examples([serialized])
    batch = {k: v.to(device) for k, v in batch.items() if k != "gold_index"}
    logits = model(
        batch["token_ids"],
        batch["attention_mask"],
        batch["typo_mask"],
        batch["candidate_masks"],
        batch["candidate_valid"],
    )
    return int(logits[0].argmax(dim=-1).item())


def predict_word(
    model: ByteSpellingReranker,
    context_before: str,
    typo: str,
    context_after: str,
    candidates: list[str | None],
    device: torch.device | None = None,
) -> str | None:
    idx = predict_index(model, context_before, typo, context_after, candidates, device=device)
    if idx < 0 or idx >= len(candidates):
        return None
    return candidates[idx]


@torch.no_grad()
def predict_indices(
    model: ByteSpellingReranker,
    items: list[tuple[str, str, str, list[str | None]]],
    *,
    device: torch.device | None = None,
    batch_size: int = 128,
) -> list[int]:
    """Batched form of `predict_index`.

    `items` are (context_before, typo, context_after, candidates) tuples. The
    benchmark scores tens of thousands of errors, and one forward pass each
    leaves the GPU almost idle.
    """
    device = device or next(model.parameters()).device
    out: list[int] = []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        serialized = []
        fallback: list[int] = []
        for offset, (before, typo, after, cands) in enumerate(chunk):
            try:
                serialized.append(
                    serialize_example(before, typo, after, cands, max_seq_len=model.cfg.max_seq_len)
                )
            except PathologicalExampleError:
                # Should not happen: DEFAULT_MAX_SEQ_LEN is sized so a full pool
                # always fits. Degrade to the first candidate rather than abort a
                # sweep of tens of thousands of errors over one freak input.
                fallback.append(offset)
        if fallback and not serialized:
            out.extend(0 for _ in chunk)
            continue
        batch = collate_examples(serialized)
        batch = {k: v.to(device) for k, v in batch.items() if k != "gold_index"}
        logits = model(
            batch["token_ids"],
            batch["attention_mask"],
            batch["typo_mask"],
            batch["candidate_masks"],
            batch["candidate_valid"],
        )
        picked = [int(i) for i in logits.argmax(dim=-1).tolist()]
        if fallback:
            merged: list[int] = []
            it = iter(picked)
            for offset in range(len(chunk)):
                merged.append(0 if offset in fallback else next(it))
            picked = merged
        out.extend(picked)
    return out
