"""Load a trained reranker and pick one Hunspell candidate."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from spelling_reranker.dataset import collate_examples
from spelling_reranker.model import ByteSpellingReranker, ModelConfig
from spelling_reranker.serialization import serialize_example


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
