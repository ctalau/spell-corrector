"""YAML config loading for model and training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from spelling_reranker.model import ModelConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config {path} must be a mapping")
    return data


def resolve_maybe_path(value: Any, relative_to: Path) -> Any:
    if isinstance(value, str) and value.endswith((".yaml", ".yml")):
        nested = Path(value)
        if not nested.is_absolute():
            # Try CWD first, then next to the parent config.
            if nested.exists():
                return load_yaml(nested)
            sibling = relative_to.parent / nested
            if sibling.exists():
                return load_yaml(sibling)
    return value


def load_train_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    cfg = load_yaml(cfg_path)
    model_field = cfg.get("model")
    if isinstance(model_field, str):
        cfg["model"] = resolve_maybe_path(model_field, cfg_path)
    elif model_field is None and "model_config" in cfg:
        cfg["model"] = cfg["model_config"]
    return cfg


def model_config_from_mapping(data: dict[str, Any] | None) -> ModelConfig:
    if not data:
        return ModelConfig()
    rename = {
        "layers": "n_layers",
        "attention_heads": "n_heads",
        "ffn": "ffn_hidden",
        "attention_dropout": "attn_dropout",
        "max_sequence_bytes": "max_seq_len",
    }
    mapped = {rename.get(k, k): v for k, v in data.items()}
    return ModelConfig.from_dict(mapped)
