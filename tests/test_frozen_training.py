"""Frozen selector training helpers: JSON NaN, BEA cadence, linear stability."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_frozen_selector import bea_gate_due, default_learning_rate, dumps_json, json_safe
from spelling_reranker.candidates import FROZEN_CANDIDATE_SLOTS
from spelling_reranker.frozen_encoder import (
    ScalarScaler,
    SelectorHead,
    assemble_selector_features,
    encoder_feature_scale,
    mask_invalid_logits,
)
from spelling_reranker.seed import seed_everything


def test_json_safe_replaces_nonfinite_with_none() -> None:
    payload = {
        "ok": 1.5,
        "nan": float("nan"),
        "inf": float("inf"),
        "nested": {"loss": np.float32(np.nan), "n": np.int64(3)},
        "seq": [1.0, float("-inf")],
    }
    safe = json_safe(payload)
    assert safe["ok"] == 1.5
    assert safe["nan"] is None
    assert safe["inf"] is None
    assert safe["nested"]["loss"] is None
    assert safe["nested"]["n"] == 3
    assert safe["seq"] == [1.0, None]
    text = dumps_json(payload)
    assert "NaN" not in text
    assert "Infinity" not in text
    assert "null" in text


def test_default_learning_rate_linear_is_3e_4() -> None:
    cfg = {"learning_rate": 1e-3}
    assert default_learning_rate("linear", cfg, None) == 3e-4
    assert default_learning_rate("linear", {"learning_rate_linear": 5e-4}, None) == 5e-4
    assert default_learning_rate("linear", cfg, 1e-3) == 1e-3
    assert default_learning_rate("mlp", cfg, None) == 1e-3
    assert default_learning_rate("scalar", cfg, None) == 1e-3


def test_bea_gate_due_on_epoch_steps_and_wall() -> None:
    kwargs = dict(use_gate=True, last_wall=0.0, wall_seconds=1800.0, last_step=-1, every_steps=200)
    assert bea_gate_due(**kwargs, now=10.0, step=50, force=True) is True
    assert bea_gate_due(**kwargs, now=10.0, step=200) is True
    assert bea_gate_due(**kwargs, now=10.0, step=50) is False
    assert bea_gate_due(**kwargs, now=1800.0, step=50) is True
    assert bea_gate_due(**{**kwargs, "last_step": 7}, now=1800.0, step=7) is False
    assert bea_gate_due(**{**kwargs, "use_gate": False}, now=1800.0, step=50, force=True) is False


def test_linear_head_uses_input_layernorm() -> None:
    head = SelectorHead("linear", dropout=0.0)
    assert isinstance(head.input_norm, torch.nn.LayerNorm)
    assert head.input_norm.normalized_shape == (head.in_dim,)
    mlp = SelectorHead("mlp", dropout=0.0)
    assert isinstance(mlp.input_norm, torch.nn.Identity)


def test_encoder_feature_scale_and_linear_stay_finite() -> None:
    seed_everything(1337)
    n, n_cand, hidden = 16, FROZEN_CANDIDATE_SLOTS, 768
    context = torch.randn(n, hidden) * 25
    typo = torch.randn(n, hidden) * 25
    candidates = torch.randn(n, n_cand, hidden) * 25
    gold = torch.arange(n) % 8
    for i in range(n):
        candidates[i, int(gold[i])] = 40.0
    scalars = torch.zeros(n, n_cand, 4)
    valid = torch.ones(n, n_cand, dtype=torch.int64)
    valid[:, 8:] = 0
    scale = encoder_feature_scale(
        context.numpy(), typo.numpy(), candidates.numpy()
    )
    assert scale > 1.0
    scaler = ScalarScaler(encoder_scale=scale)
    head = SelectorHead("linear", dropout=0.0)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4)
    last_loss = None
    for _ in range(8):
        opt.zero_grad(set_to_none=True)
        features = assemble_selector_features(context, typo, candidates, scalars, scaler=scaler)
        assert torch.isfinite(features).all()
        logits = mask_invalid_logits(head(features), valid)
        assert torch.isfinite(logits).all()
        loss = F.cross_entropy(logits, gold)
        assert torch.isfinite(loss), float(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        last_loss = float(loss.item())
    assert last_loss is not None and math.isfinite(last_loss)


def test_frozen_yaml_has_linear_lr_and_bea_step_gate() -> None:
    text = (ROOT / "configs" / "train_frozen_modernbert.yaml").read_text()
    assert "learning_rate_linear: 3.0e-4" in text
    assert "bea_every_steps: 200" in text
    assert "max_grad_norm: 1.0" in text
