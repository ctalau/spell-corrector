# Experiment report — 28M Hunspell reranker vs Aspell on BEA-60K

This file is filled in after the CUDA training run. The code, configs, and
data pipeline are in the repository; GPU wall-clock numbers are not invented
here.

## 1. Goal

Beat Aspell top-1 spelling correction accuracy on BEA-60K with a ~28M
byte-level contextual reranker over Hunspell's top-10 suggestions.

## 2. Data sources

- Train/valid: synthetic typos from WikiText-103 raw (CC BY-SA). See `data/README.md`.
- Benchmark: NeuSpell BEA-60K (downloaded, not committed).

## 3. Dataset construction

See `data/processed/data_stats.json` and `manifest.json` after
`scripts/build_training_data.py`.

| Split | Target | Actual |
|-------|--------|--------|
| Train | 240,000 (min 150,000) | _pending data build_ |
| Valid | 20,000 (min 10,000) | _pending data build_ |

Authentic typo corpora omitted (redistribution).

## 4. Hunspell

See `artifacts/hunspell_metadata.json` / `artifacts/model/hunspell_metadata.json`.

## 5–8. Model

- Architecture: 8-layer byte Transformer, d_model=512, 8 heads, SwiGLU 1536,
  RMSNorm, RoPE, bidirectional SDPA, ranking MLP 1536→320→1.
- Parameter count: run `python -m pytest tests/test_model.py::test_parameter_count_approximately_28m -s`

## 9–11. Training hyperparameters / GPU / cost

From `configs/train_full.yaml`. Seed 1337. BF16, AdamW 3e-4, wd 0.10,
cosine + 5% warmup, effective batch 256, 3 epochs.

| Item | Value |
|------|-------|
| GPU | _pending Runpod run_ |
| Runtime | _pending_ |
| Approx. cost | _pending_ |
| Peak VRAM | _pending_ |

## 12–14. Training / validation curves

See `artifacts/train_metrics.jsonl` and `reports/training_loss.png` after train.

## 15–22. BEA-60K

Populate from `reports/bea60k/results.json`.

| System | Overall correction accuracy |
|--------|-----------------------------|
| Aspell top-1 | _pending_ |
| Hunspell top-1 | _pending_ |
| Hunspell oracle@10 | _pending_ |
| 28M reranker + Hunspell | _pending_ |

**DID WE BEAT ASPELL?** _pending_

## 23–24. Conclusions / next experiment

_pending after the A5000 (or equivalent) run._
