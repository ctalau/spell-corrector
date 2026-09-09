# Spell Corrector

Tiny (~28M) **byte-level contextual spelling reranker**. Hunspell proposes up to 10
candidates; a bidirectional Transformer picks exactly one.

This repository is the first experiment from [PLAN.md](PLAN.md):

> Beat Aspell's top-1 spelling correction accuracy on BEA-60K.

The model is **not** generative. Training labels are Hunspell ranks `0..9` on
synthetic WikiText-103 typos. BEA-60K is a locked final benchmark and is never
used for training, validation, or hyperparameter selection.

## Status

- Code, configs, unit tests, and data pipeline: implemented.
- Full GPU training / BEA numbers: run on a CUDA box (see below). Results go in
  `reports/EXPERIMENT.md`.

## Install

System packages (Debian/Ubuntu):

```bash
sudo apt-get install -y hunspell libhunspell-dev hunspell-en-us aspell aspell-en
```

Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate

# GPU (Runpod / CUDA 12.x). Use the CPU index on machines without NVIDIA.
pip install torch --index-url https://download.pytorch.org/whl/cu124
# pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[dev]"
pip install -r requirements.lock
```

`requirements.lock` pins versions. Install a platform-appropriate `torch` wheel
**first** so the lockfile does not force a CPU-only build onto a GPU pod.

Confirm Hunspell:

```bash
echo teh | hunspell -d en_US -a
```

## Tests (CPU)

```bash
python -m pytest tests/ -q
```

Section 11 of PLAN.md is covered by:

| Test | File |
|------|------|
| Byte roundtrip | `tests/test_byte_encoding.py` |
| Serialization spans | `tests/test_serialization.py` |
| Gold label / padding | `tests/test_serialization.py`, `tests/test_dataset.py` |
| Shape / param count / loss | `tests/test_model.py` |
| Tiny overfit | `tests/test_tiny_overfit.py` |
| Determinism | `tests/test_dataset.py` |
| No locked-benchmark leak into train construction | `tests/test_dataset.py` |

The parameter-count test prints the trainable size and asserts it is ≤29M and
approximately 28M.

## Build training data

Synthetic only (WikiText-103 raw, CC BY-SA). No authentic typo corpus is
redistributed. See [data/README.md](data/README.md).

```bash
python scripts/download_sources.py
python scripts/build_training_data.py \
  --target-train 240000 \
  --target-valid 20000 \
  --seed 1337
```

Minimum acceptable full run: 150k train / 10k valid. Outputs:

- `data/processed/train.parquet`
- `data/processed/validation.parquet`
- `data/processed/data_stats.json`
- `data/processed/manifest.json`
- `artifacts/hunspell_metadata.json`

Committed processed data (Git LFS): **235,626 train / 19,642 valid** examples
(above the 150k/10k minimum; WikiText heading boilerplate filtered). Rebuild
on the training machine if you want a fresh generation.

## Train

```bash
# cheap sanity (~2k/500 examples, ≤200 optimizer steps)
python scripts/train.py --config configs/train_sanity.yaml

# full experiment
python scripts/train.py --config configs/train_full.yaml
```

On OOM, edit `configs/train_full.yaml` (`microbatch` 64 / `grad_accumulation` 4,
or 32 / 8) so the effective batch stays 256.

Sanity wrapper (tests + sanity train + eval smoke):

```bash
bash scripts/run_sanity.sh
```

Checkpoints land in `artifacts/model/` (`model.safetensors`, `config.json`,
`special_tokens.json`, `hunspell_metadata.json`, `training_manifest.json`).

## Benchmark BEA-60K

Do **not** commit BEA files if redistribution is restricted. Download locally:

```bash
python scripts/download_bea60k.py
python scripts/benchmark_aspell.py
python scripts/benchmark_bea60k.py \
  --model artifacts/model \
  --output reports/bea60k
```

Primary comparison: model overall success rate vs Aspell top-1 on the same
extracted word errors.

## Repository layout

Matches PLAN.md §21: `spelling_reranker/`, `configs/`, `scripts/`, `tests/`,
`data/`, `artifacts/`, `reports/`.

## Design constraints

- Byte vocabulary 0..255 plus 18 specials (274 total). No BPE/SPM.
- One forward pass scores all 10 candidates.
- Hunspell suggestion order is preserved (`CAND_0`…`CAND_9`).
- Unicode NFC; no global lowercasing.
- Seed **1337** for data and training.
- Git LFS tracks `*.parquet`, `*.safetensors`, `*.pt`, `*.bin`.
