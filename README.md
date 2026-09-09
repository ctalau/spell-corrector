# Spell Corrector

Byte-level **contextual spelling reranker**. Hunspell proposes candidates; a
bidirectional Transformer picks exactly one.

This repository is the experiment line from [PLAN.md](PLAN.md):

> Beat Aspell's top-1 spelling correction accuracy on BEA-60K.

The model is **not** generative. Training labels are Hunspell ranks on synthetic
typos over WikiText-103. BEA-60K is a locked final benchmark and is never used
for training, validation, or hyperparameter selection.

## Where the accuracy comes from

Experiment 1 reached **62.56%** overall (vs Aspell 60.55%). Decomposed:

```
overall = P(gold in Hunspell pool) x P(model picks gold | it is there)
62.56%  =        80.38%            x            77.84%
```

Hunspell is fixed as the only candidate source, so the first factor is a hard
ceiling around 81%. Everything in experiment 2 targets the second factor:

| Change | Why |
|---|---|
| Typo generator rewritten | Experiment 1 generated **only** edit-distance-1 typos. Authentic misspellings are ~73% ED1 / ~25% ED2 / ~2% ED3+, and Hunspell's top-1 collapses as edit distance grows — the ED>=2 quarter is exactly where a reranker earns its keep, and the model had never seen it. |
| Phonetic/orthographic corruptions | Real errors are how a writer *thinks* a word is spelled (doubling, silent letters, reduced vowels, suffix confusion), not uniform keyboard noise. |
| Noisy context augmentation | A corrector reads uncorrected text, so neighbouring words are often misspelled too. Training on clean context taught the model to over-trust it. |
| Gold-index balancing | Hunspell already ranks the answer first for ~81% of synthetic typos. Those examples only teach the model to agree with Hunspell. |
| 16 candidate slots | Hunspell's suggestion list is no longer truncated at 10. |
| ~17x more training data | Made affordable by the data-pipeline rewrite below. |
| 87M parameters | Modelling English context is the binding constraint once the candidate list is fixed. |
| Richer scoring head | Adds `cand*typo` and `abs(cand-typo)` interaction features. |

Calibration of the typo generator uses Wikipedia's public
[common misspellings list](https://en.wikipedia.org/wiki/Wikipedia:Lists_of_common_misspellings),
never the benchmark — see `scripts/calibrate_typo_model.py` and
`reports/typo_calibration.json`.

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

pip install -e ".[dev]"
```

`hunspell==0.5.5` does not build against setuptools >= 60 (it fails with
`AttributeError: install_layout`). Install it as:

```bash
pip install "setuptools<60" wheel
pip install --no-build-isolation hunspell==0.5.5
```

Confirm Hunspell:

```bash
echo teh | hunspell -d en_US -a
```

## Tests (CPU)

```bash
python -m pytest tests/ -q
```

| Test | File |
|------|------|
| Byte roundtrip, vocab contiguity | `tests/test_byte_encoding.py` |
| Serialization spans, padding | `tests/test_serialization.py` |
| Gold label, determinism, context noise, no benchmark leak | `tests/test_dataset.py` |
| Typo realism vs the public misspelling list | `tests/test_typo_gen.py` |
| Shapes, param counts, pooling equivalence, loss | `tests/test_model.py` |
| Tiny overfit | `tests/test_tiny_overfit.py` |

`tests/test_dataset.py` fails the build if any training-construction file so
much as mentions the locked benchmark.

## Build training data

Synthetic only (WikiText-103 raw, CC BY-SA). See [data/README.md](data/README.md).

```bash
python scripts/download_sources.py
python scripts/build_training_data.py --target-train 4000000 --target-valid 60000
```

Three passes: count the vocabulary, build a `word -> typos -> Hunspell pool`
table, then instantiate examples by dropping precomputed typos into sentences.
Hunspell is called once per **unique typo** rather than once per example, so
build cost no longer scales with dataset size.

Outputs: `data/processed/{train,validation}.parquet`, `data_stats.json`,
`manifest.json`, `artifacts/hunspell_metadata.json`.

## Train

```bash
python scripts/train.py --config configs/train_sanity.yaml   # cheap sanity
python scripts/train.py --config configs/train_full.yaml     # full experiment
```

On OOM, lower `microbatch` and raise `grad_accumulation` in
`configs/train_full.yaml` so the effective batch stays 512.

Checkpoints land in `artifacts/model/`.

## Benchmark BEA-60K

Do **not** commit BEA files.

```bash
python scripts/download_bea60k.py
python scripts/benchmark_aspell.py
python scripts/benchmark_bea60k.py --model artifacts/model --output reports/bea60k
```

## GPU runs

See [scripts/runpod/README.md](scripts/runpod/README.md). Pods bill for as long
as they exist — always finish with `scripts/runpod/terminate.py --all`.

## Design constraints

- Byte vocabulary 0..255 plus 24 specials (280 total). No BPE/SPM.
- One forward pass scores all candidates.
- Hunspell suggestion order is preserved (`CAND_0`…`CAND_15`).
- Hunspell is the only candidate generator.
- Unicode NFC; no global lowercasing.
- Seed **1337** for data and training.
