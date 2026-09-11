# Spell Corrector

Byte-level **contextual spelling reranker**. Hunspell proposes candidates; a
bidirectional Transformer picks exactly one.

This repository is the experiment line from the original charter, now at
[reports/experiments/01-byte-reranker-28m/PLAN.md](reports/experiments/01-byte-reranker-28m/PLAN.md):

> Beat Aspell's top-1 spelling correction accuracy on BEA-60K.

The model is **not** generative. Training labels are Hunspell ranks on synthetic
typos over WikiText-103. BEA-60K is a locked final benchmark and is never used
for training, validation, or hyperparameter selection.

## Start here: the experiment log

**[reports/README.md](reports/README.md)** is the index of everything that has
been tried — seven experiments, every measured number against BEA-60K in one
table with its sample size, what has been ruled out, and what is queued. Read it
before proposing anything.

The short version, as of 2026-09-11:

| | |
|---|---|
| Best system measured on the **full** benchmark | the trained 87M byte-level reranker — **64.82%** overall, n=68,429 (Aspell 60.56%, Hunspell top-1 53.67%) |
| Best system measured **at all** | `google/gemma-4-E2B-it` prompted zero-shot in open mode — **90.0%** overall, but **n=100** on a CPU box, so ±8-10 pp |
| In flight | gemma-4-E2B q4_0 on GPU via llama.cpp + DSPy prompt optimization |

## Where the accuracy comes from

Experiment 1 reached **62.56%** overall (vs Aspell 60.55%); experiment 2 scaled
it to **64.82%** (conditional 80.14%), short of its 75% target. Decomposed, for
experiment 1:

```
overall = P(gold in Hunspell pool) x P(model picks gold | it is there)
62.56%  =        80.38%            x            77.84%
```

Hunspell is fixed as the only candidate source, so the first factor is a hard
ceiling around 81%. Everything in experiment 2 targeted the second factor:

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

The `hunspell` binding builds from source and fails against setuptools >= 60
(`AttributeError: install_layout`). On Python 3.11 and older:

```bash
pip install "setuptools<60" wheel
pip install --no-build-isolation hunspell==0.5.5
```

On Python 3.12 that workaround does not apply — there is no `distutils` for old
setuptools to patch — so use the distro package, which is the same version
already compiled for the interpreter:

```bash
sudo apt-get install -y python3-hunspell
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
| Frozen encoder selector | `tests/test_frozen_encoder.py` |

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

## Frozen ModernBERT selector

Pilot from [reports/experiments/04-frozen-encoder/PLAN.md](reports/experiments/04-frozen-encoder/PLAN.md)
(**aborted mid-run** — see its front matter before relying on anything here): freeze
`answerdotai/ModernBERT-base` (Hugging Face revision
`8949b909ec900327062f0ebf497f51aef5e6f0c8`, resolved 2026-09-09) and train only
a small selector. Launch on Runpod with:

```bash
python scripts/runpod/launch.py \
  --experiment frozen \
  --config configs/train_frozen_modernbert.yaml
```

That defaults to the cu124 / Python 3.11 image (`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`) so Hunspell and `transformers>=4.48,<5` both work. Do not use the cu128 / py3.12 torch 2.8 image until Hunspell is fixed for 3.12. See [reports/experiments/04-frozen-encoder/README.md](reports/experiments/04-frozen-encoder/README.md).

Local GPU path (after synthetic parquet exists):

```bash
python scripts/cache_frozen_features.py --config configs/train_frozen_modernbert.yaml --smoke-examples 10000
python scripts/cache_frozen_features.py --config configs/train_frozen_modernbert.yaml --bea-limit 1000
python scripts/train_frozen_selector.py --config configs/train_frozen_modernbert.yaml --arm scalar
python scripts/train_frozen_selector.py --config configs/train_frozen_modernbert.yaml --arm linear
python scripts/train_frozen_selector.py --config configs/train_frozen_modernbert.yaml --arm mlp
python scripts/evaluate_frozen_selector.py --config configs/train_frozen_modernbert.yaml --split d-pair --checkpoint artifacts/frozen/heads/mlp
python scripts/evaluate_frozen_selector.py --config configs/train_frozen_modernbert.yaml --split bea --bea-limit 1000 --checkpoint artifacts/frozen/heads/mlp
```

Do not train or tune on BEA-60K. The 1k subset is a monitoring gate only.
See [reports/experiments/04-frozen-encoder/README.md](reports/experiments/04-frozen-encoder/README.md).

## LLM judge (no training)

A separate track prompts a small instruction-tuned LLM to correct the typo
directly, in one of four answer modes (pick a candidate index, answer freely with
the candidates as a hint, self-generate candidates by beam search, or rewrite the
whole sentence). It needs no training and is currently the highest-scoring thing
in the repo, on a small sample.

```bash
python scripts/llm_judge_bea60k_cpu.py --help    # CPU / llama.cpp track, four answer modes
python scripts/llm_judge_bea60k.py --help        # GPU-targeted index-mode harness
```

Results and the full method:
[experiment 5](reports/experiments/05-llm-judge-index/README.md) and
[experiment 6](reports/experiments/06-llm-judge-cpu-llamacpp/README.md).

## Benchmark BEA-60K

BEA-60K is a **locked** benchmark: never train, validate, tune or prompt-search
on it, and do **not** commit BEA files.

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
