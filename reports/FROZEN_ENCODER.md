# Frozen ModernBERT selector — how to run

Implementation of [FROZEN_ENCODER_PLAN.md](FROZEN_ENCODER_PLAN.md). The encoder
is **answerdotai/ModernBERT-base**, frozen, with a trainable selector head. No
LoRA and no backbone finetuning.

## Pinned Hugging Face revision

Resolved from `answerdotai/ModernBERT-base` HEAD on 2026-09-09:

| Field | Value |
|---|---|
| Model id | `answerdotai/ModernBERT-base` |
| Revision | `8949b909ec900327062f0ebf497f51aef5e6f0c8` |
| `transformers` | `>=4.48,<5` (ModernBERT; 5.x needs torch>=2.5) |
| Max length | 512 |
| Hidden size | 768 |
| Candidate policy | Hunspell **raw first ten** (slice before dedup/length filter) |

Recorded in `configs/train_frozen_modernbert.yaml` and
`spelling_reranker/frozen_encoder.py` (`DEFAULT_ENCODER_REVISION`).

## Launch a self-driving pod

From a machine with `RUNPOD_KEY` set:

```bash
python scripts/runpod/launch.py \
  --experiment frozen \
  --config configs/train_frozen_modernbert.yaml
```

Frozen pods default to **`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`** (Python 3.11 so `hunspell==0.5.5` builds, torch 2.4.1, `transformers>=4.48,<5` for ModernBERT). Do **not** use `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (cu128 / torch 2.8 / py3.12): hunspell 0.5.5 cannot build there (`setuptools<60` / `ImpImporter`) until the binding is fixed for 3.12.

Equivalent: `--config configs/train_frozen_modernbert.yaml` alone sets
`EXPERIMENT=frozen`. The pod prefers an RTX A5000 (24 GB), 100 GB disk, then
runs `scripts/runpod/run_frozen_experiment.sh`:

1. Fail-fast CUDA check, install `transformers>=4.48,<5`, and assert `AutoModel` imports.
2. Unit tests.
3. Synthetic data build if parquet is missing (400k/40k targets, then a
   deterministic 200k-by-example-id subset plus D-pair).
4. Smoke feature cache (10k examples).
5. Full 200k + D-pair cache, plus a fixed BEA-60K 1k feature cache for the
   monitoring gate. **Not** a full BEA run.
6. Train H1 (scalar MLP) → H2 (linear) → H3 (3854→256→64→1 MLP).
7. Every 30 minutes during an arm: checkpoint and BEA-1k eval. Stop that arm
   if overall **and** conditional accuracy fail to improve by more than 1
   percentage point versus the best prior 30-minute checkpoint. Synthetic
   D-pair patience-2 still applies.
8. D-pair evaluation of H0 (Hunspell first candidate) and H1–H3.
9. HTTP artifact server (`/run.log`, `/STATUS`, `/artifacts/...`) as in the
   existing bootstrap.

Terminate with `python scripts/runpod/terminate.py --all`.

## Local commands

```bash
python scripts/cache_frozen_features.py --config configs/train_frozen_modernbert.yaml --smoke-examples 10000
python scripts/cache_frozen_features.py --config configs/train_frozen_modernbert.yaml --bea-limit 1000
python scripts/train_frozen_selector.py --config configs/train_frozen_modernbert.yaml --arm scalar
python scripts/train_frozen_selector.py --config configs/train_frozen_modernbert.yaml --arm linear
python scripts/train_frozen_selector.py --config configs/train_frozen_modernbert.yaml --arm mlp
python scripts/evaluate_frozen_selector.py --config configs/train_frozen_modernbert.yaml --split d-pair --checkpoint artifacts/frozen/heads/mlp
python scripts/evaluate_frozen_selector.py --config configs/train_frozen_modernbert.yaml --bea-limit 1000 --checkpoint artifacts/frozen/heads/mlp
```

CPU unit tests use a dummy encoder (`--dummy-encoder` on the cache script). GPU
extraction tests skip when CUDA is absent.

Caches live under `artifacts/frozen_cache/` (gitignored). Head weights, plots
and metrics go to `artifacts/frozen/`.
