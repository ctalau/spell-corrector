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
2. Fast unit tests (`pytest tests/ -q -m "not slow" --timeout=180 --timeout-method=thread` under
   `timeout 600`). Tests that download `answerdotai/ModernBERT-base`, run CUDA
   extract, or train the byte-level overfit set are `@pytest.mark.slow` and are
   skipped here so a hung HF download cannot burn GPU hours. Default coverage
   uses `DummyTokenizer` / `DummyBackbone` on CPU (seconds). Real-encoder smoke
   is step 4 (`cache_frozen_features.py --smoke-examples`), not pytest.
   Opt-in locally: `RUN_SLOW=1 pytest tests/ -m slow`.
3. Synthetic data build if parquet is missing (400k/40k targets, then a
   deterministic 200k-by-example-id subset plus D-pair).
4. Smoke feature cache (10k examples).
5. Full 200k + D-pair cache, plus a fixed BEA-60K 1k feature cache for the
   monitoring gate. **Not** a full BEA run.
6. Train H1 (scalar MLP) → H2 (linear) → H3 (3854→256→64→1 MLP).
   H2 linear uses AdamW **3e-4**, input LayerNorm, and train-set encoder RMS
   scaling. If linear still exits non-zero (`nan_seen` is exit 2), the shell
   records `artifacts/frozen/heads/linear/failure.json` and **continues to
   H3**. Only a scalar (H1) failure aborts the experiment. H3 `rc=2` still
   gets one predeclared 3e-4 retry.
7. BEA-1k gate during an arm: every 30 minutes, **every 200 steps**, and
   **once per epoch** (head-only training finishes in minutes, so the
   wall-clock gate never fired on pod `ww31jci1imkyo9`). Stop that arm if
   overall **and** conditional accuracy fail to improve by more than 1
   percentage point versus the best prior BEA-1k checkpoint. Synthetic
   D-pair patience-2 still applies. `summary.json` writes JSON `null` for
   non-finite floats (the H2 NaN run produced invalid raw `NaN`).
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
extraction and the real ModernBERT tokenizer download are `@pytest.mark.slow`
(skipped on the pod gate; skip without CUDA / without `RUN_SLOW=1` locally).

Caches live under `artifacts/frozen_cache/` (gitignored). Head weights, plots
and metrics go to `artifacts/frozen/`.

## H2 NaN abort (pod ww31jci1imkyo9)

Commit `5a3b7ca` trained H0 (d-pair overall 81.56%, cond 82.04%) and H1
(best val_cond 80.51%, `nan_seen=false`). H2 linear set `nan_seen=true`
after about one step: unnormalized 3854-d encoder features (including `c*t`
and `|c-t|`) at lr `1e-3` overflowed. The experiment shell then died with
`FROZEN EXPERIMENT FAILED: train linear rc=0` — a bash bug: after
`if cmd; then return 0; fi`, `local rc=$?` is the status of the successful
`if` compound (0), so a NaN exit-2 was reported as rc=0 and treated as
fatal. H3 never started, and no BEA-1k checkpoints were written because
training finished in minutes.

Fixes: capture `rc=$?` immediately after the train command; continue to
MLP after a linear failure; LayerNorm + encoder RMS scale + linear lr
`3e-4`; skip non-finite batches and return 2 when `nan_seen`; evaluate
BEA-1k at least once per epoch and every 200 steps; serialize non-finite
summary values as `null`.
