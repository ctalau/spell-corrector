# Milestone 4 Report — Direct spelling correction (Qwen3.5-0.8B LoRA)

- Seed: **1337**
- Eval: frozen BEA-100 (same as M1/M1b/M2/M3); **held out of train+val**
- Base model: `Qwen/Qwen3.5-0.8B` rev `2fc06364715b967f1860aea9cf38778875588b17`
- Method: **fresh LoRA** from base (does **not** continue from M3 reranker adapter)
- Objective: marked sentence → **only the corrected word** (not candidate reranking)
- Prompt: `prompts/direct_correct_v1.txt`

## Hold-out (hard rule)

Every example in the seed-1337 BEA-100 was excluded from train and val by:
1. `error_index` membership in `data/frozen_sample_100.json`
2. NFC-match of noisy sentence (TYPO tags stripped) against any frozen sample

Documented in `data/direct_train/exclusion_meta.json` (**0 leaks** verified in train+val).
BEA word-errors excluded: **134** (100 indices + 34 sentence overlaps). Test = frozen 100 only.

## Training data

| Source | Typos / pairs | Rows | Notes |
|---|---:|---:|---|
| BEA-60K (non-holdout) | 20000 train + 500 val | 20000 / 500 | Real sentence context with `<TYPO>…</TYPO>` |
| Wikipedia misspellings | 4066 pairs | 4066 | Minimal marked sentence `<TYPO>{typo}</TYPO>` |
| **Train total** | — | **24,066** | One row per typo (no negatives) |

## Training recipe

| Setting | Value |
|---|---|
| Method | Fresh LoRA from base (not continued from M3) |
| LoRA r / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | q,k,v,o,gate,up,down_proj |
| Trainable params | ~6.4M / 859M (**0.74%**) |
| LR | 2e-4 |
| Epochs / steps | 1 epoch / **753** optimizer steps |
| Batch × accum | 2 × 16 (effective **32**) |
| Max seq len | **256** |
| Gradient checkpointing | **ON** |
| Mid-train eval | disabled (VRAM) |
| Precision | bf16 |
| Final train loss | **0.354** |
| First / last logged loss | **0.755** → **0.244** |
| Train wall | ~4572 s (~1.27 h GPU) |
| Peak VRAM | ~**3.9 GiB** |

## Results — Acc@1 on frozen 100

| System | Acc@1 (casefold) | Acc@1 (exact) | Notes |
|---|---:|---:|---|
| Hunspell top-1 | 60% | — | Prior baseline |
| M3 FT reranker (classic mix) | **85%** | — | Needs candidate pool |
| M3 FT reranker (full_union) | 87% | — | |
| **M4 direct LoRA (this work)** | **84%** | **84%** | No candidates; word-only output |
| Luna freeform first-try | 94% | — | API / large model |
| Jev Choice (approx, prior) | ~91% | — | API chooser over lists |

### Latency (GPU, RTX 3090)

| Metric | Value |
|---|---|
| p50 | **0.107 s** / sample |
| p90 | 0.146 s |
| p99 | 0.194 s |
| mean | 0.128 s |
| wall (100 samples) | **12.9 s** |

For comparison, M3 pointwise rerank over 3 pools × up to 100 candidates was ~5.65 s/sample wall-averaged.

## Plain English

We taught the same small model to **fix the misspelled word itself**, given the sentence with the typo marked — instead of picking from a list of guesses.

- On the frozen 100-test, it got the right word **84 times out of 100**.
- That beats Hunspell’s first suggestion (**60%**) and nearly matches our Milestone-3 list-picker (**85%**), while needing **no Hunspell/BM25/dense candidates** and answering in about **a tenth of a second** on a 3090.
- Big API models (Luna freeform ~94%, Jev Choice ~91%) are still ahead on accuracy.

In short: a tiny LoRA turns 0.8B into a fast standalone corrector that is already in the same ballpark as the fine-tuned reranker, with a much simpler serving path.

## Loss curve note

Logged every 20 optimizer steps (37 points over 753 steps / 1 epoch).

| Phase | Approx. steps | Avg logged loss | End of phase |
|---|---|---:|---:|
| Start | first log | — | **0.755** |
| Q1 | early | 0.47 | 0.42 |
| Q2 | | 0.36 | 0.32 |
| Q3 | mid | 0.34 | 0.31 |
| Q4 | | 0.31 | 0.30 |
| Q5 (last ~20%) | late | **0.309** | final log **0.244** |
| Trainer reported train_loss | full epoch | — | **0.354** |

**Did it improve until the end?** Yes overall: loss fell sharply in the first ~100 steps, then slowly through mid-train; the last log (0.244) is the best single reading. No divergence. One epoch was enough for strong Acc@1; a second epoch might help hard cases but risks overfitting wiki-style word-only rows.

## Mistakes (16 misses → Acc@1 84%)

Scoring is case-insensitive (`nfc_lower`); exact matched casefold on this set (84%/84%).

Examples of failure modes:
- Near-miss morphology: `crimbimg`→`crimbing` (not `climbing`); `currancies`→`currency` (not `currencies`); `sorcess`→`sorcerer` (not `sorceress`)
- Context-wrong synonym / related: `dialoging`→`dialoguing` (gold was `talking`); `pollusions`→`pollutions` (gold `pollutants`)
- Ambiguous / rare: `thursty`→`toughness` (gold `thirst`); `vacab`→`vacation` (gold `vocabulary`); `Miken` left unchanged (gold `McCain`)

## RunPod

- Pod id: **`t3r0ha2aa9sqja`** (Community RTX 3090, **$0.22/hr**, host CUDA 13.0)
- Image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Stack: torch `2.5.1+cu124` + transformers `5.18.0.dev0` + peft
- Billed roughly: bootstrap ~5 min + train ~1.27 h + eval ~1 min + sync ≈ **~1.5–1.7 h** → cost estimate **~$0.33–0.40**
- Pod **terminated** after artifact sync

## Artifacts

- `prompts/direct_correct_v1.txt`
- `scripts/build_direct_train.py`, `scripts/train_direct.py`, `scripts/eval_direct.py`
- `configs/milestone4_direct.yaml`
- `data/direct_train/{train,val}.jsonl`, `exclusion_meta.json`, `holdout_error_indices.json`
- `models/qwen35_0_8b_direct_lora/` (LoRA adapters + `train_meta.json` + `loss_curve.json`)
- `results/predictions_m4_direct.jsonl`, `results/metrics_milestone4.json`
- `results/runs.csv` (appended)
- `logs/m4_train_eval.log`, `logs/m4_bootstrap.log`
- Synced copy: `/workspace/spell-corrector/artifacts/spell_slm_m4/`

## Commands (repro)

```bash
cd /workspace/spell-slm-candidate-rerank
python scripts/build_direct_train.py --config configs/milestone4_direct.yaml
# On GPU pod (after bootstrap):
python scripts/train_direct.py --config configs/milestone4_direct.yaml
python scripts/eval_direct.py --config configs/milestone4_direct.yaml
```
