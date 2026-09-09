# Experiment report — 28M Hunspell reranker vs Aspell on BEA-60K

Generated after the CUDA training + BEA-60K run on Runpod Community RTX 3090 (pod `znrh3wrazvkk26`).

## 1. Goal

Beat Aspell top-1 spelling correction accuracy on BEA-60K with a ~28M
byte-level contextual reranker over Hunspell's top-10 suggestions.

## 2. Data sources

- Train/valid: synthetic typos from WikiText-103 raw (CC BY-SA). See `data/README.md`.
- Benchmark: NeuSpell BEA-60K (downloaded, **not committed**).

## 3. Dataset construction

| Split | Target | Actual |
|-------|--------|--------|
| Train | 240,000 (min 150,000) | 235,626 |
| Valid | 20,000 (min 10,000) | 19,642 |

Hashes: `data/processed/manifest.json`. Seed **1337**. Section-heading boilerplate dropped; train/valid sentence-hash overlap = 0.

## 4. Hunspell version / dictionary

- Hunspell: 1.7.2 (`hunspell-en-us` 2020.12.07)
- Dictionary hashes: see `artifacts/model/hunspell_metadata.json`
- Aspell: 0.60.8.1 (`aspell-en` 2020.12.07)

## 5. Number of examples

Train 235,626 / valid 19,642. BEA word errors: 68,429 from 63,044 sentence pairs.

## 6. Model architecture

Byte-level bidirectional Transformer encoder (~28M). Vocabulary 0..255 + 18 specials (274). One forward pass scores Hunspell candidates `CAND_0`…`CAND_9` with padding masks. Config: `configs/model_28m.yaml`.

## 7. Exact parameter count

**27,904,129** trainable parameters.

## 8. Training hyperparameters

From `configs/train_full.yaml`:

- seed 1337, bf16, AdamW lr 3e-4, betas [0.9, 0.95], weight_decay 0.10
- cosine schedule, warmup_ratio 0.05, max_grad_norm 1.0
- effective batch 256 (microbatch 128 x grad_accum 2)
- 3 epochs, 2760 optimizer steps
- git commit at train time: `edf5fafe274be7862bc98d816dbae79b15ae96db`

## 9. GPU used

NVIDIA GeForce RTX 3090 (Community cloud), PyTorch 2.8.0+cu128, CUDA 12.8, bf16=True.

## 10. Runpod runtime and approximate compute cost

- Training wall clock: **35.0 min** (2101 s)
- Peak reserved VRAM sampler: **22264 MiB** (~21.7 GiB)
- Rate: $0.22/hr Community RTX 3090
- Approximate full-pod cost for this experiment (setup + sanity + train + BEA): **~$0.28** (about 1.25 h x $0.22)

## 11. Peak VRAM

**22264 MiB** (nvidia-smi memory.used sampler every 5s). Train metrics also report `gpu_mem_reserved` about 21.4 GiB.

## 12. Training loss curve / table

See `reports/training_loss.png` and `artifacts/train_metrics.jsonl`.

| Checkpoint | Loss |
|------------|------|
| Initial (step 20 train) | 1.2954 |
| Final (step 2760 train) | 0.2376 |
| Best validation | 0.2730 |

## 13. Validation loss

Best validation loss: **0.2730**

## 14. Validation candidate accuracy

Best val top-1: **90.60%**; last val top-3: **98.56%**

## 15-20. BEA-60K results

| System | Overall correction accuracy |
|--------------------------------|-----------------------------|
| Aspell top-1 | 60.55% |
| Hunspell top-1 | 53.67% |
| Hunspell oracle@10 | 80.38% |
| 28M reranker + Hunspell | 62.56% |

| Metric | Value |
|-----------------------------------------|-------|
| BEA errors | 68,429 |
| Hunspell detected | 99.19% |
| Gold in Hunspell top 10 | 80.38% |
| Model accuracy when gold in top 10 | 77.84% |
| Model overall accuracy | 62.56% |
| Aspell overall top-1 accuracy | 60.55% |

### DID WE BEAT ASPELL? **YES**

Absolute difference: **+2.01 percentage points** (model 62.56% vs Aspell 60.55%).

## 21. Gold-index histogram

| Bin | Count | % |
|-----|------:|--:|
| 0 | 36725 | 53.67% |
| 1 | 8282 | 12.10% |
| 2 | 3930 | 5.74% |
| 3 | 2662 | 3.89% |
| 4 | 1403 | 2.05% |
| 5 | 767 | 1.12% |
| 6 | 472 | 0.69% |
| 7 | 293 | 0.43% |
| 8 | 296 | 0.43% |
| 9 | 149 | 0.22% |
| present_later | 0 | 0.00% |
| not_present | 12865 | 18.80% |
| hunspell_did_not_flag | 552 | 0.81% |
| hunspell_no_suggestions | 33 | 0.05% |

Plots/CSV: `reports/bea60k/hunspell_gold_index_histogram.png` (+ `.csv`, `.json`).

## 22. Example improvements / failures

Sample JSONL under `reports/bea60k/examples_*.jsonl` (fixed_top1, damaged_top1, both_failed, gold_outside_top10).

## 23. Main conclusions

1. The ~28M byte-level reranker **beats Aspell by ~2.01 pp** on BEA-60K overall top-1.
2. It also lifts Hunspell top-1 (53.67% -> 62.56%).
3. Conditional accuracy when gold is in top-10 is strong (77.84%), but oracle@10 is 80.38% — remaining gap is largely **candidate generation**.
4. Validation top-1 about 90.6% on synthetic data transfers reasonably to BEA conditional accuracy.

## 24. Recommended next experiment

- Improve candidate generation / widen oracle (edit-distance hybrids, phonetic).
- Ablation A (shared CAND token) and Ablation B (no context) from PLAN section 25.
- Mild authentic-data fine-tune if redistribution allows.
- Reduce VRAM / microbatch if targeting smaller GPUs.

## Artifacts

- Model: `artifacts/model/model.safetensors` (+ `best_val_*.safetensors`, `config.json`, `training_manifest.json`)
- Metrics: `artifacts/train_summary.json`, `artifacts/train_metrics.jsonl`
- BEA: `reports/bea60k/results.json` (BEA raw data not committed)
