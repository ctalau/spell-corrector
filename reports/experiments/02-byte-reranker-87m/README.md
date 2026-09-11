# Experiment 2 — 87M byte-level Hunspell reranker

| | |
|---|---|
| **Status** | **completed** — trained, benchmarked, target missed |
| **When** | 2026-09-09, pod `xwvjd2w980kk00`, commit `b9b66cf275a062390e4428674eb27396461b1dcb` |
| **Headline result** | **64.82% overall** top-1 on BEA-60K (n=68,429 word errors), 80.14% conditional. Target was 75%. Aspell 60.56%, Hunspell top-1 53.67%. |
| **Cost** | ~$2.95 to DONE (3.74 pod-hours × $0.79/h, Community L40S). ~$2 more was burned across the eight preceding failed pods — see [RUN_NOTES.md](RUN_NOTES.md). |
| **What it settled** | Scaling 28M → 87M and 235k → 2.08M examples buys ~+2.3 pp overall (62.56% → 64.82%) and lifts conditional accuracy 77.84% → 80.14%. The training-data fix worked: the post-filter ED1/ED2/ED3+ mixture (81.7/16.9/1.4) closely matches BEA's in-pool errors (83.0/15.7/1.3). |
| **What it left open** | Conditional accuracy is still ~12.3 pp below synthetic validation (92.42%), i.e. the gap is generalization, not fit. Pool coverage (80.89% at 16 slots) caps overall accuracy; 75% would need 92.72% conditional. `gold0_fraction` never bound (76.6% of training rows have gold at slot 0 vs BEA's 66.8%). See §"Known issues" in [RUN_NOTES.md](RUN_NOTES.md). |
| **Plan** | [PLAN.md](PLAN.md) (rationale, changes 1-9, run configuration) |
| **Operational notes** | [RUN_NOTES.md](RUN_NOTES.md) (bug list, Runpod procedure, phase timings, budget) |
| **Artifacts** | [`reports/bea60k/`](../../bea60k/), [SUMMARY.txt](SUMMARY.txt), [data_stats.json](data_stats.json), [manifest.json](manifest.json), [STATUS.txt](STATUS.txt), [`reports/training_loss.png`](../../training_loss.png), [`reports/run3_partial/`](../../run3_partial/) (the OOMed predecessor run) |

---

**DID WE HIT 75%? NO** — overall **64.82%** (target 75%).

Pod `xwvjd2w980kk00` · Community L40S @ $0.79/hr · commit `b9b66cf275a062390e4428674eb27396461b1dcb` · STATUS=SUCCESS.

## Headline

| Metric | Value |
|--------|-------|
| Model overall success | **64.82%** |
| Model conditional accuracy | **80.14%** |
| Pool ceiling (Hunspell oracle@16) | **80.89%** |
| Hunspell top-1 | 53.67% |
| Aspell top-1 | 60.56% |
| vs Aspell | +4.26 pp |
| Hit 75%? | **NO** |

### Residual error split (of all BEA word errors)

| Bucket | Share | Meaning |
|--------|------:|---------|
| Gold outside Hunspell pool | 19.11% | Unreachable with current candidate generator |
| Model picked wrong (gold in pool) | 16.07% | Reachable — better reranking could help |
| Model correct | 64.82% | |

Conditional needed to hit 75% overall at this ceiling: **92.72%** (had 80.14%).

## Setup

- **GPU**: NVIDIA L40S, PyTorch 2.4.1+cu124, CUDA 12.4, bf16=True
- **Params**: **87,352,577**
- **Config**: `configs/train_full.yaml` · seed 1337
- **Batch**: microbatch 128 × grad_accum 4 → effective 512
- **Train / valid sizes**: **2,081,171** / **20,773** (targets were 3,000,000 / 60,000; builder produced what the WikiText source supported)
- **Optimizer steps**: 8,130
- **Dataset hashes**: train `77f615b157f4…`, valid `7c82182a3c33…`
- **Preflight peak**: 30.0 / 44.5 GiB (train_full); train sampler peak **32853 MiB (32.08 GiB)**

## Timing & cost

- Pod start: 2026-09-09 12:18:46 Bucharest
- Experiment `done` marker: ~16:03:05 Bucharest (t=13414s in run.log)
- Training wall clock: **3.11 h** (11211 s)
- Full pod wall (start→DONE): **~3.74 h**
- Approximate cost to DONE: **~$2.95** (3.74 h × $0.79); final cost depends on terminate time

## Validation (in-distribution)

- Best val loss: **0.2072**
- Best val top-1: **92.42%**
- Last val @gold-index-0 / nonzero: 97.95% / 82.61%

## BEA-60K detail

- Sentence pairs: 63,044; word errors: 68,429; candidate slots: 16
- Hunspell detection rate: 99.19%
- Hunspell oracle@10: 80.34%
- Hunspell top-1 conditional (among solvable): 66.35%

## Artifacts

Local fetch path: `/workspace/spell-corrector-exp2-artifacts`

- `artifacts/train_summary.json`, `artifacts/train_metrics.jsonl`, `artifacts/vram_sampler.csv`
- `artifacts/model/` (safetensors + config + manifests)
- `reports/bea60k/results.json` + example JSONLs + histogram
- `reports/training_loss.png`, `run.log`, `STATUS`/`DONE`

## Takeaway

Scaling to ~87M + ~2.08M train examples beat Aspell by ~4.3 pp (64.8% vs 60.6%) and reached **80.1% conditional** accuracy, but **pool coverage (~80.9%) caps overall at well below 75%** unless Hunspell recall improves. Most of the remaining gap to 75% is **unreachable under the current candidate pool**, not pure reranker error.
