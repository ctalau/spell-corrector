# Experiment log

Everything this repository has tried, what it measured, and what is queued next.
Start here; the root [README](../README.md) covers installation and how to run
things.

> **BEA-60K is a locked benchmark.** Never train, validate, tune or prompt-search
> on it. It is downloaded by `scripts/download_bea60k.py`, never committed. Every
> number on this page is a measurement on it; none of them may be fed back into
> model or prompt selection. `tests/test_dataset.py` fails the build if any
> training-construction file so much as mentions the benchmark.

## 60-second answer

- **Best system so far:** `google/gemma-4-E2B-it` (bf16, transformers), prompted
  zero-shot in **open mode** — Hunspell's top-8 shown only as a hint, model free
  to answer any word — **90.0% overall on BEA-60K, n=100**, on a 4-vCPU CPU box.
  Latency for that specific run is not usable (a disk-I/O stall; see
  [experiment 6](experiments/06-llm-judge-cpu-llamacpp/README.md)); the same
  model in open mode on q4_0/llama.cpp is 87.0% at a p50 of 960ms. No training,
  no GPU, no fine-tuning.
- **Best system measured at full scale:** the trained 87M byte-level reranker,
  **64.82% overall, n=68,429** (the entire benchmark), conditional 80.14%,
  +4.26 pp over Aspell. That is the only model ever run on all of BEA-60K.
  **These two numbers are not interchangeable** — 90.0% on 100 examples carries
  roughly ±8-10 pp of binomial noise, and the trained model has never been given
  the small-sample treatment that would make the comparison fair.
- **Ruled out:** raising overall accuracy past ~81% by better *reranking* alone
  (Hunspell's pool simply lacks the gold word for ~19% of errors);
  `openbmb/MiniCPM5-1B` as a judge (39.0% — *below* Hunspell top-1 on the same
  rows, i.e. worse than not reranking at all, and 4x slower);
  q4_0 as a free lunch (consistently 2-3 points worse than bf16, zero wins in
  200 paired index/open examples); adding Aspell to the candidate pool (measured
  to lift the ceiling to 87.7% on a 6,000-error sample, but out of scope by
  explicit user decision).
- **Queued:** [experiment 7](experiments/07-gemma4-gpu-dspy/README.md) —
  gemma-4-E2B q4_0 on GPU via llama.cpp plus DSPy prompt optimization. Running
  now; numbers pending.

## Every measured system on BEA-60K

One metric: **overall top-1 correction accuracy**, i.e. correct corrections
divided by *all* eligible word errors, including those where the candidate pool
never contained the gold word. Sample sizes differ by nearly three orders of
magnitude — the `n` column is the first thing to read, not the last.

| System | Mode / backend | n | Overall | Conditional | Hunspell top-1 on the same rows | p50 latency | Source |
|---|---|---:|---:|---:|---:|---:|---|
| Hunspell top-1 | candidate generator alone | 68,429 | 53.67% | — | — | — | [bea60k/results.json](bea60k/results.json) |
| Aspell top-1 | external baseline | 68,429 | 60.56% | — | — | — | [bea60k/results.json](bea60k/results.json) |
| **Exp 1** — 28M byte reranker | trained, 10 slots | not recorded | 62.56% | 77.84% | — | not recorded | [exp 1](experiments/01-byte-reranker-28m/README.md) |
| **Exp 2** — 87M byte reranker | trained, 16 slots | **68,429** | **64.82%** | 80.14% | 53.67% | not recorded | [exp 2](experiments/02-byte-reranker-87m/README.md) |
| *Hunspell oracle@10* | *ceiling, not a system* | 68,429 | *80.34%* | — | — | — | [bea60k/results.json](bea60k/results.json) |
| *Hunspell oracle@16* | *ceiling, not a system* | 68,429 | *80.89%* | — | — | — | [bea60k/results.json](bea60k/results.json) |
| Qwen3.5-0.8B | index, bf16 CPU | 100 | 60.0% | 72.3% | 59.0% | 541ms | [exp 5](experiments/05-llm-judge-index/README.md) |
| Qwen3.5-0.8B | index, bf16 CPU, 5-min timed | 529 | 56.5% | 68.6% | 56.1% | 530ms | [exp 5](experiments/05-llm-judge-index/README.md) |
| MiniCPM5-1B | index, bf16 CPU | 100 | 39.0% | 47.0% | 59.0% | 2,183ms | [exp 5](experiments/05-llm-judge-index/README.md) |
| MiniCPM5-1B | index, bf16 CPU, 5-min timed | 139 | 37.4% | 45.6% | 56.8% | 2,146ms | [exp 5](experiments/05-llm-judge-index/README.md) |
| gemma-4-E2B-it | index, bf16 CPU | 100 | 83.0% | 100% | 59.0% | 632ms&#42; | [exp 5](experiments/05-llm-judge-index/README.md) |
| gemma-4-E2B-it | index, bf16 CPU, 5-min timed | 486 | 78.8% | 95.3% | 56.0% | 605ms | [exp 5](experiments/05-llm-judge-index/README.md) |
| gemma-4-E2B-it | **open**, bf16 CPU | 100 | **90.0%** | 96.4% | 59.0% | not comparable&#42;&#42; | [exp 6](experiments/06-llm-judge-cpu-llamacpp/README.md) |
| gemma-4-E2B-it | beam (no Hunspell list), bf16 CPU | 100 | 88.0% | 92.8% | 59.0% | 14,800ms | [exp 6](experiments/06-llm-judge-cpu-llamacpp/README.md) |
| gemma-4-E2B-it | sentence rewrite, bf16 CPU | 100 | 73.0% strict / **88.0%** ignoring punctuation | 83.1% strict | 59.0% | 5,980ms | [exp 6](experiments/06-llm-judge-cpu-llamacpp/README.md) |
| gemma-4-E2B-it-qat-**q4_0** | index, llama.cpp CPU | 100 | 81.0% | 97.6% | 59.0% | 744ms | [exp 6](experiments/06-llm-judge-cpu-llamacpp/README.md) |
| gemma-4-E2B-it-qat-**q4_0** | open, llama.cpp CPU | 100 | 87.0% | 92.8% | 59.0% | 960ms | [exp 6](experiments/06-llm-judge-cpu-llamacpp/README.md) |
| gemma-4-E2B-it-qat-**q4_0** | sentence rewrite, llama.cpp CPU | 100 | 66.0% strict / 86.0% ignoring punctuation | 74.7% strict | 59.0% | 2,118ms | [exp 6](experiments/06-llm-judge-cpu-llamacpp/README.md) |
| gemma-4-E2B q4_0 + DSPy | GPU, llama.cpp | — | *running* | *running* | — | *running* | [exp 7](experiments/07-gemma4-gpu-dspy/README.md) |

&#42; A same-box rerun of index mode measured 783ms p50; the 632ms figure is from
the earlier session. Both reproduced 83.0% / 100% exactly.
&#42;&#42; The bf16 open-mode run hit a disk-I/O stall (median 10.2s, confirmed at
~5-6MB/s via `/proc/<pid>/io`), so no honest latency or speedup ratio can be
quoted from it.

Definitions, because the denominators differ and mixing them is the easiest way
to be wrong here:

- **Overall** = correct / all eligible word errors.
- **Conditional** = correct / errors where the gold word was actually among the
  candidates shown. Index mode's 100% conditional is not a perfect system — it
  means the model always picked gold *when gold was on the list*, and scored 0%
  on the 17/100 where it was not.
- **Oracle@k** = what a perfect chooser could reach given the candidate list. It
  is a ceiling, not a model.
- `overall = coverage × conditional`. Exp 2: `64.82% = 80.89% × 80.14%`.
- All LLM latencies are single-request (batch 1) on the same 4-vCPU / 15GB /
  no-GPU box. None of them is a GPU number; no GPU latency exists for any model
  in this repo.

### Small-sample health warning

Every LLM-judge row is n=100-529 on one fixed seed-1337 shuffle of the eligible
set. A 95% CI on a 100-sample proportion near 40-80% is roughly ±8-10 pp. The
q4_0-vs-bf16 gap (2-3 points) is inside that noise in *magnitude* even though its
*direction* is consistent across every comparison made.

## The experiments

| # | Experiment | Status | Headline | Where |
|---|---|---|---|---|
| 1 | 28M byte-level Hunspell reranker | completed, superseded | 62.56% overall (beat Aspell, the original goal) | [01-byte-reranker-28m](experiments/01-byte-reranker-28m/README.md) |
| 2 | 87M byte-level Hunspell reranker | completed, target missed | 64.82% overall on the full 68,429-error benchmark; 75% target not met | [02-byte-reranker-87m](experiments/02-byte-reranker-87m/README.md) |
| 3 | The 75% campaign (E0-E6) | **planned, never run**, superseded | No result. Left behind the evaluation contract and the correction that 75% *is* reachable inside the current pool | [03-scaling-campaign-75](experiments/03-scaling-campaign-75/README.md) |
| 4 | Frozen ModernBERT + selector head | **aborted mid-run** (H2 NaN) | No BEA number. H0/H1 on synthetic D-pair only | [04-frozen-encoder](experiments/04-frozen-encoder/README.md) |
| 5 | LLM judge, index mode | completed on CPU; **GPU pods never ran** | gemma-4-E2B-it 83.0% (n=100) / 78.8% (n=486) | [05-llm-judge-index](experiments/05-llm-judge-index/README.md) |
| 6 | LLM judge on CPU: 4 answer modes, q4_0 + llama.cpp | completed | **90.0% (n=100)** open mode — the best number in the repo | [06-llm-judge-cpu-llamacpp](experiments/06-llm-judge-cpu-llamacpp/README.md) |
| 7 | gemma-4-E2B q4_0 on GPU via llama.cpp + DSPy prompt optimization | **running** | pending | [07-gemma4-gpu-dspy](experiments/07-gemma4-gpu-dspy/README.md) |

Each experiment directory holds its own front-matter block (status, date/commit,
headline, cost, what it settled, what it left open), the write-up, and the plan
that produced it.

New to the metric vocabulary? Read
[03-scaling-campaign-75/DEVELOPER_GUIDE.md](experiments/03-scaling-campaign-75/DEVELOPER_GUIDE.md)
first — coverage vs conditional vs overall, retention vs rescue, and how to read
a comparison statistically.

## What has been ruled out, and why

| Ruled out | Evidence |
|---|---|
| Getting past ~81% overall by better reranking alone | Hunspell's pool contains the gold word for only 80.89% of BEA errors at 16 slots (80.34% at 10). 19.11% of all errors are unreachable by *any* chooser restricted to that list. [exp 2](experiments/02-byte-reranker-87m/README.md) |
| Widening the pool with Aspell | Would raise the ceiling to 87.7% (measured on a 6,000-error BEA sample) — the single biggest lever available — but **out of scope by explicit user decision**. Do not quietly reintroduce it to hit a number. [exp 2 run notes §2](experiments/02-byte-reranker-87m/RUN_NOTES.md) |
| `openbmb/MiniCPM5-1B` as a judge | 39.0% / 37.4% overall, *below* Hunspell top-1 on the same rows (59.0% / 56.8%) — reranking with it actively hurts. Not a parsing bug; predictions are varied and in range. Also ~4x slower than models on either side of it in size. [exp 5](experiments/05-llm-judge-index/README.md) |
| q4_0 as a free speedup | 2-3 points worse than bf16 and one-directional: 0 wins vs 2 losses (index), 0 vs 3 (open), 2 vs 9 (sentence). Buys a trustworthy 2.8x in sentence mode and nothing at all in index mode. [exp 6](experiments/06-llm-judge-cpu-llamacpp/README.md) |
| Reading sentence mode's strict score at face value | 15 of its 27 strict errors are punctuation attachment on correct corrections; 73/100 rewrites reflow BEA's pre-tokenised spacing despite the prompt forbidding it. Honest number is the punctuation-insensitive 88.0%. [exp 6](experiments/06-llm-judge-cpu-llamacpp/README.md) |
| Trusting synthetic validation as a proxy for BEA | Exp 2 reached 92.42% synthetic conditional while scoring 80.14% conditional on BEA — a ~12.3 pp transfer gap. [exp 4 plan §1](experiments/04-frozen-encoder/PLAN.md) |
| cu128 wheels / the py3.12 torch 2.8 image on CUDA 12.4 hosts | Silently fall back to CPU; `hunspell==0.5.5` also cannot build on 3.12. Use the py3.11/cu124 image and `apt-get install python3-hunspell` where 3.12 is unavoidable. [exp 5](experiments/05-llm-judge-index/README.md), [exp 4](experiments/04-frozen-encoder/README.md) |

## Queued next

1. **Experiment 7 (in flight)** — gemma-4-E2B q4_0 on GPU via llama.cpp + DSPy
   prompt optimization. Write its numbers into
   [experiments/07-gemma4-gpu-dspy/README.md](experiments/07-gemma4-gpu-dspy/README.md)
   and add its rows to the comparison table above.
2. **Re-score the best prompt at a real sample size.** Every LLM number here is
   n≤529. Nothing has been measured against the 68,429-error denominator the
   trained model was measured on.
3. **The two unrun prompt fixes** from experiment 6: an explicit
   `<corrected_word>` field alongside the rewrite, or marking the correction
   inside it. Both target the punctuation artifact that costs sentence mode 15
   points of strict accuracy.
4. **Finish or formally drop experiment 4.** The frozen-encoder pilot died on an
   H2 NaN with its central question unanswered and its fixes already written
   down.

## Result artifacts

These directories are written directly by the scripts, at paths hardcoded in
them, so they stay where they are:

| Path | Written by | Contents |
|---|---|---|
| [`bea60k/`](bea60k/) | `scripts/benchmark_bea60k.py`, `scripts/benchmark_aspell.py` | Experiment 2's full-benchmark `results.json`, the Aspell baseline, the gold-index histogram, and four example JSONLs (fixed / damaged / both-failed / gold-outside-top10) |
| [`llm_judge/`](llm_judge/) | `scripts/llm_judge_bea60k.py` | Experiment 5: per-model `results.json`, predictions, latency histograms, `summary.json` |
| [`llm_judge_cpu/`](llm_judge_cpu/) | `scripts/llm_judge_bea60k_cpu.py` | Experiment 6: one directory per model/answer-mode/quantization |
| [`run3_partial/`](run3_partial/) | fetched from pod `3gxtgufkzcvl40` | Data-build statistics from the run that OOMed at step 18 — kept because the build is expensive and these stats are the evidence the data fix worked |
| [`training_loss.png`](training_loss.png) | `scripts/train.py` (global path — **overwritten by every run**; archive before retraining) | Experiment 2's loss curve |
| [`typo_calibration.json`](typo_calibration.json) | `scripts/calibrate_typo_model.py` | Synthetic-vs-authentic typo distribution, calibrated against Wikipedia's public misspelling list, never against BEA |
| `full/`, `sanity/` | `scripts/train.py`, `scripts/run_sanity.sh` | VRAM sample from the full run; sanity-run placeholder |

Experiment 2's fetched text artifacts (`SUMMARY.txt`, `STATUS.txt`,
`data_stats.json`, `manifest.json`) live with the experiment, in
[experiments/02-byte-reranker-87m/](experiments/02-byte-reranker-87m/).
