# LLM-judge experiment: prompting small instruct LLMs to rerank Hunspell suggestions

A separate track from the trained ~28M byte-level reranker (`reports/EXPERIMENT.md`):
instead of a purpose-trained model, a general-purpose small instruction-tuned LLM is
prompted zero-shot with Hunspell's numbered suggestion list and asked to pick the
best one. Code: `spelling_reranker/llm_judge.py`, `scripts/llm_judge_bea60k.py`.

## Setup

- **Benchmark**: BEA-60K (locked, never trained/tuned on) via `scripts/download_bea60k.py`.
  68,429 word-level errors; Hunspell flags 67,877 of them (99.19%) and returns at
  least one suggestion for 67,844 -- that eligible set is what both models were
  sampled from.
- **Candidates shown to the model**: up to 8 Hunspell suggestions, in Hunspell's own
  order (`--max-candidates 8`).
- **Prompt**: sentence with the typo marked `<TYPO>...</TYPO>`, a numbered candidate
  list, instruction to reply with only the candidate number. Greedy decoding,
  `max_new_tokens=8`, thinking mode disabled where the chat template supports it.
- **Sampling**: seed 1337, same shuffle of the eligible set for both models, so they
  are scored on directly comparable examples.
  1. **Fixed sample**: 100 examples -- accuracy + latency.
  2. **Timed run**: as many examples as fit in a 5-minute wall-clock budget --
     throughput/latency at scale.
- **Models**: `Qwen/Qwen3.5-0.8B`, `google/gemma-4-E2B-it`, and `openbmb/MiniCPM5-1B`.
- **Environment -- CPU, not GPU.** `RUNPOD_KEY` was not available in this session, so
  this ran on the local box instead: 4 vCPU (Intel Xeon @2.10GHz), 15GB RAM, no GPU,
  `torch==2.14.0+cpu`, `transformers==5.17.0`. Qwen and gemma loaded via the
  `AutoModelForMultimodalLM` class (both are multimodal releases; text-only prompts
  were used here); MiniCPM5-1B is a plain `LlamaForCausalLM` and loaded via
  `AutoModelForCausalLM`. All three in bfloat16 with `low_cpu_mem_usage=True`.
  Checkpoint sizes: Qwen ~1.7GB, MiniCPM5-1B ~2.2GB, gemma-4-E2B-it **~10.2GB** in
  bf16 despite the "E2B" (effective-2B) name -- it fit in 15GB RAM without swapping,
  but left little headroom. Qwen's log also reports that `causal_conv1d` and
  `flash-linear-attention` (its hybrid linear-attention layers' optimized kernels)
  are not installed, so it ran on the slower reference PyTorch path; those kernels
  are CUDA-oriented, so this is expected on CPU and is not a bug in the harness.
  MiniCPM5-1B ships a hybrid Think/No-Think chat template; `enable_thinking=False`
  was passed (supported by its template) to keep it in fast/direct-answer mode,
  consistent with the other two models.
- **Latency measurement**: wall-clock around each `model.generate()` call only
  (excludes the one-time model load and a discarded warmup call). Single request at a
  time (batch size 1), to measure realistic per-query latency rather than
  batched throughput.

## Results

| System | Overall accuracy | Conditional accuracy&#42; | Hunspell top-1 (same sample) | p50 latency | p99 latency |
|---|---|---|---|---|---|
| Hunspell top-1 (full BEA-60K, `reports/bea60k/results.json`) | 53.7% | -- | -- | -- | -- |
| Qwen3.5-0.8B, 100-sample | **60.0%** | 72.3% | 59.0% | 541ms | 823ms |
| gemma-4-E2B-it, 100-sample | **83.0%** | 100.0% | 59.0% | 632ms | 825ms |
| MiniCPM5-1B, 100-sample | **39.0%** | 47.0% | 59.0% | 2183ms | 2772ms |
| Qwen3.5-0.8B, timed 5min (n=529) | **56.5%** | 68.6% | 56.1% | 530ms | 823ms |
| gemma-4-E2B-it, timed 5min (n=486) | **78.8%** | 95.3% | 56.0% | 605ms | 830ms |
| MiniCPM5-1B, timed 5min (n=139) | **37.4%** | 45.6% | 56.8% | 2146ms | 2787ms |

&#42; Conditional accuracy = accuracy among examples where the gold correction was
actually one of the candidates shown (83/100 on the fixed sample for all three
models, since all three were shown the same examples).

Full machine-readable results:
`reports/llm_judge/{qwen3.5-0.8b,gemma-4-e2b,minicpm5-1b}/results.json`,
per-example predictions in `predictions_sample100.jsonl` / `predictions_timed.jsonl`,
and latency histograms (JSON/CSV/PNG) alongside them.

### Latency histograms (100-example fixed sample, ms)

| Bin | Qwen3.5-0.8B | gemma-4-E2B-it | MiniCPM5-1B |
|---|---|---|---|
| 400-600ms | 62 | 32 | -- |
| 600-800ms | 33 | 66 | -- |
| 800-1000ms | 5 | 2 | -- |
| 1500-2000ms | -- | -- | 9 |
| 2000-3000ms | -- | -- | 90 |
| 3000-5000ms | -- | -- | 1 |

Qwen and gemma land in a tight, unimodal 400-1000ms band per call; MiniCPM5-1B is a
full order of magnitude to the right, in an equally tight 1.5-3s band -- there is no
overlap between the two groups at all on this box. Both are consistent with
single-request, short-generation (≤8 tokens) latency rather than throughput-oriented
batched serving. See the `*_latency_histogram.png` files for the full-resolution
charts (fixed-sample and timed-run separately).

## Reading the results

- **gemma-4-E2B-it and Qwen3.5-0.8B beat Hunspell top-1 and the trained 28M
  reranker's Hunspell-top-1 baseline outright**, and gemma-4-E2B-it's *conditional*
  accuracy (100% / 95.3%) is higher than the trained reranker's conditional accuracy
  (80.1%, from `reports/bea60k/results.json`) on a much smaller sample -- worth
  treating as a promising signal rather than a settled comparison, given n=100-529
  vs the trained reranker's full-benchmark n=68,429.
- **MiniCPM5-1B is a clear outlier, in the wrong direction.** Its accuracy (39.0% /
  37.4%) sits *below* Hunspell top-1 alone (59.0% / 56.8% on the same samples) --
  i.e. on this task, in this configuration, letting MiniCPM5-1B rerank actively hurts
  versus just taking Hunspell's rank-0 suggestion. Spot-checking
  `predictions_sample100.jsonl` rules out a parsing bug (it returns varied,
  in-range candidate numbers, not a stuck default); this looks like a genuine
  capability gap for this specific model/prompt/task combination, not a harness
  issue. It is also ~4x slower per call (p50 2183ms vs 530-630ms) despite being
  closer in nominal parameter count to Qwen3.5-0.8B than to gemma-4-E2B-it.
- **gemma-4-E2B-it noticeably outperforms both smaller models on accuracy** (83.0%
  vs 60.0% vs 39.0% overall; 100% vs 72.3% vs 47.0% conditional) while its per-call
  latency is only modestly higher than Qwen's (~90-100ms more at p50) and lower than
  MiniCPM5-1B's. Given gemma-4-E2B-it's much larger checkpoint (~10.2GB vs ~1.7-2.2GB
  bf16), the accuracy edge over Qwen is not a surprising trade, but the latency gap
  is smaller than the parameter-count gap would suggest -- consistent with "E2B"
  effective-compute design intent. MiniCPM5-1B, the intermediate-sized model,
  landing both least accurate *and* slowest is the most interesting result here.
- **Caveat -- this is a CPU run, not the GPU run originally planned.** `RUNPOD_KEY`
  was unavailable in this session; latency numbers here reflect a 4-vCPU box with no
  GPU and (for Qwen) missing optimized kernels for its hybrid attention layers.
  Absolute latency on a GPU pod would very likely be substantially lower for all
  three models, and their relative latency ranking could plausibly differ once
  GPU-optimized kernels are available (Qwen's reference-path fallback in particular
  is a known slowdown that a GPU run would remove; MiniCPM5-1B's slowness has not
  been root-caused and could equally be CPU-specific or could persist on GPU).
  Accuracy numbers should be unaffected by the compute backend (greedy decoding is
  deterministic modulo floating-point non-associativity).
- **Sample sizes are small.** 100 (fixed) and 139-529 (timed) examples are enough to
  see clear accuracy gaps between these three models, but not enough to treat these
  percentages as precise; a 95% CI on a 100-sample binomial proportion near 40-80%
  is roughly ±8-10pp.

## Reproducing

Default (sample of 100, then a 5-minute wrapping timed pass):

```bash
python scripts/download_bea60k.py
python scripts/llm_judge_bea60k.py \
    --model-id Qwen/Qwen3.5-0.8B --model-name qwen3.5-0.8b \
    --bea-dir data/bea60k --output reports/llm_judge/qwen3.5-0.8b
python scripts/llm_judge_bea60k.py \
    --model-id google/gemma-4-E2B-it --model-name gemma-4-e2b \
    --bea-dir data/bea60k --output reports/llm_judge/gemma-4-e2b
python scripts/llm_judge_bea60k.py \
    --model-id openbmb/MiniCPM5-1B --model-name minicpm5-1b \
    --bea-dir data/bea60k --output reports/llm_judge/minicpm5-1b
```

`--time-budget-seconds 3600` is the primary longer-run mode: it scores as many
eligible examples as fit in one hour and writes overall/conditional accuracy,
Hunspell top-1 on the same set, latency p50/p99/mean, throughput, `n_ok`, and
elapsed time to `results.json` (`timed`, with `timed_5min` kept as an alias).
`--skip-sample` drops the 100-example phase so the hour is spent on the timed
pass. `--full` is optional: one non-wrapping pass over the shuffled eligible
set as `full_bea60k` (still honor a time budget; this is not an unbounded
full-BEA requirement). Progress is logged every 500 examples; predictions are
checkpointed every 2000.

### Full Gemma on Runpod (1 hour, Gemma-only)

GPU pods use `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` with `transformers>=5.5,<5.15` (Gemma-4 needs 5.5; 5.15+ disables this image's torch 2.4.1).

Cheapest planned GPU: Community RTX A4000 (~$0.17/hr). Bootstrap skips the
100-sample + 5-min combo when `TIME_BUDGET_SECONDS` is not `300`, and skips
model B when `MODEL_B_ID` is `none` / empty. Single timed phase at 3600s:

```bash
python scripts/runpod/launch.py \
  --name llm-judge-gemma-1h \
  --bootstrap-path scripts/runpod/bootstrap_llm_judge.sh \
  --branch cursor/llm-judge-1h-gemma-ff25 \
  --gpu "NVIDIA RTX A4000" \
  --max-price 0.20 \
  --disk-gb 40 \
  --env MODEL_A_ID=google/gemma-4-E2B-it \
  --env MODEL_A_NAME=gemma-4-e2b \
  --env MODEL_B_ID=none \
  --env TIME_BUDGET_SECONDS=3600
```

Optional: add `--env FULL_BEA=1` for a non-wrapping pass over the eligible set
(still capped at 1h). Add `--env ALSO_SAMPLE=1` if you want sample-100 first,
then the 1h timed/full phase. After the run, fetch artifacts and terminate:

```bash
python scripts/runpod/fetch_artifacts.py <pod-id> --dest .
python scripts/runpod/terminate.py --all
```

Monitor `https://<pod-id>-8000.proxy.runpod.net/{run.log,STATUS,DONE}`.
On a GPU pod, `scripts/runpod/bootstrap_llm_judge.sh` otherwise still defaults
to Qwen then Gemma with the original 100-sample + 300s timed pair. The CPU
numbers in this report used the CLI directly on the local box.
