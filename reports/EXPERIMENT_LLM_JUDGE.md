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
- **Models**: `Qwen/Qwen3.5-0.8B` and `google/gemma-4-E2B-it`.
- **Environment -- CPU, not GPU.** `RUNPOD_KEY` was not available in this session, so
  this ran on the local box instead: 4 vCPU (Intel Xeon @2.10GHz), 15GB RAM, no GPU,
  `torch==2.14.0+cpu`, `transformers==5.17.0`. Both models loaded via the
  `AutoModelForMultimodalLM` class (both are multimodal releases; text-only prompts
  were used here) in bfloat16 with `low_cpu_mem_usage=True`. Qwen's checkpoint is
  ~1.7GB; gemma-4-E2B-it's is **~10.2GB** in bf16 despite the "E2B" (effective-2B)
  name -- it fit in 15GB RAM without swapping, but left little headroom. Qwen's log
  also reports that `causal_conv1d` and `flash-linear-attention` (its hybrid
  linear-attention layers' optimized kernels) are not installed, so it ran on the
  slower reference PyTorch path; those kernels are CUDA-oriented, so this is
  expected on CPU and is not a bug in the harness.
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
| Qwen3.5-0.8B, timed 5min (n=529) | **56.5%** | 68.6% | 56.1% | 530ms | 823ms |
| gemma-4-E2B-it, timed 5min (n=486) | **78.8%** | 95.3% | 56.0% | 605ms | 830ms |

&#42; Conditional accuracy = accuracy among examples where the gold correction was
actually one of the candidates shown (83/100 on the fixed sample for both models,
since they were shown the same examples).

Full machine-readable results: `reports/llm_judge/{qwen3.5-0.8b,gemma-4-e2b}/results.json`,
per-example predictions in `predictions_sample100.jsonl` / `predictions_timed.jsonl`,
and latency histograms (JSON/CSV/PNG) alongside them.

### Latency histograms (100-example fixed sample, ms)

| Bin | Qwen3.5-0.8B | gemma-4-E2B-it |
|---|---|---|
| 400-600ms | 62 | 32 |
| 600-800ms | 33 | 66 |
| 800-1000ms | 5 | 2 |

Both models land in a tight, unimodal 400-1000ms band per call on this CPU box --
consistent with single-request, short-generation (≤8 tokens) latency rather than
throughput-oriented batched serving. See the `*_latency_histogram.png` files for the
full-resolution charts (fixed-sample and timed-run separately).

## Reading the results

- **Both LLMs beat Hunspell top-1 and the trained 28M reranker's Hunspell-top-1
  baseline outright**, and gemma-4-E2B-it's *conditional* accuracy (100% / 95.3%) is
  higher than the trained reranker's conditional accuracy (80.1%, from
  `reports/bea60k/results.json`) on a much smaller sample -- worth treating as a
  promising signal rather than a settled comparison, given n=100-529 vs the trained
  reranker's full-benchmark n=68,429.
- **gemma-4-E2B-it noticeably outperforms Qwen3.5-0.8B on accuracy** (83.0% vs 60.0%
  overall, 100% vs 72.3% conditional) at only a modest latency cost (~90-100ms
  more at p50). Given gemma-4-E2B-it's much larger checkpoint (~10.2GB vs ~1.7GB
  bf16), this is not a surprising trade, but the latency gap is smaller than the
  parameter-count gap would suggest -- consistent with "E2B" effective-compute
  design intent.
- **Per-call latency is close between the two models** despite the size difference,
  and both are dominated by fixed overhead (prompt processing, short generation)
  rather than raw FLOPs at this scale and prompt length.
- **Caveat -- this is a CPU run, not the GPU run originally planned.** `RUNPOD_KEY`
  was unavailable in this session; latency numbers here reflect a 4-vCPU box with no
  GPU and (for Qwen) missing optimized kernels for its hybrid attention layers.
  Absolute latency on a GPU pod would very likely be substantially lower for both
  models, and the two models' relative latency ranking could plausibly differ once
  GPU-optimized kernels are available (Qwen's reference-path fallback in particular
  is a known slowdown that a GPU run would remove). Accuracy numbers should be
  unaffected by the compute backend (greedy decoding is deterministic modulo
  floating-point non-associativity).
- **Sample sizes are small.** 100 (fixed) and 486-529 (timed) examples are enough to
  see a clear accuracy gap between the two models here, but not enough to treat
  these percentages as precise; a 95% CI on a 100-sample binomial proportion near
  70-80% is roughly ±8pp.

## Reproducing

```bash
python scripts/download_bea60k.py
python scripts/llm_judge_bea60k.py \
    --model-id Qwen/Qwen3.5-0.8B --model-name qwen3.5-0.8b \
    --bea-dir data/bea60k --output reports/llm_judge/qwen3.5-0.8b
python scripts/llm_judge_bea60k.py \
    --model-id google/gemma-4-E2B-it --model-name gemma-4-e2b \
    --bea-dir data/bea60k --output reports/llm_judge/gemma-4-e2b
```

On a GPU pod, `scripts/runpod/bootstrap_llm_judge.sh` runs both automatically (see
`scripts/runpod/launch.py --bootstrap-path scripts/runpod/bootstrap_llm_judge.sh
--env MODEL_A_ID=... --env MODEL_B_ID=...`); this run instead used the CLI directly
on the local CPU box since `RUNPOD_KEY` was not available.
