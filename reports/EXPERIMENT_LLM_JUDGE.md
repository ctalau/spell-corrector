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

## Follow-up: how much does the forced-choice format itself cost?

gemma-4-E2B-it's index-mode result above (83.0% overall / 100% conditional) is
capped by construction: 17 of the 100 sampled errors never had the gold correction
anywhere in Hunspell's candidate list, so no forced choice among those candidates
could ever get them right. Two follow-up modes test the same fixed 100-sample
subset with that constraint loosened or removed entirely:

- **`--answer-mode open`** (`build_open_messages`/`parse_open_word`): same Hunspell
  candidates shown as a hint, but the model may write any word instead of being
  restricted to picking one.
- **`--answer-mode beam`** (`build_generative_messages`/`beam_word_candidates`/
  `select_by_edit_distance_and_probability`): no Hunspell candidates shown at all.
  The model's own beam search (width 3) generates up to 3 candidate words; each is
  cut at its first word boundary and scored `logprob - weight * edit_distance(word,
  typo)` (weight 1.0), and the top-scoring one is the answer. This replaces Hunspell
  as the candidate generator entirely, using only the LLM plus a classic
  noisy-channel-style edit-distance prior.

| | Index (pick from list) | Open (list as hint, free answer) | Beam (no list, self-generated + edit-distance rerank) |
|---|---|---|---|
| Overall accuracy | 83.0% | **90.0%** | 87.0% |
| Conditional accuracy (gold was offered, 83/100) | 100% | 96.4% (80/83) | 91.6% (76/83) |
| Accuracy when gold was *not* offered (17/100) | 0% (impossible by construction) | 58.8% (10/17) | **64.7% (11/17)** |

Both follow-ups beat index mode overall by removing its hard ceiling. Open mode
wins on total accuracy (90.0%), but beam mode -- despite getting *no* Hunspell hint
at all -- recovers the most of the previously-unreachable cases (64.7% vs 58.8%),
at the cost of being weaker on the "easy" gold-in-pool subset (91.6% vs 96.4%,
unsurprising since it never sees Hunspell's list to fall back on). A characteristic
open-mode recovery: typo "thursty" in "I had the worst thursty I have ever had" --
Hunspell's only candidates were "thirsty" and "hurst" (both wrong; the context
calls for the noun "thirst", not the adjective "thirsty"), and gemma-4-E2B-it
produced "thirst" directly from context despite it never appearing in the
candidate list.

**A genuine flaw, not glossed over:** 2 of beam mode's 13 wrong answers are the
model echoing the typo completely unchanged ("ugry" -> "ugry" instead of "ugly";
"Miken" -> "Miken" instead of "McCain"). The scoring formula is structurally
responsible: `edit_distance(word, typo)` is 0 when a beam candidate equals the
typo itself, so the formula rewards *not correcting at all* whenever the
logprob gap to a real correction is small (for "ugry": logprob -0.73 for the
echoed typo vs -0.70 for "ugly" -- nearly tied on probability, but the +1 edit
distance was enough to flip it). A straightforward fix is to exclude or heavily
penalize candidates equal to the typo before reranking; not applied here so the
result reported is the honest, unpatched one. Full predictions with all 3 beam
candidates and their scores per example: `reports/llm_judge/{gemma-4-e2b-open,
gemma-4-e2b-beam}/predictions_sample100.jsonl`.

**Latency is not comparable across these three rows and is deliberately left out
of the table.** The open-mode run happened to hit a period of unusually slow disk
I/O on this box (`low_cpu_mem_usage=True` loads the already-bf16 checkpoint via
mmap, so the *first* touch of each weight page during a forward pass faults it in
from disk rather than RAM; confirmed live via `/proc/<pid>/io` at only ~5-6MB/s
sustained), driving its median latency to 10.2s -- an I/O artifact, not a property
of open-mode generation. The beam-mode run's page cache was warm (0 disk reads
observed via the same `/proc/<pid>/io` check), so its 14.8s median is a real, if
expensive, measurement of beam-width-3 generation cost on this CPU box -- roughly
23x the index-mode run's 632ms median, more than the naive 3x-the-beams estimate
would suggest, likely from Gemma4's hybrid sliding/full-attention layers and
shared-KV bookkeeping not being especially optimized for batched beam search on
CPU. Neither number should be read as "mode X is N times slower than mode Y" in
general; both are one-off measurements on a noisy shared box.

## Reproducing

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

# Open-answer ablation (same 100-sample subset, candidates shown only as a hint):
python scripts/llm_judge_bea60k.py \
    --model-id google/gemma-4-E2B-it --model-name gemma-4-e2b-open \
    --bea-dir data/bea60k --output reports/llm_judge/gemma-4-e2b-open \
    --answer-mode open --skip-timed

# Beam-search ablation (same 100-sample subset, no Hunspell candidates at all):
python scripts/llm_judge_bea60k.py \
    --model-id google/gemma-4-E2B-it --model-name gemma-4-e2b-beam \
    --bea-dir data/bea60k --output reports/llm_judge/gemma-4-e2b-beam \
    --answer-mode beam --beam-width 3 --edit-distance-weight 1.0 --skip-timed
```

On a GPU pod, `scripts/runpod/bootstrap_llm_judge.sh` runs both automatically (see
`scripts/runpod/launch.py --bootstrap-path scripts/runpod/bootstrap_llm_judge.sh
--env MODEL_A_ID=... --env MODEL_B_ID=...`); this run instead used the CLI directly
on the local CPU box since `RUNPOD_KEY` was not available.
