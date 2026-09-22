# Experiment 10 — the quantized 0.8B corrector on one RTX 3090: 420 corrections/second

| | |
|---|---|
| **Status** | **completed** — 2026-09-22 |
| **When** | 2026-09-22, pod `4068yfukcyxtnv`, NVIDIA GeForce RTX 3090 (community, $0.22/hr), commit `2337ef8` |
| **Headline result** | **420 corrections/s** sustained on one 3090, at a p50 of **0.304 s** and p90 0.378 s under 128-way concurrency — **$0.000145 per 1,000 corrections**, at **87.40%** on the held-out BEA test split (n=2,000), 0.15 pp below the unquantized model |
| **Cost** | ~$0.36 of GPU time (the climb on one 60-minute pod, two short failed preparations before it, and a ~20-minute pod for the accuracy follow-up) |
| **What it settles** | What a single card actually serves. Also: on this workload **int8 activations are the whole win and int4 weights are worth nothing** — W4A16 ties unquantized fp16 (348 rps), W8A8 beats it by 27% |
| **What it leaves open** | Nothing on quantization choice. The follow-up below scored all three checkpoints on the held-out BEA splits: **W8A8 costs 0.15 pp against fp16 (p=0.63, indistinguishable); W4A16 costs 1.15 pp (p=0.003, real)** — so int4 is dominated on both axes |
| **Code** | `scripts/bench_spell_throughput.py`, `scripts/distill/quantize_w4a16.py`, `scripts/distill/eval_openai_chat.py`, `scripts/runpod/{launch,bootstrap}_throughput.*`, `scripts/runpod/throughput_{control,driver}.py` |
| **Artifacts** | [`results/`](results/) — every step's raw throughput JSON, the six accuracy metric files, the split metadata and the quantizer's module list |

---

## The question

Every previous number for the 0.8B student is a **latency** number taken one
request at a time: 0.597 s p50 on 4 vCPUs, 0.318 s for the 2B teacher on a 3090.
None of them says what one card serves **per second** when requests arrive
together — which is the only number a deployment is costed on.

## What was served

`ctalau/qwen35-08b-spell-m7-distill` (the M7 student merged into fp16),
requantized **for the GPU** rather than reusing the deployed `Q4_K_M` GGUF: that
file exists for llama.cpp on a CPU, and vLLM's GGUF path is a compatibility
shim, not a fast one. vLLM 0.29.0, compressed-tensors, the workload unchanged
from production — one `<TYPO>…</TYPO>` sentence in (88 prompt tokens), greedy,
`max_tokens=5`, 2.84 output tokens on average.

The shape of that workload decides everything below: **31 prompt tokens for
every output token**. This is a prefill-bound, compute-bound service, and the
optimizations that matter for chat serving — KV cache tricks, speculative
decoding, big batches — mostly do not apply.

## The climb, step by step

Ten steps, each a measurement under closed-loop load on the same pod, each
changing one lever and kept or reverted on the number. Steps 1–9 measured over
30-second windows after a 10-second warmup; step 10 re-confirmed the winner over
60-second windows.

| # | Lever | Best | Δ | Verdict |
|---|---|---:|---:|---|
| 1 | W4A16 GPTQ, vLLM defaults — baseline | 329.7 rps | — | knee already at c=128 |
| 2 | `--max-num-batched-tokens 32768`, `--max-num-seqs 512` | 344.3 | +4.4% | keep (small) |
| 3 | `--api-server-count 4` | 348.4 | +1.2% | keep (small) — **the front end was not the ceiling** |
| 4 | **W8A8 int8** instead of W4A16 | 403.5 | +15.8% | **keep** |
| 5 | **also quantize the 90 linear-attention projections** | 424.3 | +5.2% | **keep** |
| 6 | `--async-scheduling` | 441.5 | +4.1% | **keep** |
| 7 | prefix caching off | 427.4 | −3.2% | **revert** |
| 8 | fp16 reference (not a candidate) | 348.5 | — | quantization is worth +27% |
| 9 | chat template rendered client-side (`/v1/completions`) | 429.3 | −2.8% | **revert** |
| 10 | the winner, re-confirmed over 60 s windows | **420.3** | — | headline |

Zero errors and zero empty answers in all 10 steps, at every concurrency.

### Step 10, the final sweep

| Concurrency | Throughput | p50 | p90 | GPU memory |
|---:|---:|---:|---:|---:|
| 32 | 170.0 rps | 0.191 s | 0.238 s | 22.9 GB |
| 64 | 287.1 rps | 0.225 s | 0.277 s | 22.9 GB |
| **128** | **420.3 rps** | **0.304 s** | **0.378 s** | 22.9 GB |
| 192 | 406.9 rps | 0.469 s | 0.601 s | 22.9 GB |
| 256 | 418.2 rps | 0.615 s | 0.763 s | 22.9 GB |
| 384 | 417.4 rps | 0.921 s | 1.145 s | 22.9 GB |

The card saturates at **c=128**. Past it throughput is flat and latency is
pure queueing: c=384 costs 3× the p50 for nothing. The service should cap
in-flight requests at ~128, not because of memory but because a deeper queue
buys no work.

## What the climb actually found

**1. Int4 weights bought nothing; int8 arithmetic bought 27%.** W4A16 (348 rps
at the same flags) and unquantized fp16 (348.5 rps) are the same number. That is
not a surprise once the regime is named: W4A16 is *weight-only* — Marlin
unpacks int4 back to fp16 and runs the GEMM on fp16 tensor cores, so it buys
memory, and memory was never scarce here (the weights are 1.7 GB of a 24 GB
card). W8A8 quantizes the **activations** too, which moves the GEMMs onto the
3090's int8 tensor cores — a different, roughly 2× ceiling. vLLM confirms the
kernel in its log: `Selected CutlassInt8ScaledMMLinearKernel`.

The deployed artifact is Q4_K_M, and the reflex "use the quantized model for
speed" is right on a CPU and wrong here. **On a GPU, pick the quantization that
changes which tensor cores run, not the one that makes the file smaller.**

**2. Quantize the linear-attention projections, not just the MLPs.** The model
is a hybrid — 18 linear-attention layers to 6 full-attention ones. The
conservative first pass left all 90 linear-attention projections in fp16, about
a third of the non-embedding weights, running on fp16 tensor cores while the
MLPs ran on int8. Quantizing `in_proj_qkv`, `in_proj_z` and `out_proj` too (and
leaving only the 16-wide `in_proj_a`/`in_proj_b` gates alone) was worth another
5%.

**3. Prefix caching cannot work here, and turning it off still costs 3%.** Every
request shares a 55-token instruction prefix, which looks like a free 60% cut of
prefill — and the hit rate was **0.0% in every single step**. The hybrid cache is
why: vLLM sets the attention block size to **544 tokens** so the attention page
is not smaller than the mamba page, and an 88-token prompt never fills one
block. Disabling it then measured *worse*, because the flag also changes the
mamba cache mode away from `align`. So: left on, hitting nothing.

**4. Neither the front end nor the scheduler was the bottleneck.** Four API
server processes bought 1.2%; a 4× larger scheduler token budget bought 4.4%;
moving Jinja rendering to the client *lost* 2.8%. All three say the same thing —
the GPU is busy. The model has ~497M non-embedding parameters, so a token
costs about 1.0 GFLOP of weight arithmetic. The fp16 and W4A16 configurations
moved ~31.5k tokens/s, i.e. **~31 TFLOPS — about 88% of the 3090's ~35.6 TFLOPS
fp16 peak**; they were already against the roofline, which is why no amount of
scheduling tuning moved them. The winning int8 configuration does ~38k tokens/s,
which is *past* that fp16 ceiling precisely because the GEMMs are no longer
running in fp16.

## What one card is worth

At 420.3 corrections/s on a $0.22/hr community 3090:

| | |
|---|---|
| Corrections per hour | 1.51 million |
| **Cost per 1,000 corrections** | **$0.000145** |
| Cost per million corrections | $0.15 |
| Equivalent 4-vCPU CPU boxes (at 0.597 s p50, sequential) | ~251 |

For scale: the whole nine-experiment project cost about $10 of GPU time, which
is about 69 million corrections at this rate.

## Health warnings

- **The throughput numbers above contain no accuracy claim.** Their load is
  synthesised from `data/wikipedia_misspellings.txt`, never BEA-60K — a
  hill-climbing loop is exactly the repeated-measurement surface a locked
  benchmark must stay out of. Accuracy was measured separately and once, below.
- **Run-to-run variance is about ±5%.** The winning configuration measured
  441.5 rps over a 30-second window in step 6 and 420.3 over a 60-second window
  in step 10. The longer window is the honest number; both are reported.
- **One GPU, one hour, one card of one type.** Community-cloud hosts differ in
  CPU and PCIe; this is one 3090, not the population of 3090s.
- **A step is 30–60 seconds of steady state**, not a soak test. Nothing here
  speaks to thermal behaviour or stability over hours.

## Follow-up: what the quantization costs in accuracy

Measured after the climb, on a second pod, as a **single scoring pass** — no
tuning, no repeats, no selection among variants. Same decode contract as every
previous corrector number in this repo: greedy, temperature 0, `max_tokens` 5,
`direct_correct_v1.txt`, the same `normalize_prediction`.

### Which rows are eligible

Only 2,100 of BEA-60K's 68,395 word errors can honestly be used here. The M7
student was **trained on 64,295 of them** — `scripts/distill/build_data.py`
splits the benchmark by source sentence into `train`/`val`/`dev`/`test`/
`frozen_100`, and running the full benchmark would mostly be scoring the
training set and calling it accuracy. The splits were rebuilt on the pod from
the same seed with the same frozen-100 reconstruction check, so `test.jsonl` is
the split the deployed Q4_K_M's 86.60% was measured on, and the control plane
refuses to score `train` or `val` at all.

### Results

| Checkpoint | Engine | test, n=2,000 exact | test casefold | frozen 100 |
|---|---|---:|---:|---:|
| **fp16 (unquantized)** | vLLM, this pod | 86.65% | **87.55%** | 89.0% |
| **W8A8 int8** (the throughput winner) | vLLM, this pod | 86.60% | **87.40%** | 87.0% |
| **W4A16 int4** | vLLM, this pod | 85.55% | **86.40%** | 83.0% |
| NF4 4-bit (experiment 8) | transformers GPU | — | 87.30% | — |
| Q4_K_M GGUF (deployed) | llama.cpp, 4 vCPU | 85.60% | 86.60% | 88.0% |

95% Wilson intervals on the n=2,000 rows are about ±1.5 pp and overlap heavily,
so the *paired* comparison is the one that carries information — the three
checkpoints answered the same 2,000 rows, so the disagreements can be counted
directly:

| vs fp16, on the same 2,000 rows | Agreement | fp16 right → quantized wrong | fp16 wrong → quantized right | Net | McNemar |
|---|---:|---:|---:|---:|---:|
| **W8A8** | 98.30% | 10 | 7 | **−3** | p = 0.63 |
| **W4A16** | 94.95% | 40 | 17 | **−23** | p = 0.0032 |

**W8A8 is statistically indistinguishable from the unquantized model.** Three
rows out of 2,000, seven of which it got right where fp16 got them wrong: that
is noise, not a tax. **W4A16's 1.15 pp is a real regression** (p = 0.003), and
that settles the trade-off the throughput climb opened:

> int4 weight-only quantization on this model is **slower-or-equal *and* less
> accurate** than int8. It is dominated on both axes, and the only thing it buys
> is a smaller file on a card that had 21 GB spare.

Where W8A8 does differ from fp16, it differs the way a slightly blunter model
would — `worthful` → `worthy` instead of `worthwhile`, `Desingres` → `Designs`
instead of `Designers`. Nothing pathological, no empty answers and no request
failures in any of the six scoring passes.

### Two asides worth keeping

- **Scoring 2,000 rows took 7.75 seconds**, against 1,209 seconds for the same
  split on the CPU Q4_K_M path — a 156× shorter evaluation loop. Evaluation
  being this cheap is itself a result: it makes a held-out re-score a routine
  step rather than an experiment.
- The `frozen_100` column moves by 6 points across these rows on 100 examples.
  It should not be read as a ranking; it is here because previous milestones
  reported it.

## Reproducing

```bash
RUNPOD_KEY=... python scripts/runpod/launch_throughput.py \
    --branch <branch> --token-out /somewhere/outside/the/repo/pod.json
# wait for STATUS to read "ready", then per step:
python scripts/runpod/throughput_driver.py job --token-file pod.json --job step.json
python scripts/runpod/throughput_driver.py wait --token-file pod.json
python scripts/runpod/throughput_driver.py results --token-file pod.json
python scripts/runpod/terminate.py <pod-id>     # never --all on a shared account
```

The winning server:

```
vllm serve <w8a8-checkpoint> --max-model-len 1024 --max-num-batched-tokens 32768 \
    --max-num-seqs 512 --api-server-count 4 --async-scheduling
```

and the checkpoint it serves:

```bash
python scripts/distill/quantize_w4a16.py --model <fp16> --output <w8a8> \
    --scheme W8A8 --algorithm gptq --samples 256 \
    --ignore lm_head --ignore 're:.*mtp.*' --ignore 're:.*visual.*' \
    --ignore 're:.*vision.*' --ignore 're:.*in_proj_a' --ignore 're:.*in_proj_b'
```

and the accuracy pass, against that server, on a rebuilt held-out split:

```bash
python scripts/distill/eval_openai_chat.py --split data/distill/test.jsonl \
    --base-url http://127.0.0.1:8080 --concurrency 64 \
    --out-metrics metrics.json --out-predictions predictions.jsonl
```

## Three things that cost a pod each, recorded so they do not again

1. `llmcompressor` installs cleanly and then fails at import: it pins
   `compressed-tensors` to an exact patch (0.13.0 wants 0.18.0) that the vLLM
   image does not ship. `--no-deps` only moves the failure from the resolver to
   an ImportError *after* the model has loaded. The fix is a
   `--system-site-packages` venv: torch and transformers are inherited,
   compressed-tensors and llmcompressor are the venv's own, the quantizer runs
   under that interpreter and the engine keeps its own. The W4A16/W8A8
   checkpoint is the handoff, and its format is stable across those patches.
2. `--no-deps` also means llmcompressor's runtime imports are yours to install.
   The first launch died on `ModuleNotFoundError: datasets`.
3. A hill-climb cannot be a pod entrypoint, because step N+1 is chosen from step
   N's number. The pod therefore prepares and *waits*, driven over a
   token-authenticated control port. A job names an engine, a prepared model and
   serving flags — never a command line.
