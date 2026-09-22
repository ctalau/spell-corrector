# Experiment 10 — the quantized 0.8B corrector on one RTX 3090: 420 corrections/second

| | |
|---|---|
| **Status** | **completed** — 2026-09-22 |
| **When** | 2026-09-22, pod `4068yfukcyxtnv`, NVIDIA GeForce RTX 3090 (community, $0.22/hr), commit `2337ef8` |
| **Headline result** | **420 corrections/s** sustained on one 3090, at a p50 of **0.304 s** and p90 0.378 s under 128-way concurrency — **$0.000145 per 1,000 corrections** |
| **Cost** | ~$0.28 of GPU time (three pods: two short failed preparations, one 60-minute working pod) |
| **What it settles** | What a single card actually serves. Also: on this workload **int8 activations are the whole win and int4 weights are worth nothing** — W4A16 ties unquantized fp16 (348 rps), W8A8 beats it by 27% |
| **What it leaves open** | The quality of the W8A8 checkpoint. This experiment measured throughput only and scored no accuracy |
| **Code** | `scripts/bench_spell_throughput.py`, `scripts/distill/quantize_w4a16.py`, `scripts/runpod/{launch,bootstrap}_throughput.*`, `scripts/runpod/throughput_{control,driver}.py` |
| **Artifacts** | [`results/`](results/) — every step's raw JSON, the pod's setup log, the quantizer's module list |

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

- **No accuracy was measured, and none should be inferred.** BEA-60K is locked;
  benchmark prompts here are synthesised from `data/wikipedia_misspellings.txt`,
  the same rule `scripts/benchmark_llama_server.py` follows. The W8A8
  checkpoint's answers were only eyeballed (they are plausible single words:
  `achieve`, `accomplish`, `accurately`), never scored. **Quantization quality is
  the open question this experiment does not answer** — the Q4_K_M student cost
  0.7 points against NF4, and W8A8 has not been given the same treatment.
- **Run-to-run variance is about ±5%.** The winning configuration measured
  441.5 rps over a 30-second window in step 6 and 420.3 over a 60-second window
  in step 10. The longer window is the honest number; both are reported.
- **One GPU, one hour, one card of one type.** Community-cloud hosts differ in
  CPU and PCIe; this is one 3090, not the population of 3090s.
- **A step is 30–60 seconds of steady state**, not a soak test. Nothing here
  speaks to thermal behaviour or stability over hours.

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
