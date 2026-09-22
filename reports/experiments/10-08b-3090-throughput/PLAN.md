# Experiment 10 — plan: how much throughput does the quantized 0.8B have on one 3090?

## The question

Every number this project has published for the 0.8B corrector is a *latency*
number measured one request at a time: 0.597s p50 on 4 vCPUs, 0.318s for the 2B
teacher on a 3090. None of them says what a single card can serve **per second**
when requests arrive together, which is the number a deployment is costed on.

This experiment answers exactly that: **corrections per second on one community
RTX 3090, serving the quantized 0.8B student**, and what it costs per thousand.

## What is being served

`ctalau/qwen35-08b-spell-m7-distill` (the M7 student merged into fp16), quantized
for the GPU rather than reusing the deployed `Q4_K_M` GGUF — that file exists for
llama.cpp on a CPU, and vLLM's GGUF path is a compatibility shim, not a fast one.
The GPU-native equivalent is compressed-tensors **W4A16 group-128**, which vLLM
dispatches to Marlin kernels on `sm_86`.

The workload is the deployed one and does not change during the climb: one
`<TYPO>…</TYPO>` sentence in, `max_tokens=5`, greedy, `direct_correct_v1.txt`.

The model is a hybrid — 18 linear-attention layers to 6 full-attention ones, a
248k vocabulary on a 1024-wide trunk, and a multi-token-prediction head. Those
three facts are what make the answer non-obvious: the KV cache is small and
cheap, the logits GEMM is disproportionately large, and MTP makes speculative
decoding available for a 5-token answer.

## Method: ten measured steps

One pod, kept for the whole climb, driven over a token-authenticated control
port (`scripts/runpod/throughput_control.py`). Each step changes **one** lever,
measures for a fixed window under closed-loop load, and either keeps the change
or reverts it. A step is a measurement, not a guess: throughput is counted over
the measurement window's wall clock, so queueing and HTTP overhead are inside
every number.

Levers, roughly in the order they are expected to matter:

1. concurrency itself — where the knee is, and where it OOMs
2. prefix caching (the 60-token instruction prefix is shared by every request)
3. `--max-num-seqs` / `--max-num-batched-tokens` (batch shape)
4. CUDA graphs and compilation
5. quantization scheme: W4A16 vs in-flight fp8-Marlin vs fp16 reference
6. request path: chat template rendered server-side vs client-side
7. speculative decoding off the MTP head
8. whatever the first six turn up

## Benchmark integrity

BEA-60K is **locked** and is not touched here. Benchmark prompts are synthesised
from `data/wikipedia_misspellings.txt` (vendored, CC BY-SA), the same rule
`scripts/benchmark_llama_server.py` already follows. No accuracy number is
produced or tuned; this experiment measures serving performance only. The
quantized checkpoint's *quality* is a separate question, flagged as open rather
than answered here.

## Cost discipline

Community 3090, ~$0.22/hr. Ten 10-minute steps plus preparation is under three
hours. `scripts/runpod/terminate.py <pod-id>` on every exit path — and never
`--all` on this account, which has other pods running.
