# Experiment 8 / Milestone 7 — distilling the 2B Q4 corrector into the 0.8B Q4 student

| | |
|---|---|
| Status | **completed** |
| Date / commit | 2026-09-19, branch `claude/distill-2b-to-07b-bea-fsqdjt` |
| Headline | The 0.8B student reaches **87.30%** Acc@1 (n=2,000) against a teacher measured at **88.75%** on the same rows — it recovers all but 1.45 points of a model 2.5x its size, and beats the previous 0.8B correctors (M4 84%, M5 82%) |
| Target | **Missed.** The goal was >90%; the teacher's own ceiling is 88.75% |
| Cost | **$0.70** total, including two failed runs and two broken-CUDA hosts |
| Artifacts | [`artifacts/spell_slm_m7/`](../../../artifacts/spell_slm_m7/), [`artifacts/spell_slm_m7_q4/`](../../../artifacts/spell_slm_m7_q4/) |

The method, the split table and the benchmark-policy exception are in
[PLAN.md](PLAN.md).

## Results

Every row is greedy decoding on held-out, sentence-disjoint BEA-60K errors.
Student and teacher were scored by the **same harness on the same rows**, which
is the only comparison here that is apples-to-apples.

| System | Backend | n | Acc@1 exact | Acc@1 casefold | p50 |
|---|---|---:|---:|---:|---:|
| **Student 0.8B** distilled | NF4, GPU | 2,000 | 86.45% | **87.30%** | 0.310 s |
| **Student 0.8B** distilled | Q4_K_M, CPU | 2,000 | 85.60% | **86.60%** | 0.597 s |
| Teacher 2B (M6) | NF4, GPU | 2,000 | 87.95% | **88.75%** | 0.318 s |
| **Student 0.8B** distilled | NF4, GPU | 100 | 89.0% | 89.0% | — |
| **Student 0.8B** distilled | Q4_K_M, CPU | 100 | 88.0% | 88.0% | 0.602 s |
| Teacher 2B (M6) | NF4, GPU | 100 | 86.0% | 87.0% | 0.312 s |
| *M6's own reported figure* | *NF4, GPU* | *100* | *91%* | *91%* | *0.095 s* |

Read the n=2,000 rows. A 100-row proportion near 88% carries roughly ±6 pp;
2,000 rows carry ±1.5 pp. Both student and teacher move by 1-2 points between
the two sample sizes, in opposite directions, which is exactly what that noise
looks like.

GPU latencies are from an RTX 3090 at batch 1; the CPU figures are this
repository's 4-vCPU box at `-t 4`, where M4-M6 used `-t 8`. Latency is
therefore comparable within this table, not against M4-M6.

## What the experiment settles

**Distillation works, and is worth the 13 extra minutes of teacher forward.**
The student gains +3.3 points over M4 (84%) and +5.3 over M5 (82%), the two
previous 0.8B direct correctors, at the same parameter count. Part of that is
the larger training set (64,277 BEA rows against M4/M5's 20,000), part is the
teacher's distribution; this experiment does not separate the two, and an
ablation against a CE-only run on the same 64k rows is the obvious next
measurement.

**The 90% target was never reachable from this teacher.** A student cannot
systematically exceed the signal it is trained on, and the teacher measures
88.75% at n=2,000. The 91% in the M6 write-up is a 100-row number.

**The student beats its teacher on the frozen 100 (89.0% vs 87.0%) and loses to
it on 2,000 rows (87.30% vs 88.75%).** The first is not a paradox — the student
saw 2.7x more training data and a gold-CE term alongside KD — but the second is
the honest summary of the pair.

## What it leaves open

**Why M6 reported 91% and the same adapter scores 87.0% here on the identical
100 rows.** Greedy decoding is deterministic, so this is not sampling noise.
The prediction files differ in the raw generation of **93 of 100 rows** while
the verdict flips on only 4, i.e. the two harnesses decode differently
throughout. I suspected left-padded batched generation — Qwen3.5 runs linear
attention with recurrent state in 18 of its 24 layers, and such layers do not
always respect an attention mask — **and the Q4 run refuted it**: sequential
Q4_K_M scored *below* batched NF4, by about the quantization tax, when the
hypothesis predicts it should have scored well above. The remaining candidate
is the stack itself (transformers 5.18.dev and bitsandbytes 0.50+ here, against
whatever M6 ran), which would mean M6's number is not reproducible on a current
stack. Worth one hour of somebody's time before any 91% is quoted again.

## Getting past 90%

Not by training this student longer: the dev curve was flat from step 4,750 to
6,024, and the ceiling is the teacher's. The options are, in order of expected
value:

1. **A better teacher.** `gemma-4-E2B-it` in open mode scored 90.0% (n=100,
   [exp 6](../06-llm-judge-cpu-llamacpp/README.md)) without any training. Score
   it at n=2,000 first — on this evidence, 100-row numbers in this repository
   run 1-2 points optimistic — and if it holds up, distil from it instead.
2. **Ablate the KD term** (CE-only on the same 64k rows) to find out what the
   teacher's distribution is actually buying. If it buys little, the cheaper
   recipe wins and the teacher choice matters more than the method.
3. **Widen the candidate pool.** Out of scope by standing decision, but the
   measured ceiling lift (87.7% oracle with Aspell added) is the largest
   single lever anybody has quantified in this repository.

## Run notes

Three pods died before the run that finished, which is the useful part of this
section:

| Failure | Cause | Fix |
|---|---|---|
| Pod 1 (L40S), dead at the pip stage | Host's `nvidia_uvm` never loaded: `nvidia-smi` healthy, every `cudaInit` returns "CUDA unknown error" | Probe CUDA with the image's own torch **before** installing anything; the bootstrap now fails in 90 s (~$0.02) instead of 10 minutes, and a launch loop cycles GPU types until a host passes |
| Pod 2 (3090), OOM at step ~45 | The LM head ran over every position: `[16, seq, 248320]` logits, 2 GiB a forward, plus a same-shaped int64 gather index. Peak 22.3 GiB of 24 | Run the transformer stack alone, select the ≤10 answer positions out of the hidden states, apply the LM head to those only. Same loss to 3 decimals, 25x less memory, 46% faster |
| Pod 3 (3090), OOM at step ~1,550 after reaching 85.4% | Fixed-row batches made memory track the longest sentence; and the OOM retry kept the failed frame's activations alive, so splitting *added* memory and recursed to a batch of one with 22.8 GiB live | Cap `rows x longest-row` at 2,048 tokens per micro-batch, and clear the frame's tensors (not just the traceback) before retrying. Peak VRAM fell to 12.2 GiB with zero splits across 6,024 steps |
| GGUF conversion | Stripping the MTP head before conversion to pre-empt the M5/M6 `block_count=25` bug — llama.cpp's Qwen3.5 converter **asserts** the head is present | Convert with it, repair after (`check_gguf.py --fix`) |
| First Q4 scoring read 0% | The chat template opens a `<think>` block and the 5-token budget went to scaffolding; every answer was empty | `llama-server --reasoning off` |

Training itself: 6,024 steps, 3 epochs, 96.5 minutes on a community RTX 3090 at
$0.22/hr, peak VRAM 12.2 GiB, effective batch 32, best dev 87.8% at step 4,750.
Kernels that applied: flash-attention-2, Liger fused RMSNorm/SwiGLU/RoPE, paged
8-bit AdamW, NF4 with bf16 compute, TF32, the answer-position LM head and
token-budget batching. `flash-linear-attention` would not import against triton
3.1, so the linear-attention layers used the reference path — as in M6.
