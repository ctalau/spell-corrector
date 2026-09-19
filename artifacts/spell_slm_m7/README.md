# M7 — 2B Q4 teacher distilled into a 0.8B Q4 student (BEA-60K)

Token-level knowledge distillation from the milestone-6 `Qwen/Qwen3.5-2B`
QLoRA corrector (4-bit NF4) into a fresh `Qwen/Qwen3.5-0.8B` QLoRA student of
the same 4-bit recipe. Trained on 64,277 BEA-60K word errors; every number
below is measured on rows held out of training, sentence-disjoint.

## Results — same harness, same rows, greedy decode

| System | n | Acc@1 exact | Acc@1 casefold |
|---|---:|---:|---:|
| Student 0.8B NF4 — frozen BEA-100 | 100 | 89.0% | **89.0%** |
| Student 0.8B NF4 — test | 2,000 | 86.45% | **87.30%** |
| Teacher 2B NF4 — frozen BEA-100 | 100 | 86.0% | 87.0% |
| Teacher 2B NF4 — test | 2,000 | 87.95% | **88.75%** |

The 2,000-row column is the one to read: +-1.5 pp against +-6 pp on 100 rows.
The student recovers all but **1.45 points** of a teacher 2.5x its size.

**The 90% goal was not reached, and is not reachable by distilling this
teacher**: the teacher itself measures 88.75% at n=2,000. M6's headline 91% was
a 100-row number; re-measured here on those same 100 rows the teacher gives
87.0%. See `results/predictions_teacher_nf4_frozen_100.jsonl` versus
`../spell_slm_m6/predictions_m6_qlora_2b.jsonl`: the verdicts differ on 4 rows
but the raw generations differ on 93, so the two harnesses are not decoding
identically. The open question is whether left-padded batched generation
degrades Qwen3.5 (18 of its 24 layers are linear attention with recurrent
state, which does not always respect an attention mask) -- if so, both models
here are under-measured.

## Training

| | |
|---|---:|
| Loss | `0.5*CE(gold) + 0.5*T^2*KL(teacher top-64 \|\| student)`, T=2 |
| Steps / epochs | 6,024 / 3 |
| Wall | 96.5 min on a community RTX 3090 ($0.22/hr) |
| Peak VRAM | 12.2 GiB |
| Effective batch | 32 (micro 8 x accum 4, capped at 2,048 tokens/batch) |
| LoRA | r=32, alpha=64, dropout 0.05, 12.78M trainable |
| Best dev (n=500) | 87.8% casefold at step 4,750 |
| Cost, whole experiment | $0.70 including two failed runs |

Kernels that actually applied: flash-attention-2, Liger fused RMSNorm/SwiGLU/RoPE,
paged 8-bit AdamW, NF4 weights with bf16 compute, TF32, an answer-position LM
head (the loss touches <=10 positions per row, so the head is never run over the
full sequence), and token-budget batching. `flash-linear-attention` would not
import against triton 3.1, so the linear-attention layers used the reference
path -- the same situation M6 ran in.

## Dev accuracy trace

81.0, 80.8, 82.8, 84.0, 84.4, 85.0, 85.2, 85.6, 85.4, 85.0, 85.8, 86.2, 85.2,
86.8, 87.2, 87.0, 87.0, 86.6, **87.8**, 87.2, 87.2, 87.2, 87.4, 87.4
(every 250 steps, n=500, casefold). Full trace in `results/metrics.jsonl`.

## Layout

- `qwen35_0_8b_distill_qlora/` — the best adapter (step 4,750)
- `results/` — metrics and predictions for both models on both splits, the
  training trace, the teacher dump meta, the kernel probe and the run log

## Not done here

The Q4_K_M GGUF export failed on the pod: llama.cpp's Qwen3.5 converter asserts
`mtp_num_hidden_layers != 0`, so pre-stripping the MTP head (to avoid M5/M6's
`block_count=25` bug) breaks conversion outright. The order is fixed in
`scripts/distill/merge_adapter.py` — convert with the head present, repair the
metadata afterwards with `check_gguf.py --fix`.
