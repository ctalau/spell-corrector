# Experiment 8 / Milestone 7 — distil the 2B Q4 corrector into the 0.8B Q4 student

| | |
|---|---|
| Status | completed — see [README.md](README.md) |
| Started | 2026-09-18 |
| Branch | `claude/distill-2b-to-07b-bea-fsqdjt` |
| Teacher | `Qwen/Qwen3.5-2B` + M6 QLoRA adapters, 4-bit NF4 — 91% Acc@1 on the frozen BEA-100 |
| Student | `Qwen/Qwen3.5-0.8B`, 4-bit NF4 QLoRA, fresh LoRA r=32 |
| Goal | student above 90% Acc@1, served as Q4_K_M GGUF — **missed**: 87.30% at n=2,000, against a teacher measured at 88.75% |

## Why

M6 showed the 2B is worth ~5-7 points of Acc@1 over the 0.8B (91% vs 84%/82%),
but it is ~1.7x slower on CPU and 2.4x larger on disk. The question here is how
much of that gap is recoverable in the small model when it is trained against
the big one's *distribution* rather than against one-hot gold alone.

## Method

Word-level (token-level) knowledge distillation on the gold path:

1. The teacher runs **once** over the training split, teacher-forced on
   `prompt + gold answer`, and its top-K=64 logits at every answer position are
   written to disk (`dump_teacher_logits.py`). The teacher is never resident
   during student training — distillation costs one teacher epoch, not one
   teacher forward per student step.
2. The student trains against

   ```
   L = (1 - alpha) * CE(gold) + alpha * T^2 * KL(teacher_topK || student)
   ```

   with `alpha = 0.5`, `T = 2.0`, the teacher distribution renormalized over its
   own top-K support. Both models share the same 248,320-token vocabulary, which
   `common.tokenizer_fingerprint` asserts before any KD happens — token-level KD
   across two different tokenizers would be meaningless, so the run aborts
   instead of silently producing noise.

## Data — and the benchmark-policy exception

`reports/README.md` and `CLAUDE.md` lock BEA-60K: never train, validate or tune
on it. **This run trains on BEA-60K**, at the user's explicit instruction, and
so did milestones M4, M5 and M6 before it (`data/direct_train/` was 20,000
BEA rows). The line to hold, then, is not "BEA is untouched" — it is that the
numbers reported are measured on rows the model has never seen:

| Split | n errors | n sentences | Used for |
|---|---:|---:|---|
| `frozen_100` | 100 | 100 | the M4-M6 holdout, reconstructed from the committed M6 predictions — comparability only |
| `test` | 2,000 | 1,771 | scored once, at the end |
| `dev` | 1,000 | 900 | progress tracking and checkpoint selection |
| `val` | 1,000 | 905 | teacher-forced loss |
| `train` | 64,295 | 57,339 | training |

Splits are disjoint **at the level of the source sentence**, not merely the word
error: a BEA line can carry several errors, and putting two of them on opposite
sides of a split would leak the context. `build_data.py` fails the build if any
error index or source sentence appears twice.

The frozen-100 reconstruction was verified error-by-error against
`artifacts/spell_slm_m6/predictions_m6_qlora_2b.jsonl`: all 100 typos, sentences
and error indices match, and the two gold labels M6 hand-corrected (BEA's own
"clean" side reads `climbimg` and `studant`) are carried over so the number is
comparable.

Because training now touches BEA, **no number from this experiment is
comparable to the byte-reranker line (experiments 1-2)**, which was trained on
synthetic WikiText typos and evaluated on all 68,429 BEA errors.

## Training stack

Every item here is a throughput or memory measure, and none of them is allowed
to be load-bearing — each is probed and skipped if unavailable, with what
actually applied recorded in `run_meta.json`:

| Measure | Why |
|---|---|
| Precomputed teacher top-K logits | removes the teacher from the training loop entirely |
| 4-bit NF4 weights, bf16 compute | the M5/M6 QLoRA recipe; ~0.6 GiB of student weights |
| flash-attention-2 (prebuilt wheel only) | falls back to SDPA rather than spending an hour on a source build |
| Liger fused RMSNorm / SwiGLU / RoPE | applied per-instance when it patches the arch |
| `flash-linear-attention` | Qwen3.5 is a hybrid: 18 of 24 layers are linear attention, which otherwise falls back to reference PyTorch ops |
| paged 8-bit AdamW (bitsandbytes) | optimizer state off the critical VRAM path |
| TF32 matmul / cuDNN | free on Ampere and later |
| Length-bucketed batches, dynamic padding | the prompts vary from ~40 to ~250 tokens |
| No gradient checkpointing | the 0.8B in NF4 has the headroom; recompute is pure cost |
| OOM-splitting micro-step | one long batch halves itself rather than killing a multi-hour run |

## What gets reported

`metrics_student_nf4_{frozen_100,test}.json` (GPU, NF4),
`metrics_student_q4km_{frozen_100,test}.json` (llama.cpp Q4_K_M, CPU), the same
two splits scored with the **teacher** so the distillation gap is measured here
rather than quoted from M6, plus `train_summary.json`, `metrics.jsonl` (the full
loss/accuracy trace) and `kernels.json`.
