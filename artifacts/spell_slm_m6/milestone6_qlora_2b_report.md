# Milestone 6 Report — QLoRA direct spelling correction (Qwen3.5-2B)

- Seed: **1337**
- Eval: frozen BEA-100 (same holdout as M1–M5)
- Base: `Qwen/Qwen3.5-2B` rev `15852e8c16360a2fea060d615a32b45270f8a8fc`
- Method: **4-bit NF4 QLoRA** (bitsandbytes double-quant, bf16 compute), **fresh LoRA** (not continue M4/M5)
- Prompt / data: same as M4/M5 (`prompts/direct_correct_v1.txt`, `data/direct_train/`)

## Plain English

We trained a ~2B Qwen spelling fixer with the same memory-saving 4-bit recipe as M5, but starting from scratch on the larger base. One epoch on the ~24k direct-train rows. On the frozen 100-example BEA holdout it hit **91% Acc@1** (exact and casefold) — better than the 0.8B M4/M5 runs (mid-80s / low-80s). Training fit easily on a Community RTX 3090 (~5.3 GiB peak VRAM) with micro-batch 2 and grad checkpointing. Extra CUDA kernels (flash-attn, causal-conv1d, FLA) did not install in time, so the run used slower reference PyTorch fallbacks; `torch.compile` was skipped because PEFT + bitsandbytes QLoRA does not play nicely with Trainer.

## Speedups / kernels

| Speedup | Result |
|---|---|
| Micro-batch **2** × accum **16** (eff **32**) | **Stuck** (first attempt; no OOM) |
| Gradient checkpointing | **ON** |
| `torch.compile` | **Skipped** — PEFT+bitsandbytes QLoRA incompatible with Trainer |
| `flash-attn` | **Not installed** (skipped after bootstrap time sink) |
| `causal_conv1d` | **Not installed** (build killed mid-way) |
| `flash-linear-attention` | **Not installed** |

Training logs: `causal_conv1d_fn` / `chunk_gated_delta_rule` fell back to reference PyTorch ops (correct but slower).

## Training recipe

| Setting | Value |
|---|---|
| Base precision | 4-bit NF4 + bf16 compute |
| Init | Fresh LoRA (~10.9M trainable / ~2.22B total ≈ 0.49%) |
| LoRA r / α / dropout | 16 / 32 / 0.05 |
| Batch × accum | 2 × 16 (eff 32) |
| Grad checkpointing | ON |
| max_seq_length | 256 |
| Optim | paged_adamw_8bit |
| Epochs | 1 (753 optimizer steps) |
| LR | 2e-4 |
| Train wall | **77.1 min** (4624 s) |
| Peak VRAM | **5.31 GiB** |
| Final train loss (logged) | 0.1826 (first 0.4474) |

## Results — Acc@1 on frozen 100

| System | Acc@1 casefold | Acc@1 exact |
|---|---:|---:|
| Hunspell top-1 | 60% | — |
| M4 direct bf16 LoRA | 84% | 84% |
| M4 GGUF Q4 | 86% | 85% |
| M5 QLoRA | 82% | 81% |
| M5 GGUF Q4 | 81% | 81% |
| **M6 QLoRA 2B (this)** | **91%** | **91%** |

Eval: greedy, `max_new_tokens≤5`, device CUDA (4-bit base + adapters). GPU latency p50 ~0.095 s; wall ~11.7 s for 100.

## RunPod

- Pod id: **`iddztzpa5f0ib3`**
- GPU: Community **RTX 3090**, **$0.22/hr**, host CUDA **13.0**
- Image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Stack: torch 2.5.1+cu124, transformers 5.18.0.dev0, peft 0.21.0, bitsandbytes 0.50.2
- Cost estimate: ~2.2–2.5 hr wall from create→delete ≈ **~$0.48–$0.55** at $0.22/hr
- Pod **deleted** after artifact sync

## Artifacts

- Adapters: `models/qwen35_2b_direct_qlora/` (also under `spell-corrector/artifacts/spell_slm_m6/`)
- Metrics: `results/metrics_milestone6.json`
- Predictions: `results/predictions_m6_qlora_2b.jsonl`
- Config: `configs/milestone6_qlora_2b.yaml`
- Train log: `logs/m6_train_eval.log`

## Notes / follow-ups

- No mid-train eval (avoids prior OOM pattern).
- Optional CPU/GGUF for 2B was **not** done (GPU eval + sync prioritized).
- Lots of VRAM headroom on 3090; a later run could bump micro-batch (e.g. 4 or 8) and/or install kernels for faster steps.
