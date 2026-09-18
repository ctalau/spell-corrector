# Milestone 6 Report — QLoRA direct spelling correction (Qwen3.5-2B)

- Seed: **1337**
- Eval: frozen BEA-100 (same holdout as M1–M5)
- Base: `Qwen/Qwen3.5-2B` rev `15852e8c16360a2fea060d615a32b45270f8a8fc`
- Method: **4-bit NF4 QLoRA** (bitsandbytes double-quant, bf16 compute), **fresh LoRA** (not continue M4/M5)
- Prompt / data: same as M4/M5 (`prompts/direct_correct_v1.txt`, `data/direct_train/`)

## Plain English

We trained a ~2B Qwen spelling fixer with the same memory-saving 4-bit recipe as M5, but starting from scratch on the larger base. One epoch on the ~24k direct-train rows. On the frozen 100-example BEA holdout it hit **91% Acc@1** (exact and casefold) on GPU — better than the 0.8B M4/M5 runs (mid-80s / low-80s). Training fit easily on a Community RTX 3090 (~5.3 GiB peak VRAM) with micro-batch 2 and grad checkpointing. Extra CUDA kernels (flash-attn, causal-conv1d, FLA) did not install in time, so the run used slower reference PyTorch fallbacks; `torch.compile` was skipped because PEFT + bitsandbytes QLoRA does not play nicely with Trainer.

For CPU serving we used the **best stack** from M4/M5: merge PEFT → GGUF F16 → **Q4_K_M** → `llama-server`. After the same MTP/`block_count` metadata rewrite as M5, frozen BEA-100 scored **86% exact / 87% casefold** at p50 **~0.38 s** (~3.1 GiB RSS; ~1.2 GiB on disk).

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
| **M6 QLoRA 2B GPU (HF+PEFT)** | **91%** | **91%** |
| **M6 GGUF Q4_K_M CPU (this)** | **87%** | **86%** |

GPU eval: greedy, `max_new_tokens≤5`, device CUDA (4-bit base + adapters). GPU latency p50 ~0.095 s; wall ~11.7 s for 100.

## CPU / GGUF Q4_K_M (best serving stack)

Recipe (same as M4/M5): merge PEFT adapters into fp16 base → `convert_hf_to_gguf.py` F16 → `llama-quantize` **Q4_K_M** → serve with **llama.cpp** `llama-server`.

Decode: **greedy**, temperature=0, **max_tokens=5**, reasoning off, jinja chat template. Threads **8**, context **1024**, parallel **1**.

### MTP / block_count fix

Convert produced `block_count=25` + `nextn_predict_layers=1` + `attention.recurrent_layers` length 25, while only tensor blocks `blk.0`…`blk.23` exist (same Qwen3.5 quirk as M5). Fix applied before serve:

- set `qwen35.block_count` **25→24**
- truncate `qwen35.attention.recurrent_layers` **25→24**
- remove `qwen35.nextn_predict_layers`

### CPU metrics

| Metric | M6 GGUF Q4 | M6 GPU | M5 GGUF Q4 | M4 GGUF Q4 |
|---|---:|---:|---:|---:|
| Acc@1 casefold | **87%** | 91% | 81% | 86% |
| Acc@1 exact | **86%** | 91% | 81% | 85% |
| latency p50 | **0.384 s** | 0.095 s | 0.216 s | ~0.22 s |
| latency mean | **0.402 s** | 0.117 s | 0.244 s | — |
| wall (100) | **40.2 s** | 11.7 s | 24.4 s | ~26 s |
| peak RSS | **~3144 MiB** | — | ~1.2 GiB | ~1.2 GiB |
| model on disk | **~1215 MiB** | — | ~505 MiB | ~505 MiB |
| load time | **~1.3 s** | — | — | — |

Files: `results/metrics_m6_qlora_q4_bea100.json`, `results/latency_cpu_m6_gguf_q4.json`, `results/predictions_m6_qlora_2b_q4.jsonl`, `models/gguf/qwen35_2b_direct_qlora_merged-Q4_K_M.gguf`.

Compared to 0.8B Q4 serves (~0.22 s), the 2B Q4 is ~1.7× slower on this CPU (expected). Acc@1 stays above M4/M5 Q4 while trailing the M6 GPU HF eval by ~4–5 pp (quantization / merge gap).

## RunPod (training / GPU eval)

- Pod id: **`iddztzpa5f0ib3`**
- GPU: Community **RTX 3090**, **$0.22/hr**, host CUDA **13.0**
- Image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Stack: torch 2.5.1+cu124, transformers 5.18.0.dev0, peft 0.21.0, bitsandbytes 0.50.2
- Cost estimate: ~2.2–2.5 hr wall from create→delete ≈ **~$0.48–$0.55** at $0.22/hr
- Pod **deleted** after artifact sync

CPU GGUF eval ran on the shared box (no RunPod GPU).

## Artifacts

- Adapters: `models/qwen35_2b_direct_qlora/` (also under `spell-corrector/artifacts/spell_slm_m6/`)
- Merged HF (local): `models/qwen35_2b_direct_qlora_merged/`
- GGUF Q4: `models/gguf/qwen35_2b_direct_qlora_merged-Q4_K_M.gguf` → sync `spell-corrector/artifacts/spell_slm_m6_q4/`
- Metrics: `results/metrics_milestone6.json` (GPU), `results/metrics_m6_qlora_q4_bea100.json` (CPU)
- Predictions: `results/predictions_m6_qlora_2b.jsonl`, `results/predictions_m6_qlora_2b_q4.jsonl`
- Config: `configs/milestone6_qlora_2b.yaml`
- Train log: `logs/m6_train_eval.log`

## Notes / follow-ups

- No mid-train eval (avoids prior OOM pattern).
- CPU/GGUF path completed with M5-style MTP metadata rewrite.
- Lots of VRAM headroom on 3090; a later run could bump micro-batch (e.g. 4 or 8) and/or install kernels for faster steps.
