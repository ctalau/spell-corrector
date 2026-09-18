# Milestone 5 Report — QLoRA direct spelling correction (Qwen3.5-0.8B)

- Seed: **1337**
- Eval: frozen BEA-100 (same holdout as M1–M4)
- Base: `Qwen/Qwen3.5-0.8B` rev `2fc06364715b967f1860aea9cf38778875588b17`
- Method: **4-bit NF4 QLoRA** (bitsandbytes double-quant, bf16 compute)
- Init: **continued from M4 LoRA adapters** on the 4-bit base (`PeftModel.from_pretrained(..., is_trainable=True)` — succeeded)
- Prompt / data: same as M4 (`prompts/direct_correct_v1.txt`, `data/direct_train/`, 24,066 train rows)

## Plain English

We retrained the tiny spelling fixer with a memory-saving 4-bit base (QLoRA) and speed tricks, starting from the Milestone-4 LoRA weights instead of from scratch.

- On the frozen 100-test it got the right word **82 times out of 100** (casefold; **81** exact).
- That is a bit below M4’s bf16 LoRA (**84%**) and below the M4 GGUF Q4 serve (**85%/86%**).
- Training finished in about **13 minutes** on a 3090 — about **6× faster** than M4’s ~76 minutes — mostly from a larger micro-batch (8 vs 2) and turning off gradient checkpointing.
- Continuing M4 for a full second epoch likely nudged the adapters a little off the M4 sweet spot; accuracy did not improve over M4.

## Continue-from-M4

| Question | Answer |
|---|---|
| Could QLoRA load M4 adapters on 4-bit base? | **Yes** |
| Used for this run? | **Yes** (`mode=continue_from_m4`) |
| Fresh LoRA fallback? | Not needed |

## Speedups that stuck

| Speedup | Result |
|---|---|
| Micro-batch **8** × accum **4** (eff **32**) | **Stuck** (no OOM; ~18 GiB peak) |
| `torch.compile` | **Blocked** — transformers/PEFT Trainer rejects compile on quantized PEFT models; script auto-skips |
| `flash-attn` 2.8.3 | **Installed & importable** |
| `causal_conv1d` | **Failed** — CUDA host 13.0 vs torch cu124 mismatch when building from source |
| `flash-linear-attention` / `fla` | Package installed but **unusable** (Triton 3.1 vs FLA needing ≥3.3); model fell back to reference kernels |
| Gradient checkpointing | **OFF** (4-bit headroom) |

Training still logged: `causal_conv1d_fn` / `chunk_gated_delta_rule` falling back to slow reference PyTorch ops.

## Training recipe vs M4

| Setting | M4 bf16 LoRA | M5 QLoRA (this) |
|---|---|---|
| Base precision | bf16 | **4-bit NF4** + bf16 compute |
| Init | Fresh from base | **Continue M4 adapters** |
| LoRA r / α / dropout | 16 / 32 / 0.05 | same |
| Batch × accum | 2 × 16 | **8 × 4** |
| Grad checkpointing | ON | **OFF** |
| Optim | default AdamW | **paged_adamw_8bit** |
| Steps / epoch | 753 / 1 | 753 / 1 |
| Train wall | **~4572 s (~1.27 h)** | **~775 s (~12.9 min)** |
| Peak VRAM | ~3.9 GiB | **~18.0 GiB** |
| Final train loss | 0.354 | 0.208 (continued; not comparable) |
| First → last log loss | 0.755 → 0.244 | 0.257 → 0.259 |

## Results — Acc@1 on frozen 100

| System | Acc@1 casefold | Acc@1 exact | Notes |
|---|---:|---:|---|
| Hunspell top-1 | 60% | — | |
| M3 FT reranker (classic) | 85% | — | needs pool |
| **M4 direct bf16 LoRA** | **84%** | **84%** | prior best local direct |
| M4 GGUF Q4_K_M serve | **86%** | **85%** | CPU llama.cpp |
| **M5 QLoRA continue-M4 (this)** | **82%** | **81%** | GPU HF+PEFT eval |

### Latency (GPU, RTX 3090, bf16 base + adapters at eval)

| Metric | M4 | M5 |
|---|---:|---:|
| p50 | 0.107 s | **0.085 s** |
| wall (100) | 12.9 s | **10.3 s** |

## Why Acc@1 dropped vs M4

Best working hypothesis: a **full extra epoch** on already-converged M4 adapters (same LR 2e-4) overfit / drifted. Late logged losses rose (~0.26–0.35 in the last fifth). A shorter continue (e.g. 200 steps / lower LR) or **fresh QLoRA from scratch** might match or beat M4; not run in this milestone.

GGUF re-export was **skipped** (adapters saved; Acc@1 already below M4 Q4 serve).



## CPU eval (HF fallback)

GGUF path (merge → F16 → Q4_K_M → llama.cpp) was attempted but **failed to load** after convert: Qwen3.5 MTP/`nextn_predict_layers` made `block_count=25` while only 24 tensor blocks exist. Stripping MTP metadata still left `attention.recurrent_layers` length mismatched vs llama.cpp expectations. M4’s older GGUF loads fine; M5 reconvert with the current `convert_hf_to_gguf.py` does not.

**Fallback used:** HF transformers + PEFT adapters on CPU (fp32 base, not 4-bit), greedy, `max_new_tokens=5`, full frozen BEA-100.

| Metric | M5 GPU (prior) | M5 CPU HF (this) | M4 GGUF Q4 |
|---|---:|---:|---:|
| Acc@1 casefold | 82% | **82%** | 86% |
| Acc@1 exact | 81% | **81%** | 85% |
| latency p50 | 0.085 s (GPU) | **0.350 s** | ~0.22 s |
| peak RSS | — | **4409 MiB** | ~1.2 GiB |
| wall (100) | 10.3 s | **47.7 s** | ~26 s |

Files: `results/metrics_m5_cpu_hf.json`, `results/latency_cpu_m5_hf.json`, `results/predictions_m5_cpu_hf.jsonl`.

Merged HF weights kept at `models/qwen35_0_8b_direct_qlora_merged/` (for a future GGUF fix). Partial GGUF artifacts under `models/gguf/qwen35_0_8b_direct_qlora_merged-*.gguf` (not used for metrics).

## RunPod

- Pod id: **`4wy2rw06dk9f0x`**
- GPU: Community **RTX 3090**, **$0.22/hr**, host CUDA **13.0**
- Image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Stack: torch `2.5.1+cu124`, transformers `5.18.0.dev0`, peft `0.21.0`, bitsandbytes `0.50.2`, flash-attn `2.8.3.post1`
- Wall clock billed (approx): sync ~3 min + bootstrap ~8 min + train ~13 min + eval ~1 min + sync-back ~2 min ≈ **~0.5 h** → cost estimate **~$0.11**
- Pod **terminated** after artifact sync

## Artifacts

- `configs/milestone5_qlora.yaml`
- `scripts/train_direct_qlora.py`, `scripts/runpod_bootstrap_m5.sh`, `scripts/runpod_m5_train_eval.sh`
- `models/qwen35_0_8b_direct_qlora/` (adapters + `train_meta.json` + `loss_curve.json`)
- `results/predictions_m5_qlora.jsonl`, `results/metrics_milestone5.json`
- `results/runs.csv` (appended)
- `logs/m5_{bootstrap,train,eval,train_eval}.log`
- Sync bundle: `/workspace/spell-corrector/artifacts/spell_slm_m5/`

## Commands (repro)

```bash
cd /workspace/spell-slm-candidate-rerank
# On GPU pod after bootstrap:
bash scripts/runpod_bootstrap_m5.sh
source .m5_env
python scripts/train_direct_qlora.py --config configs/milestone5_qlora.yaml
python scripts/eval_direct.py --config configs/milestone5_qlora.yaml \
  --adapter-dir models/qwen35_0_8b_direct_qlora
```
