# M6 QLoRA direct corrector (Qwen3.5-2B)

Fresh 4-bit NF4 QLoRA on `Qwen/Qwen3.5-2B` for marked-typo → corrected-word.

## Results (frozen seed-1337 BEA-100)
- GPU Acc@1 exact **91%** / casefold **91%**
- Train wall ~77 min; peak VRAM ~5.3 GiB on Community RTX 3090
- Batch: micro 2 × accum 16 (eff 32), grad checkpointing ON

## Layout
- `qwen35_2b_direct_qlora/` — PEFT adapters + train_meta / loss_curve
- `metrics_milestone6.json` — GPU eval
- `predictions_m6_qlora_2b.jsonl` — per-example preds
- `milestone6_qlora_2b_report.md` — write-up
- `milestone6_qlora_2b.yaml` — train config
- `direct_correct_v1.txt` — prompt

## Kernels
flash-attn / causal_conv1d / flash-linear-attention not installed; torch.compile skipped (PEFT+bnb).
