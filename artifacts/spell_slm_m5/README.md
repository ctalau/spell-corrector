# M5 QLoRA direct corrector (Qwen3.5-0.8B)

Continue-from-M4 QLoRA adapters + GPU eval + **CPU HF fallback** metrics.

## Results (frozen seed-1337 BEA-100)
- GPU Acc@1 exact **81%** / casefold **82%**
- CPU HF Acc@1 exact **81%** / casefold **82%** (p50 ~0.35 s; ~4.4 GiB RSS)
- GGUF Q4 serve not used for metrics (MTP/block_count convert bug vs llama.cpp)

## Layout
- `qwen35_0_8b_direct_qlora/` — PEFT adapters
- `metrics_milestone5.json` — GPU eval
- `metrics_m5_cpu_hf.json` / `latency_cpu_m5_hf.json` — CPU HF
- `milestone5_qlora_report.md` — write-up
- `direct_correct_v1.txt` — prompt

## CPU / GGUF
- HF CPU: Acc@1 81%/82%, p50 ~0.35s (`metrics_m5_cpu_hf.json`)
- GGUF Q4_K_M: Acc@1 81%/81%, p50 ~0.22s (`metrics_m5_qlora_q4_bea100.json`, `qwen35_0_8b_direct_qlora_merged-Q4_K_M.gguf`)
