# M6 QLoRA 2B direct corrector — Q4_K_M GGUF (CPU serving)

Merged QLoRA (`qwen35_2b_direct_qlora`) into `Qwen/Qwen3.5-2B`, converted to GGUF, quantized **Q4_K_M**. MTP/`block_count` metadata rewritten (same fix as M5).

## Results (frozen seed-1337 BEA-100)
- Acc@1 exact **86%** / casefold **87%** (`metrics_m6_qlora_q4_bea100.json`)
- Prior M6 GPU HF+PEFT: **91%**
- CPU latency (llama.cpp, greedy, max_tokens=5, threads=8, ctx=1024): p50 **~0.38 s**, wall **~40 s**/100, peak RSS **~3.1 GiB**, disk **~1.2 GiB**

## Serve
```bash
llama-server -m qwen35_2b_direct_qlora_merged-Q4_K_M.gguf -t 8 -c 1024 --jinja --reasoning off
# POST /v1/chat/completions  temperature=0  max_tokens=5
```
Prompt template: `direct_correct_v1.txt` (sentence with `<TYPO>…</TYPO>` → corrected word).

Requires `git lfs pull` for the `.gguf`.
