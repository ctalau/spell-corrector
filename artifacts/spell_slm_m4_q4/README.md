# M4 direct corrector — Q4_K_M GGUF (CPU serving)

Merged LoRA (`qwen35_0_8b_direct_lora`) into `Qwen/Qwen3.5-0.8B`, converted to GGUF, quantized **Q4_K_M**.

## Results (frozen seed-1337 BEA-100)
- Acc@1 exact **85%** / casefold **86%** (see `metrics_m4_direct_q4_bea100.json`)
- Prior bf16/HF GPU M4: **84%**
- CPU latency (llama.cpp, greedy, max_tokens=5): p50 ~**0.22–0.24 s**/typo (~1.2 GiB RSS)

## Serve
```bash
llama-server -m qwen35_0_8b_direct_merged-Q4_K_M.gguf -t 8 -c 1024 --jinja --reasoning off
# POST /v1/chat/completions  temperature=0  max_tokens=5
```
Prompt template: `direct_correct_v1.txt` (sentence with `<TYPO>…</TYPO>` → corrected word).

Requires `git lfs pull` for the `.gguf`.
