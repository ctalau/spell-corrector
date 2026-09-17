# Milestone 4 — Qwen3.5-0.8B LoRA direct spelling corrector

Fresh LoRA adapters fine-tuned so the model **predicts the corrected word only**,
given a sentence with the misspelling marked as `<TYPO>…</TYPO>`.

- Base: `Qwen/Qwen3.5-0.8B` @ `2fc06364715b967f1860aea9cf38778875588b17`
- Held out: seed-1337 frozen BEA-100 (0 train leaks)
- Frozen-100 Acc@1 (exact / casefold): **84% / 84%**
- GPU latency p50 ≈ **0.11 s** (RTX 3090)

Load with PEFT on top of the base model; see `qwen35_0_8b_direct_lora/train_meta.json`
and `milestone4_direct_report.md`. Prompt: `direct_correct_v1.txt`.

Do not commit secrets. Tokenizer files are included in the adapter dir for convenience;
you may also use the base HF tokenizer.
