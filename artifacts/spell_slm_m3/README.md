# Milestone 3 — Qwen3.5-0.8B LoRA spelling reranker

LoRA adapters fine-tuned for pointwise yes/no candidate reranking on BEA+Wikipedia
(held out seed-1337 BEA-100). Base model: `Qwen/Qwen3.5-0.8B` @ `2fc06364715b967f1860aea9cf38778875588b17`.

Load with PEFT on top of the base model; see `qwen35_0_8b_rerank_lora/train_meta.json`
and `milestone3_finetune_report.md`.

Do not commit secrets. Tokenizer not bundled — use the base HF tokenizer.
