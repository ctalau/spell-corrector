# M7 distilled 0.8B corrector — Q4_K_M GGUF (CPU serving)

The M7 student (`../spell_slm_m7/qwen35_0_8b_distill_qlora`, step 4,750) merged
into the fp16 `Qwen/Qwen3.5-0.8B` base, converted to GGUF and quantized
**Q4_K_M**. 505 MiB on disk.

## Results (sequential, one request at a time)

| Split | n | Acc@1 exact | Acc@1 casefold | p50 |
|---|---:|---:|---:|---:|
| frozen BEA-100 | 100 | 88.0% | **88.0%** | 0.602 s |
| test | 2,000 | 85.60% | **86.60%** | 0.597 s |

Measured on the repository's 4-vCPU box with `-t 4`. M4/M5/M6's CPU numbers used
`-t 8`, so the latency is not comparable to theirs; the accuracy is.

Against the same student in 4-bit NF4 on GPU (87.30% casefold at n=2,000),
Q4_K_M costs **0.7 points** — the usual quantization tax.

## Serve

```bash
llama-server -m qwen35_0_8b_distill_q4-Q4_K_M.gguf -t 8 -c 1024 --jinja --reasoning off
# POST /v1/chat/completions  temperature=0  max_tokens=5
```

`--reasoning off` is not optional: the Qwen3.5 chat template opens a `<think>`
block, and with `max_tokens=5` the entire budget goes to reasoning scaffolding.
Every answer comes back empty without it — scored 0% before the flag was added.

Prompt template: `direct_correct_v1.txt` (sentence with `<TYPO>…</TYPO>` → the
corrected word).

## The MTP / block_count fix

Unchanged from M5/M6 in substance, but the order matters and is now recorded:
llama.cpp's Qwen3.5 converter **asserts** `mtp_num_hidden_layers != 0`, so the
multi-token-prediction head cannot be stripped before conversion (M7's first
attempt did exactly that and the conversion died). Convert with the head
present, then repair:

- `qwen35.block_count` **25 → 24** (only `blk.0`…`blk.23` exist)
- `qwen35.attention.recurrent_layers` truncated **25 → 24**
- `qwen35.nextn_predict_layers` removed

`scripts/distill/check_gguf.py --fix` does this and verifies the result;
`gguf_metadata_before_fix.json` and `gguf_metadata.json` are the before/after.

Requires `git lfs pull` for the `.gguf`.
