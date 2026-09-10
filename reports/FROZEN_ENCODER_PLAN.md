# Frozen ModernBERT encoder + small spelling selector

Date: 2026-09-09. Status: implementation and execution plan; no new training performed.

Pinned Hugging Face revision for this implementation (resolved 2026-09-09):
`answerdotai/ModernBERT-base` @ `8949b909ec900327062f0ebf497f51aef5e6f0c8`.
Launch: see [FROZEN_ENCODER.md](FROZEN_ENCODER.md).

## 1. Diagnosis before implementation

The observed failure is **transfer from synthetic validation to real errors**, not an observed rise in synthetic validation loss. Synthetic validation reached 92.42% conditional top-1 and was still improving; real BEA conditional accuracy was 80.14%. Both refer to solvable examples in the old 16-slot system.

Leading hypotheses, in order of current evidence:

1. Synthetic examples do not adequately represent real error/context distributions. This is the clearest observed mismatch, although no controlled ablation proves its causal share.
2. Synthetic validation may overestimate generalization to unseen spelling mappings: train and validation share the typo table, despite article separation. Test pair-disjoint evaluation.
3. A randomly initialized encoder has limited language-learning supervision. Pretrained representations may transfer better. That is the hypothesis this experiment tests, not an established diagnosis.
4. Remaining limitations include optimization duration (validation was still improving), input/truncation issues, candidate rank shortcuts and label/extraction ambiguity.

Do not infer memorization merely from low training loss, or declare that more synthetic copies solve the gap.

Recorded data: 2,081,171 training examples, 20,773 synthetic validation examples, a table of 410,555 usable entries covering 46,825 correct word types; two epochs and 8,130 optimizer updates. Effective batch was 128 microbatch × 4 accumulated batches = 512 examples per usual update, about 4.16M example presentations total. Epoch-edge handling can differ; this is not 4.16M unique examples.

Mean serialized length was 202.95 byte/special-token positions. One pass over stored examples therefore represents about 422M input positions, or roughly 845M over two epochs, including repeated text, candidate strings and delimiters. These are **not language-model training tokens or unique corpus bytes**. The main loss has one candidate-index label per example; the auxiliary objective masks only some context bytes.

The often-cited Chinchilla heuristic is around 20 training tokens per parameter for compute-optimal autoregressive pretraining: about 1.75B tokens for 87M parameters. It is a rough reference, not a data requirement for this byte-level supervised reranker or proof that a particular dataset size reaches 75%. Tokenization, objective, data repetition and compute allocation differ. Modern pretrained models can be trained far beyond compute-optimal ratios to improve downstream economics. We should reuse that pretraining, not recreate it.

Sources: [run summary](../artifacts/train_summary.json), [data statistics](exp2_data_stats.json), [loss graph](training_loss.png), [Chinchilla paper](https://arxiv.org/abs/2203.15556). A development approach consistent with [Karpathy's published recipe](https://karpathy.github.io/2019/04/25/recipe/) is to inspect actual inputs, establish simple baselines, overfit a tiny batch to validate implementation, use pretrained models and change one factor at a time. This is an application of his published advice, not a claim about what he would personally diagnose here.

## 2. Model and exact trainable boundary

Choose **answerdotai/ModernBERT-base**. It is a modern encoder available as of September 2026, originally published in 2024, not a claim to be the newest or smallest model. The base checkpoint has 149M parameters, 22 layers, and was pretrained on 2T tokens of English/code. It is compact relative to LLMs, not a tiny mobile encoder. Use its native subword tokenizer and encoder hidden states. [Official model card](https://huggingface.co/answerdotai/ModernBERT-base)

- Load tokenizer and base encoder with a pinned Hugging Face revision. Record the resolved revision, library versions, precision and attention backend.
- Freeze every encoder parameter, including embeddings and normalization. Keep encoder in eval mode so dropout is disabled even when selector.train() is called.
- Extract under torch.no_grad(); only pass detached pooled representations to the selector. No optimizer state for the encoder. Do not train newly added special-token embeddings: use existing separator tokens and offset mappings.
- Preserve Hunspell raw first ten suggestions: slice before deduplication/length filtering. Preserve order and NFC exact equality. No gold-dependent input construction.
- The entire typo/context/candidate list goes through the encoder **once per example**, not once per candidate. The new selector outputs one score per valid candidate.

### Input and selector

Serialize original left context, marked typo, right context, then candidate spans separated with existing tokenizer separators. Mark spans through tokenizer offset mappings and sequence metadata, not string search; repeated spellings must map to the right occurrence. The text encoder sees the actual typo, not the corrected sentence.

Maximum 512 subword tokens, dynamic padding. Reserve space for the complete typo/candidate segment first; truncate context symmetrically around the typo. Log truncation and serialization failures. Never truncate a candidate silently, insert gold, or discard failures from the overall denominator. If required segments exceed the cap, use the documented candidate-zero fallback.

Pool last-layer hidden states into context vector x (context-only positions), typo vector t, and candidate vector c_i. Use zero context vector when no context exists. Hidden size is 768. Build each candidate's feature vector:

[c_i, t, x, c_i * t, abs(c_i - t), one_hot_rank_i, four spelling features]

The four cheap features are normalized edit distance, candidate-minus-typo character length, exact case-sensitive match indicator, and casefold-match indicator. They use no labels or external frequencies.

Train a shared MLP: **3854 → 256 → 64 → 1**, GELU between layers, dropout 0.1. Approximately 1.003M trainable parameters. Mask invalid candidate logits, then apply cross-entropy across the candidate list. Fit any numeric feature scaling on training data only.

This tests whether fixed contextual representations support selection. It does not establish that a frozen encoder is as effective as finetuning, or that this unfamiliar joint-input format fully exploits its pretraining.

## 3. Cheap pod and feature caching

| Setting | Selection |
|---|---|
| GPU | **1 × NVIDIA RTX A5000, 24 GB VRAM** |
| Product | Runpod on-demand Pod; no interruptible/spot run |
| Host RAM / CPUs | At least 24 GB RAM and 8 allocated vCPUs; prefer 32 GB+ RAM |
| Disk | 100 GB temporary container disk; stream caches in shards |
| Encoder precision | bf16 if supported, otherwise validated fp16 |
| Encoder extraction batch | Start at 16 examples, 512 tokens; test 32 only after measuring |
| Selector batch | 512 examples from cached features |
| Memory gate | Worst-case measured peak below 20 GiB on the 24 GB card |

Published Runpod pricing checked on 2026-09-09 lists RTX A5000 at **$0.27/hour**; actual offers and CPU/RAM vary. Container storage is listed at $0.10/GB/month, adding approximately $0.014/hour for 100 GB using a 720-hour month. Confirm the console's total. [Pricing](https://www.runpod.io/pricing)

Frozen weights occupy only about 0.30 GB at two bytes/parameter; peak runtime also includes attention buffers, hidden states and allocator overhead. Target 4–10 GB working VRAM initially, **an estimate to verify**, not a guaranteed peak. GPU VRAM and host RAM are different allocations. A frozen encoder needs neither backward activations nor Adam states. Reduce extraction batch if the actual peak exceeds the gate.

Use a CUDA-compatible PyTorch image and a tested Transformers release supporting ModernBERT (official support begins at 4.48). Pin the resolved working environment after the smoke check; don't assume the old byte-model runtime works unchanged. Validate CUDA, tokenizer offsets and attention implementation before the data job.

Cache only x, t and up to ten c_i vectors, plus metadata and scalar features. FP16 storage is about 18,432 bytes/example: approximately **3.69 GB for 200k examples**, 0.37 GB for 20k validation examples, or 38.36 GB for all 2.08M train examples, excluding metadata. Reconstruct interaction features during selector training. Do not cache full token hidden states or all expanded MLP features.

Cache key: encoder/tokenizer revisions, input data hash, serialization/pool settings, max length, pooling, precision and code SHA. Data changes or context ablations require new encoder features. Head learning rate, head seed and head architecture do not.

**Initial budget: $5, with a 12-pod-hour stop/checkpoint.** Estimate 4–10 hours for setup, extraction and head comparisons, giving about $1.42–$3.55 including storage and 25% contingency. These are unmeasured allowances; time 10k examples before extracting 200k. Stop/replan if projected work exceeds 12 hours. No GPU should idle during implementation or human analysis. An optional full-data expansion has a separate $15/40-hour cap and is not part of the first run.

Use the console for explicit GPU/rate selection. The old launcher does not enforce --max-price and auto-runs BEA unless --idle is used. Its idle mode requires manual clone/setup/execution. Archive and verify results before terminating this specific pod.

## 4. Data and comparisons

First freeze a **200k-example train subset** selected deterministically by example ID, with no new balancing or corruption rules. Construct D-pair from official held-out articles, disjoint in NFC (typo, gold) pairs from training; target 20k, report any shortfall. The historical D-in set remains a separate reference. Keep unfiltered error inventories for overall metrics as well as solvable rows for training.

The independent D-real development/test split and BEA restrictions from the previous plan still apply. If D-real is unavailable, the pilot tests synthetic generalization only; it cannot establish real-error improvement. Keep the old benchmark JSONLs out of training and error mining. Public checkpoint pretraining may have unknown benchmark overlap; record that limitation instead of asserting verified absence.

Run these controls on the same new splits:

| Arm | Trainable model | Purpose |
|---|---|---|
| H0 | None; Hunspell first candidate | Minimum baseline |
| H1 | MLP on rank + four spelling features only | Can cheap clues explain the improvement? |
| H2 | Linear head on frozen encoder + scalar features | Simple representation baseline |
| H3 | The proposed ~1M MLP on identical cached features | Does nonlinear selection help? |

Train H1–H3 with AdamW, learning rate 1e-3, weight decay 0.01, batch 512, seed 1337, maximum 10 epochs, validation each epoch, patience 2. Use validation conditional accuracy for early stopping with loss as tie-breaker. Always evaluate the last epoch. If H3 is unstable, allow one predeclared 3e-4 retry; no large search. Record all runs. Confirm only the selected winner with seeds 1338 and 1339 using the same cache.

Evaluate the existing byte model on these same splits as an additional reference, exposing only top10 candidates. Record that it was trained on more examples and may have seen D-pair mappings, so it is a historical reference, not a matched-data causal control.

Report top1/top3 conditional accuracy, coverage/overall accuracy where inventories allow, retention/rescue, rank slices, unseen-pair performance, runtime, and per-example predictions. Export head training loss, development loss and development accuracy plots plus machine-readable logs. Never compare head training loss to the old combined ranking+MLM training loss.

Admission to full-data expansion: H3 or H2 must beat H1 by at least 1 pp on D-pair and show a positive paired document-bootstrap difference; real-data improvement must be evaluated if D-real exists. This is a pilot gate, not evidence of 75% already achieved. With small real sets, report confidence intervals and avoid treating insignificant tiny gains as decisive.

A failed frozen-head pilot can reflect features, serialization or the frozen boundary. It does not prove all pretrained encoders are ineffective. Check a no-context development ablation to test context use; do not unfreeze the backbone as a hidden fix.

## 5. Implementation tasks and proposed interfaces

The following files/commands are **to implement**, not currently runnable:

1. configs/train_frozen_modernbert.yaml: all settings above, including frozen=true and exact revisions.
2. spelling_reranker/frozen_encoder.py: tokenizer, span mapping, pooled features and frozen-state enforcement.
3. scripts/cache_frozen_features.py: deterministic split input, batched extraction, sharded cache, manifest and throughput/memory report.
4. scripts/train_frozen_selector.py: linear/MLP controls, head-only optimizer, early stopping and plot/log exports.
5. scripts/evaluate_frozen_selector.py: matching serialization/pool policy, prediction ledger and metrics for named development splits.
6. A separate explicit final-benchmark adapter; no automatic BEA download/evaluation inside training scripts.

Proposed command sequence after those interfaces exist:

    python scripts/cache_frozen_features.py --config configs/train_frozen_modernbert.yaml --smoke-examples 10000
    python scripts/cache_frozen_features.py --config configs/train_frozen_modernbert.yaml
    python scripts/train_frozen_selector.py --config configs/train_frozen_modernbert.yaml --arm scalar
    python scripts/train_frozen_selector.py --config configs/train_frozen_modernbert.yaml --arm linear
    python scripts/train_frozen_selector.py --config configs/train_frozen_modernbert.yaml --arm mlp
    python scripts/evaluate_frozen_selector.py --config configs/train_frozen_modernbert.yaml --split d-pair

Acceptance tests before paid extraction:

- Encoder state hash unchanged after selector optimization; all encoder gradients absent and optimizer contains only head parameters.
- Encoder remains eval even when head trains; cached and online predictions agree within declared numerical tolerance.
- Overfit 32 solvable training examples with the MLP; if unsuccessful, inspect inputs/masks/labels before scaling.
- Correct offsets for repeated strings, punctuation, Unicode and subword fragments; no new untrained special embeddings.
- Padded candidates never win, first-ten policy cannot backfill rank eleven, and serialization failures stay in overall denominator.
- Pair/document split overlap checks and deliberate cache invalidation when input policy changes.

Commit configs, dependency lock, manifests, aggregate metrics, plots, head weights and reproducibility instructions. Keep large caches outside normal Git and record their hashes. Freeze the selected model before the single final BEA evaluation; success remains >=75% overall under the primary top10 policy. This plan tests whether existing language representations make the task easier at low cost; it promises no particular accuracy.
