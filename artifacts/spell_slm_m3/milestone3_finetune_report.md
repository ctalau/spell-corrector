# Milestone 3 Report — Fine-tuned contextual reranker (Qwen3.5-0.8B LoRA)

- Seed: **1337**
- Eval: frozen BEA-100 (same as M1/M1b/M2); **held out of train+val**
- Base model: `Qwen/Qwen3.5-0.8B` rev `2fc06364715b967f1860aea9cf38778875588b17`
- Method: **LoRA** adapters only (base weights frozen)
- Objective: pointwise yes/no (same prompt as M2: `prompts/rerank_pointwise.txt`)
- Eval pools reused from M1b (`results/predictions_milestone1b.jsonl`), budget **100**

## Hold-out (hard rule)

Every example in the seed-1337 BEA-100 was excluded from train and val by:
1. `error_index` membership in `data/frozen_sample_100.json`
2. NFC-match of `noisy_sentence` against any frozen sample

Documented in `data/rerank_train/exclusion_meta.json` (**0 leaks** verified in train+val).
BEA word-errors excluded: **134** (100 indices + 34 sentence overlaps). Test = frozen 100 only.

## Training data

| Source | Typos / pairs | Pointwise rows | Notes |
|---|---:|---:|---|
| BEA-60K (non-holdout) | 8000 train + 500 val | ~29.8k of train / 2.5k val | Real sentence context |
| Wikipedia misspellings | 4066 pairs | ~15.2k of train | **Word-only** synthetic sentence `<TYPO>{typo}</TYPO>` |
| **Train total (capped)** | — | **45,000** (9k yes / 36k no) | 1:4 pos:neg |

Candidate negatives for training: **Hunspell + dense_typo** (BM25 skipped in build for speed; gold force-included when missing).

## Training recipe

| Setting | Value |
|---|---|
| Method | LoRA (not full FT) |
| LoRA r / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | q,k,v,o,gate,up,down_proj |
| Trainable params | ~6.4M / 859M (**0.74%**) |
| LR | 2e-4 |
| Epochs / steps | 1 epoch / **1407** optimizer steps |
| Batch × accum | 2 × 16 (effective **32**) |
| Max seq len | **256** (cut from 512 after OOM) |
| Gradient checkpointing | **ON** |
| Mid-train eval | disabled (VRAM) |
| Precision | bf16 |
| Final train loss | **0.114** |
| Train wall | ~6828 s (~1.9 h GPU) |

**VRAM note:** First attempt (batch 8, seq 512) OOM’d at step 201 on `logits.float()` in causal LM loss (~22.7 GiB). Restarted with batch 2 + seq 256 + gradient checkpointing + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` → steady **~3.9 GiB**.

## Results (budget 100) — Acc@1

| Reranker | Pool | Acc@1 | Gold in top-5 | Gold in pool |
|---|---|---:|---:|---:|
| unreanked | `bm25_dense_typo_hunspell` | 28.0% | 75.0% | 92.0% |
| zero-shot 0.8B (M2) | `bm25_dense_typo_hunspell` | 48.0% | 78.0% | 92.0% |
| **fine-tuned 0.8B LoRA** | `bm25_dense_typo_hunspell` | **85.0%** | **92.0%** | 92.0% |
| unreanked | `full_union` | 28.0% | 93.0% | 97.0% |
| zero-shot 0.8B (M2) | `full_union` | 48.0% | 79.0% | 97.0% |
| **fine-tuned 0.8B LoRA** | `full_union` | **87.0%** | **97.0%** | 97.0% |
| unreanked | `slm_large` | 85.0% | 90.0% | 91.0% |
| zero-shot 0.8B (M2) | `slm_large` | 72.0% | 89.0% | 91.0% |
| **fine-tuned 0.8B LoRA** | `slm_large` | **85.0%** | **91.0%** | 91.0% |
| Hunspell alone | — | 60.0% | — | — |

### Vs M2 zero-shot (headline)

| Pool | M2 zero-shot Acc@1 | M3 FT Acc@1 | Δ |
|---|---:|---:|---:|
| `bm25_dense_typo_hunspell` | 48% | **85%** | **+37 pp** |
| `full_union` | 48% | **87%** | **+39 pp** |
| `slm_large` | 72% | **85%** | **+13 pp** (also matches unreanked 85%; no longer hurts) |

**Success criterion:** FT clearly beats zero-shot on conventional/full_union Acc@1 **without destroying slm_large** — **met**. FT also beats Hunspell top-1 (60%) on all three pools.

Eval latency (FT): p50≈5.65 s / sample (3 pools), wall ≈589 s on RTX 3090.

## Plain English

We taught a small language model to pick the right spelling from a list of guesses,
using thousands of labeled typos — while carefully **not peeking** at the 100 test sentences.

- Before teaching (Milestone 2 zero-shot), first-pick accuracy on messy guess-lists was about **48%**.
- After fine-tuning, first-pick on the same lists jumped to about **85–87%**.
- That is better than the old Hunspell-alone shortcut (**60%**), and on the already-strong short SLM guess-list it no longer makes things worse (stays at **85%**, whereas zero-shot had dropped it to 72%).

In short: a little supervised practice turned the 0.8B judge from “somewhat helpful” into a strong first-pick ranker on this frozen test set.

## RunPod

- Pod id: **`7znpbmb4klpugz`** (Community RTX 3090, **$0.22/hr**, host CUDA 12.8)
- Image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Stack: torch `2.5.1+cu124` + transformers `5.18.0.dev0` + peft
- Billed roughly: bootstrap ~20 min + failed OOM run ~10 min + successful train ~1.9 h + eval ~10 min ≈ **~2.5–3.0 h** → cost estimate **~$0.55–0.70**
- Pod **terminated** after artifact sync; `list-pods` empty

## Artifacts

- `scripts/build_rerank_train.py`, `scripts/train_reranker.py`, `scripts/eval_reranker_ft.py`
- `configs/milestone3_finetune.yaml`
- `data/rerank_train/{train,val}.jsonl`, `exclusion_meta.json`, `holdout_error_indices.json`
- `models/qwen35_0_8b_rerank_lora/` (LoRA adapters + `train_meta.json`)
- `results/predictions_m3_ft.jsonl`, `results/metrics_milestone3.json`
- `results/runs.csv` (appended)
- `logs/m3_train_eval.log`
- `docs/milestone3_finetune_report.md`

## Commands (repro)

```bash
cd /workspace/spell-slm-candidate-rerank
source .venv/bin/activate
python scripts/build_rerank_train.py
# On GPU pod (after bootstrap):
python scripts/train_reranker.py --config configs/milestone3_finetune.yaml
python scripts/eval_reranker_ft.py --config configs/milestone3_finetune.yaml --batch-size 8
```


## Training loss curve

Logged every 20 optimizer steps (70 points over 1407 steps / 1 epoch).

| Phase | Approx. steps | Avg logged loss | End of phase |
|---|---|---:|---:|
| Start | first logs | — | **0.60** |
| Q1 | early | 0.19 | 0.15 |
| Q2 | | 0.12 | 0.08 |
| Q3 | mid | 0.11 | 0.09 |
| Q4 | | 0.08 | 0.07 |
| Q5 (last 20%) | late | **0.072** | final log **0.076** |
| Trainer reported train_loss | full epoch | — | **0.114** |

**Did it improve until the end?** Mostly yes: average loss in the last fifth (~0.072) is well below the middle third (~0.107). The curve flattened / got noisier near the end (best single log **0.050**, last log **0.076**) — still trending down overall, not diverging. One epoch was enough for a big Acc@1 jump; a second epoch might shave a little more but risk overfitting the train style.

## Classic mix size (eval pools)

M3 eval reused M1b pools with **budget 100**:

| Pool | Candidates per typo | Gold in list | Notes |
|---|---|---|---|
| `bm25_dense_typo_hunspell` (classic mix) | **exactly 100** (p50=100) | 92% | BM25 ∪ dense_typo ∪ Hunspell, truncated |
| `full_union` | **exactly 100** | 97% | classic ∪ Qwen2.5-3B SLM guesses |
| `slm_large` alone | p50 **4**, mean ~6 (0–19) | 91% | short list |

Related Jev classic expansion (same 100 typos, separate experiment): best Choice-safe union was also capped at **100** options (gold-in-list 93%); uncapped classic union p50 ~**163** (gold-in-list 95%).

## Mistakes (classic mix Acc@1 = 85% → 15 misses)

Scoring is **case-insensitive** (`nfc_lower`).

**Right word never in the 100-list (8):**  
`thursty→thirst`, `Miken→McCain`, `sespend→suspension`, `enjener→engineer`, `vacab→vocabulary`, `aeche→each`, `ur→your`, `dialoging→talking`.

**Right word was in the list; model picked wrong (7):**  
`actulayy→actively` (want *actually*), `sorcess→sorcerers` (*sorceress*), `dialy→diary` (*daily*), `Frence→France` (*French*), `pollusions→pollutions` (*pollutants*), `Goog→Google` (*Good*), `catacumbas→catacombs` (*catacomb*).

Pattern: leftover errors are either **missing candidates** (same hard cases as Jev) or **near-neighbor confusions** (plural/related word/proper name).

On `full_union` (87%): 13 misses — a few hard ones get rescued when the 3B SLM put gold in the list (e.g. `sespend`), but some new wrong picks appear when the list is noisier (`crimbimg→crawling`, `commonder→commoner`).

## Improve setup vs continue training?

**Worth trying (setup):**
1. Train negatives from the **same classic union** used at eval (include BM25 + edit-distance), not only Hunspell+dense_typo.
2. Add **case / diacritic** augmentation and pairwise or listwise loss (not only pointwise yes/no).
3. Force gold into train lists less often / hard-negative mine near-misses (`diary`/`daily`, `France`/`French`).
4. Mild second epoch or lower LR continuation from the adapter — only if val loss still falling.

**Less urgent:** full-parameter FT; bigger backbone (Jev already at 91% as an API chooser).

**Continue as-is:** one more epoch from the saved LoRA is the cheapest experiment if you want a quick delta.

## Generalization (English holdout)

- Test **100** never in train (0 leaks by index + sentence).
- Large lift vs zero-shot on held-out BEA (**48% → 85–87%**) = generalizes inside English BEA-style typos.
- Still fails on proper names, slang/abbreviations, and meaning-changing “corrections” (`dialoging→talking`) — distribution shift / missing recall.
- **Romanian OOD probe:** see `results/romanian_ood_m3.json` (10 invented sentences; English-only train → expected weak unless model relies on generic orthography).

## Latency

| Device | Setting | p50 | Notes |
|---|---|---|---|
| **GPU (RTX 3090)** | FT eval, 3 pools / sample | **~5.65 s / typo** (p95 ~7.5 s) | Wall ~589 s for 100 typos × 3 pools; ~2.7 s/pool for 100-cand classic lists |
| **GPU** | short `slm_large` lists | ~**0.15 s** / pool | Scales with #candidates × forward passes |
| **CPU** | 20 cands / 100 cands | **~7.6 s** / **~39 s** per typo | Load ~7 s; peak RSS ~6.6 GiB (`results/latency_cpu_m3.json`) |

H4 (practical latency) is only partially answered: GPU is usable for batch/offline; interactive CPU likely needs distillation, fewer candidates, or a smaller specialized reranker.

## What the experiment handoff suggests next

From `/workspace/SPELL_SLM_CANDIDATE_RERANK_HANDOFF.md` (truncated draft) + completed milestones:

1. **Keep generator ⊥ reranker** — already doing this.
2. **H2 / H4:** we showed contextual rerank helps after FT; still need cleaner **latency/VRAM/$ per query** on the intended serving path (and/or a lighter head than full 0.8B pointwise over 100 cands).
3. **H3:** union still wins on recall; production recipe is likely **classic (+ optional SLM) candidates → FT picker** (or Jev Choice).
4. Original Q5 (“enough evidence to fine-tune?”) → **yes**; next research steps in the spirit of the handoff:
   - Scale eval beyond the frozen 100 (full Hunspell-eligible BEA) for honest generalization.
   - Tighten candidate gen for the remaining misses.
   - Compare FT 0.8B vs Jev on the **same** classic-100 lists (accuracy / cost / latency).
   - Optional: train a tiny cross-encoder / byte-level reranker as in the spelling-line mapping (ModernBERT / byte reranker) for cheaper serving.

## Romanian OOD (10 invented sentences)

English-only LoRA; gold always planted in a small candidate list (~8–15).

- **Acc@1: 4/10 (40%)** — gold in list **10/10**
- Hits: `maine→mâine`, `Bucuresti→București`, `sanatate→sănătate`, `mancarea→mâncarea` (mostly **diacritic** fixes)
- Misses: `scoala→school` (English!), `intotdeauna→always` (English!), plus leaving typos unchanged (`frumoss`, `intalnit`, `copii`, `romana`)
- Takeaway: **not** Romanian-competent; some transfer on “add diacritics,” but it can fall back to English translations. See `results/romanian_ood_m3.json`.
