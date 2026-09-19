# Experiment 9 — open-weights **kev** in the seat "Jev Choice" occupied

| | |
|---|---|
| **Status** | **completed** (local CPU box only; no pod was rented, $0) |
| **When** | 2026-09-19, branch `claude/kev-vs-jev-comparison-r1sfic`. Local box: 4 vCPU, 15GB RAM, no GPU. |
| **Headline result** | **`kev-4b` reaches 79.0% overall / 94.0% conditional on the frozen 100**, against Hunspell top-1's 60.0% on the same rows — 20 rescues for 1 break. It is pinned at 79% only because Hunspell's pool contains the gold word for 84 of the 100 items. The two small checkpoints are *worse than not reranking*: `kev-0.5b` 50.0%, `kev-0.6b` 58.0%. |
| **Cost** | $0. Wall clock: ~1 min per 100 items for the 0.5B/0.6B checkpoints, ~10 min for 4B (bf16, CPU). |
| **What it settled** | (1) A typed-decision model *is* a competitive chooser, but only at 4B: conditional accuracy goes 59.5% → 69.0% → **94.0%** across 0.5B → 0.6B → 4B. (2) The "Jev Choice ~91%" row is now bracketed by something reproducible — and the gap to it is **candidate recall, not chooser quality**. (3) Below ~1B, a decision model has no orthographic prior at all: it picks Hunspell's word-split artifacts (`an-thing`, `concent rat`, `pe rents`) over the real word. |
| **What it left open** | `kev-8b` was not run (16GB in bf16 does not fit this box's 15GB; a pod was not judged worth it for a checkpoint kev's own card puts 1.5 points above 4B out of domain). kev over a *wider* candidate pool is unrun — Hunspell alone saturates at 84% coverage on this set, so the interesting condition needs the classic BM25 ∪ dense union that this repository does not own. n=100, ±8-10 pp. |
| **Artifacts** | [`reports/kev_choice/`](../../kev_choice/) — one directory per condition, each with `report.json` and `predictions.jsonl`. |
| **Code** | [`scripts/kev/run_kev_choice.py`](../../../scripts/kev/run_kev_choice.py). Plan: [PLAN.md](PLAN.md). |

---

## Why this experiment exists

Three milestone reports quote a number this repository has never been able to
re-run: **"Jev Choice (approx, prior) ~91% — API chooser over lists"**
(`artifacts/spell_slm_m4/milestone4_direct_report.md`, and
`"jev_choice_approx": 0.91` in the M4/M5/M6 metrics JSONs). Jev is TypeSafe's
hosted decision model — typed questions in, calibrated probabilities out, one
forward pass, no decoding. It is closed, and the measurement came from a
session whose code is not in this repository. It has sat in the comparison
table as the one row nobody here can reproduce.

[`jaredpalmer/kev`](https://github.com/jaredpalmer/kev) is an open-weights
reconstruction of that architecture: a LoRA adapter and a pointer head on a
Qwen backbone, a block-causal mask that lets every question see the document
but never a sibling question, and a softmax over option spans read off a
`<decide>` token. It serves the same `/v1/systemone` contract and publishes
0.5B / 0.6B / 4B / 8B checkpoints. So the Jev row can be replaced by one that
runs here.

## Setup

- **Items**: the frozen 100, recovered from
  `artifacts/spell_slm_m7/results/predictions_teacher_nf4_frozen_100.jsonl` —
  the same 100 typos M3 through M7 were scored on.
- **Candidates**: Hunspell, via `spelling_reranker.candidates.build_pool`. The
  local `en_US.dic`/`en_US.aff` SHA-256 hashes match
  `artifacts/hunspell_metadata.json` exactly, so these are the pools earlier
  experiments saw.
- **The ask**: one `choice` question per typo. `state` = the sentence with the
  typo marked `<TYPO>…</TYPO>`; `criteria` = the candidate words with no
  descriptions; the answer is the argmax of the returned probabilities.
- **Instruction string**: one, fixed before the first run and never varied.
  BEA-60K is locked — no prompt search, no condition chosen after seeing a score.
- **Scoring**: casefold, as in milestone 3.

## Results

Frozen 100. `conditional` = correct / the 84 rows where Hunspell's list actually
contained the gold word; it is the number that measures *the chooser*.

| System | list | overall | conditional | vs Hunspell top-1 | p50 latency |
|---|---|---:|---:|---:|---:|
| Hunspell top-1 (do nothing) | — | 60.0% | — | — | — |
| `kev-0.5b` fp32 | top-8 | **50.0%** | 59.5% | −10 pp | 252 ms |
| `kev-0.5b` fp32 | uncapped | **50.0%** | 59.5% | −10 pp | 298 ms |
| `kev-0.6b` fp32 | top-8 | **58.0%** | 69.0% | −2 pp | 320 ms |
| `kev-0.6b` fp32 | uncapped | **58.0%** | 69.0% | −2 pp | 340 ms |
| `kev-4b` bf16 | top-8 | **79.0%** | **94.0%** | **+19 pp** | 5,623 ms |
| *Hunspell coverage@8* | *ceiling, not a system* | *84.0%* | — | — | — |
| Jev Choice (hosted, prior, approx) | classic ~100 | *~91%* | *~98%* | — | — |

Uncapping the list changes almost nothing, because **Hunspell saturates**: gold
is in the pool for 60% of items at rank 1, 81% by rank 4, 84% by rank 8, and
84% however far you go (the longest pool this set produces is 15 words).

### The chooser is not the bottleneck at 4B

| | rescues | breaks | net |
|---|---:|---:|---:|
| `kev-0.5b` vs Hunspell top-1 | 5 | 15 | **−10** |
| `kev-0.6b` vs Hunspell top-1 | 11 | 13 | **−2** |
| `kev-4b` vs Hunspell top-1 | 20 | 1 | **+19** |

`kev-4b` gets 79 of 84 in-list items right. Its five misses are the same
near-neighbour confusions milestone 3 catalogued for the fine-tuned picker —
two of them (`dialy→diary`, `Frence→France`) are on M3's list verbatim:

```
afrid     gold=afraid   picked=arid     conf=0.72
dialy     gold=daily    picked=diary    conf=0.60
Frence    gold=French   picked=France   conf=0.42
dalls     gold=dolls    picked=falls    conf=0.32
wheather  gold=weather  picked=whether  conf=0.74
```

The remaining 16 points to a perfect score are the 16 items where **the gold
word is not in Hunspell's list at all**. No chooser can fix those; that is
experiment 6's finding restated, and it is why Jev's ~91% was measured over
classic ~100-candidate unions with 93% gold-in-list, not over Hunspell alone.
On *conditional* accuracy — chooser against chooser — kev-4b's 94.0% is in the
neighbourhood of the ~98% Jev's 91%/93% implies.

### Below 1B there is no orthographic prior

`kev-0.6b`'s wrong picks are not near-misses. They are Hunspell's word-split
suggestions, which no speller-aware model would ever rank first:

```
anthing    gold=anything    picked=an-thing     list=[anting, anything, anteing, an thing, an-thing]
concentrat gold=concentrate picked=concent rat  list=[concentrate, concent rat, concent-rat, …]
perents    gold=parents     picked=pe rents     list=[repents, percents, parents, pe rents, …]
frends     gold=friends     picked=f rends      list=[fends, rends, friends, fiends, trends]
```

The pointer head scores what an option *means* in context, not how far it is
from the typo. At 0.5-0.6B there is not enough backbone to notice that
`concent rat` is not a word; at 4B there is.

### Calibration

Confidence separates right from wrong at every size, and sharply at 4B (mean
confidence on in-list rows):

| | correct | wrong |
|---|---:|---:|
| `kev-0.5b` | 0.573 | 0.349 |
| `kev-0.6b` | 0.659 | 0.336 |
| `kev-4b` | **0.962** | 0.560 |

That is the property the architecture is for, and it survives a domain
(orthography) that none of these checkpoints was trained on — kev's own model
card trains on banking77, BoolQ, AG News, MNLI, SST-5 and Yelp, and warns that
calibration is verified only on those distributions.

## Honest caveats

- **n=100**, one fixed set, ±8-10 pp of binomial noise. The 0.5b-vs-0.6b gap
  (8 points) is inside that noise; the 0.6b-vs-4b gap (21 points) is not.
- **The Jev row is still not reproduced.** It was measured on different
  candidate lists, in a session this repository does not have. kev is a
  reconstruction of Jev's architecture by a third party, not Jev. Nothing here
  licenses editing the 91% or claiming it was confirmed or refuted.
- **kev-4b ran in bf16**, not the fp32 its published numbers use; kev's README
  puts the difference in the third decimal of the probabilities, which can flip
  an argmax on a close call.
- **Latency is not a product number.** 5.6s p50 is a 4B backbone doing a
  prefill on 4 vCPUs. On a GPU this is one forward pass over ~150 tokens.
