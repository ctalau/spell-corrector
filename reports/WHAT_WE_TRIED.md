# What we tried, and what it produced

A narrative pass over every system this repository has measured against
BEA-60K, in the order the ideas arrived. The per-experiment write-ups are in
[`reports/experiments/`](experiments/); the index with the full comparison
table is [`reports/README.md`](README.md). This document is the story those
tables do not tell.

> **BEA-60K is locked.** Nothing here was trained, validated, tuned or
> prompt-searched on it. Every number is a measurement.

---

## The one-paragraph answer

The project started as a *reranking* problem — Hunspell proposes, a trained
model disposes — and spent two experiments proving that framing has a hard
ceiling at about 81%, because Hunspell's suggestion pool simply does not
contain the right word for roughly a fifth of real errors. Everything that
followed was an escape from that ceiling. Two escapes worked: **let the model
answer freely instead of picking from a list** (gemma-4-E2B, 90.0%), and
**fine-tune a small model to emit the correction directly** (Qwen3.5-2B QLoRA,
91%, then distilled into an 0.8B student at 87.3%). The best *chooser* we have
ever measured is the one added last — `kev-4b`, at 94.0% accuracy on the items
where the answer was actually on the list — and it confirms the ceiling rather
than breaking it.

---

## 1. The baselines

| System | n | Overall | What it is |
|---|---:|---:|---|
| **Hunspell top-1** | 68,429 | **53.67%** | Take Hunspell's first suggestion. The do-nothing baseline. |
| **Aspell top-1** | 68,429 | **60.56%** | The external baseline the project set out to beat. |
| *Hunspell oracle@10* | 68,429 | *80.34%* | **Ceiling, not a system.** What a perfect chooser could reach given Hunspell's first ten. |
| *Hunspell oracle@16* | 68,429 | *80.89%* | Same, sixteen slots. |

The two oracle rows are the most important numbers in the project. They say
that **any** system that can only pick from Hunspell's list is capped at ~81%,
no matter how good the picker is. Every later result is best read as "how close
to 81%, or how did it get out."

A measured aside: adding Aspell's suggestions to the pool lifts the ceiling to
**87.7%** on a 6,000-error sample. That was ruled out of scope by explicit
decision, not by evidence.

## 2. Trained byte-level rerankers — the original line

| Experiment | n | Overall | Conditional |
|---|---:|---:|---|
| **Exp 1** — 28M byte reranker, 10 slots | not recorded | **62.56%** | 77.84% |
| **Exp 2** — 87M byte reranker, 16 slots | **68,429** | **64.82%** | 80.14% |

Exp 1 met its goal: beat Aspell. Exp 2 scaled the model 3x and is **the only
system in this repository ever run on the entire benchmark** — every other
number on this page is n=100 to n=2,000. It missed its 75% target, and the
reason is arithmetic, not training: `64.82% = 80.89% coverage × 80.14%
conditional`. The reranker was already converting four of every five
solvable cases. The pool was the problem.

Exp 3 (a planned six-stage campaign to reach 75%) was written and never run.
Exp 4 (frozen ModernBERT + a selector head) aborted mid-run on a NaN and
produced no BEA number.

## 3. LLM judges — and the discovery that the *format* was the constraint

Instead of a purpose-trained reranker, prompt a general small LLM. Exp 5
established the harness (index mode: show Hunspell's top-8, ask for a number);
exp 6 added three more answer modes and a q4_0/llama.cpp backend. All n=100,
same fixed sample, 4-vCPU CPU box.

| Model | Mode | Overall | Note |
|---|---|---:|---|
| **MiniCPM5-1B** | index | **39.0%** | *Below* Hunspell top-1 (59%) on the same rows — reranking with it is worse than not reranking. And 4x slower. |
| **Qwen3.5-0.8B** | index | 60.0% | Level with Hunspell top-1. |
| **gemma-4-E2B-it** | index | 83.0% | 100% conditional — it never once picked wrong *when gold was on the list*. |
| **gemma-4-E2B-it** | **open** | **90.0%** | List shown as a hint only; free to answer any word. **The best number in the repository.** |
| **gemma-4-E2B-it** | beam | 88.0% | No Hunspell list at all. |
| **gemma-4-E2B-it** | sentence rewrite | 88.0% (73.0% strict) | Strict score wrecked by punctuation reflow, not spelling. |
| gemma-4-E2B q4_0 | open / index / sentence | 87.0% / 81.0% / 86.0% | q4_0 costs 2-3 points, consistently, in exchange for speed. |

This is where the project actually turned. gemma's index mode scored **100%
conditional** — a perfect chooser — and still only reached 83% overall,
because 17 of the 100 sampled errors have no gold candidate anywhere in
Hunspell's list. Letting the model answer freely recovered 58.8% of exactly
those items. **The forced-choice format was the binding constraint, not the
model.**

Exp 7 (gemma q4_0 on GPU + DSPy prompt optimization) is still marked *running*
and has produced no numbers.

## 4. External API models — the rows we cannot re-run

Two numbers carried into the milestone tables from sessions whose code is not
in this repository, both marked approximate:

| System | Acc@1 | What it is |
|---|---:|---|
| **Luna** freeform, first try | **~94%** | Large hosted model, asked to correct the word directly. |
| **Jev Choice** | **~91%** | TypeSafe's hosted decision model, used as a chooser over classic ~100-candidate lists (93% gold-in-list). |

They are the accuracy targets the small-model work was measured against, and
neither can be reproduced here. Experiment 9 (below) addresses the Jev row.

## 5. Fine-tuning Qwen — picker, then direct corrector

All on the **frozen 100**, Acc@1 casefold, where Hunspell top-1 = 60%.

| Milestone | What was trained | Acc@1 | Latency (3090) |
|---|---|---:|---:|
| **M3** | Qwen3.5-0.8B LoRA **picker** over candidate lists | **85%** (classic mix) / **87%** (full union) | ~5.65 s/typo |
| **M4** | Qwen3.5-0.8B LoRA **direct corrector** — no candidates at all | **84%** | **0.107 s** |
| **M5** | Same, QLoRA | 82% | 0.085 s |
| **M6** | **Qwen3.5-2B** QLoRA direct corrector | **91%** | 0.095 s |

M3→M4 is the second escape from the ceiling, and the more useful one: dropping
the candidate pipeline entirely cost **one point** of accuracy and bought a
**53x latency win**. M6 then showed the obvious lever — 2.5x the backbone buys
7 points, landing level with the hosted Jev Choice figure.

Quantized to Q4_K_M for CPU serving via llama.cpp:

| | GPU bf16 | CPU Q4_K_M | Cost of quantizing |
|---|---:|---:|---:|
| M4 (0.8B) | 84% | 86% | — |
| M6 (2B) | 91% | **87%** | −4 pp, at 384 ms p50 on CPU |

## 6. Distillation — 2B teacher into an 0.8B student

Experiment 8 asked whether M6's 2B quality fits in M4's 0.8B body.

| | Acc@1 (n=2,000) | Acc@1 (frozen 100) |
|---|---:|---:|
| Teacher (2B Q4) | **88.75%** | 87% |
| **Student (0.8B, distilled)** | **87.30%** | **89%** |
| Student Q4_K_M on CPU | — | 88% |

The student recovers all but **1.45 points** of a model 2.5x its size, and
beats both previous 0.8B correctors (M4 84%, M5 82%). It missed its >90% target
— but so did the teacher, whose own ceiling was 88.75%. Total cost: **$0.70**,
including two failed runs and two broken-CUDA hosts. The distilled Q4 student
is what the Vercel serving layer and the client-side highlighter page ship.

## 7. kev — an open-weights stand-in for Jev

Experiment 9, added 2026-09-19. [`jaredpalmer/kev`](https://github.com/jaredpalmer/kev)
is an open-weights reconstruction of Jev's architecture (LoRA + pointer head on
a Qwen backbone, block-causal branch mask, softmax over option spans) serving
the same `/v1/systemone` contract. Put in the chooser seat on the frozen 100,
over Hunspell's pool, on the 4-vCPU box:

| System | Overall | Conditional | vs Hunspell top-1 |
|---|---:|---:|---:|
| Hunspell top-1 | 60.0% | — | — |
| `kev-0.5b` | **50.0%** | 59.5% | −10 pp (5 rescues, 15 breaks) |
| `kev-0.6b` | **58.0%** | 69.0% | −2 pp (11 rescues, 13 breaks) |
| **`kev-4b`** | **79.0%** | **94.0%** | **+19 pp (20 rescues, 1 break)** |
| *Hunspell coverage@8* | *84.0%* | — | *ceiling* |

Three findings. **(1)** Size is the whole story: conditional accuracy runs
59.5% → 69.0% → 94.0% across 0.5B → 0.6B → 4B. **(2)** Below ~1B a decision
model has no orthographic prior at all — `kev-0.6b` picks Hunspell's word-split
artifacts (`an-thing`, `concent rat`, `pe rents`, `f rends`) over the real
word, because the pointer head scores what an option *means*, not how far it is
from the typo. **(3)** `kev-4b` is pinned at 79% only by candidate recall: it
gets 79 of the 84 in-list items right, and its five misses are the same
near-neighbour confusions M3 catalogued for the fine-tuned picker — two of them
(`dialy→diary`, `Frence→France`) verbatim.

So the ceiling result holds one more time, from a new direction. **The Jev row
is still not reproduced** — kev is a third party's reconstruction, measured on
different candidate lists — but the gap between kev-4b's 79% and Jev's ~91% is
mostly the gap between an 84%-coverage Hunspell pool and a 93%-coverage classic
union, not a gap between choosers.

---

## What the whole arc says

1. **Reranking Hunspell is capped at ~81%, and we hit the cap.** Exp 2 converted
   80.14% of what was reachable; gemma's index mode converted 100% of it on a
   small sample; kev-4b converted 94%. Three independent systems, same wall.
2. **The two ways out both work.** Answer freely (gemma open, 90.0%) or train
   the model to emit the correction directly (M6, 91%). Neither needs a
   candidate generator at inference time.
3. **Dropping the candidate pipeline is nearly free and enormously fast.**
   M3 → M4: one point of accuracy for a 53x latency win.
4. **Small models can carry it.** A 2B teacher distills into an 0.8B student at
   a 1.45-point loss, quantizes to Q4_K_M at another ~1 point, and serves on a
   CPU at sub-second latency. Total GPU spend for that experiment: $0.70.
5. **Most of these numbers are n=100.** A 95% CI near 40-80% is roughly ±8-10
   pp. Exactly one system — exp 2's 87M reranker, at 64.82% — has ever been run
   on all 68,429 errors, and it is not the one with the best headline.

## Known gaps

- **Two different "frozen 100" sets** are in play. Hunspell top-1 scores 59% on
  the exp 5/6 sample and 60% on the milestone sample, so the LLM-judge rows and
  the milestone rows are not strictly the same denominators.
- **The best headline (90.0%) and the only full-benchmark number (64.82%) are
  not comparable**, and nothing in this repository has closed that gap: the
  trained reranker has never been given the small-sample treatment, and no
  LLM-judge or fine-tuned corrector has ever been run at full scale.
- **Exp 7 never reported.** Exp 3 never ran. Exp 4 aborted.
- **`kev-8b` was not run** (16GB bf16 does not fit the 15GB box).
- **Aspell-in-the-pool** (measured ceiling 87.7%) remains out of scope by
  decision.
