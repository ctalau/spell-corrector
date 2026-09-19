# What we tried, and what it produced

A narrative pass over every system this repository has measured against
BEA-60K, in the order the ideas arrived. The per-experiment write-ups are in
[`reports/experiments/`](experiments/); the index with the full comparison
table is [`reports/README.md`](README.md). This document is the story those
tables do not tell — including [the part that did not work](#what-failed-and-why).

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

# Part I — what worked

## 1. The baselines

| System | n | Overall | What it is |
|---|---:|---:|---|
| **Hunspell top-1** | 68,429 | **53.67%** | Take Hunspell's first suggestion. The do-nothing baseline. |
| **Aspell top-1** | 68,429 | **60.56%** | The external baseline the project set out to beat. |
| *Hunspell oracle@10* | 68,429 | *80.34%* | **Ceiling, not a system.** What a perfect chooser could reach given Hunspell's first ten. |
| *Hunspell oracle@16* | 68,429 | *80.89%* | Same, sixteen slots. |

A measured aside: adding Aspell's suggestions to the pool lifts the ceiling to
**87.7%** on a 6,000-error sample. That was ruled out of scope by explicit
decision, not by evidence.

> **Conclusion.** The two oracle rows are the most important numbers in the
> project. **Any** system that can only pick from Hunspell's list is capped at
> ~81%, however good the picker. Every later result reads as "how close to 81%,
> or how did it get out." "Hunspell's first ten" is not a baseline you can
> beat — it is a ceiling you can approach.

## 2. Trained byte-level rerankers — the original line

| Experiment | n | Overall | Conditional |
|---|---:|---:|---|
| **Exp 1** — 28M byte reranker, 10 slots | not recorded | **62.56%** | 77.84% |
| **Exp 2** — 87M byte reranker, 16 slots | **68,429** | **64.82%** | 80.14% |

Exp 1 met its goal: beat Aspell. Exp 2 scaled the model 3x and is **the only
system in this repository ever run on the entire benchmark** — every other
number on this page is n=100 to n=2,000.

> **Conclusion.** Exp 2 missed its 75% target, and the reason is arithmetic,
> not training: `64.82% = 80.89% coverage × 80.14% conditional`. The reranker
> was already converting four of every five solvable cases. **Scaling the
> picker was the wrong lever** — the pool was the problem. This is the finding
> the next five experiments are all reactions to.

## 3. LLM judges — and the discovery that the *format* was the constraint

Instead of a purpose-trained reranker, prompt a general small LLM. Exp 5
established the harness (index mode: show Hunspell's top-8, ask for a number);
exp 6 added three more answer modes and a q4_0/llama.cpp backend. All n=100,
same fixed sample, 4-vCPU CPU box.

| Model | Mode | Overall | Note |
|---|---|---:|---|
| **gemma-4-E2B-it** | index | 83.0% | 100% conditional — it never once picked wrong *when gold was on the list*. |
| **gemma-4-E2B-it** | **open** | **90.0%** | List shown as a hint only; free to answer any word. **The best number in the repository.** |
| **gemma-4-E2B-it** | beam | 88.0% | No Hunspell list at all. |
| **gemma-4-E2B-it** | sentence rewrite | 88.0% (73.0% strict) | Strict score wrecked by punctuation reflow, not spelling. |
| gemma-4-E2B q4_0 | open / index / sentence | 87.0% / 81.0% / 86.0% | q4_0 costs 2-3 points, consistently. |

(The two models that scored at or below the baseline are in
[Part II](#b-negative-results--systems-that-worked-and-were-worse-than-nothing).)

> **Conclusion.** This is where the project turned. gemma's index mode scored
> **100% conditional** — a literally perfect chooser — and still reached only
> 83% overall, because 17 of the 100 sampled errors have no gold candidate
> anywhere in Hunspell's list. Letting the model answer freely recovered 58.8%
> of exactly those items. **The forced-choice format was the binding
> constraint, not the model.** The way past ~81% is to stop asking a
> multiple-choice question.

## 4. External API models — the rows we cannot re-run

| System | Acc@1 | What it is |
|---|---:|---|
| **Luna** freeform, first try | **~94%** | Large hosted model, asked to correct the word directly. |
| **Jev Choice** | **~91%** | TypeSafe's hosted decision model, used as a chooser over classic ~100-candidate lists (93% gold-in-list). |

> **Conclusion.** These set the accuracy target the small-model work aimed at,
> and neither is reproducible here — the code is not in this repository and
> both are marked *approx, prior*. Treating them as measurements rather than
> as landmarks would be a mistake. Experiment 9 addresses the Jev row; the Luna
> row remains unexamined.

## 5. Fine-tuning Qwen — picker, then direct corrector

All on the **frozen 100**, Acc@1 casefold, where Hunspell top-1 = 60%.

| Milestone | What was trained | Acc@1 | Latency (3090) |
|---|---|---:|---:|
| **M3** | Qwen3.5-0.8B LoRA **picker** over candidate lists | **85%** (classic mix) / **87%** (full union) | ~5.65 s/typo |
| **M4** | Qwen3.5-0.8B LoRA **direct corrector** — no candidates at all | **84%** | **0.107 s** |
| **M6** | **Qwen3.5-2B** QLoRA direct corrector | **91%** | 0.095 s |

Quantized to Q4_K_M for CPU serving via llama.cpp: M4 84% → 86%, M6 91% →
**87%** at 384 ms p50 on CPU.

> **Conclusion.** M3 → M4 is the second escape from the ceiling and the more
> useful one: **dropping the candidate pipeline entirely cost one point of
> accuracy and bought a 53x latency win.** A candidate generator turns out to
> be a liability at inference time, not an asset. M6 then showed the boring
> lever works — 2.5x the backbone buys 7 points, landing level with the hosted
> Jev Choice figure.

## 6. Distillation — 2B teacher into an 0.8B student

| | Acc@1 (n=2,000) | Acc@1 (frozen 100) |
|---|---:|---:|
| Teacher (2B Q4) | **88.75%** | 87% |
| **Student (0.8B, distilled)** | **87.30%** | **89%** |
| Student Q4_K_M on CPU | — | 88% |

Total cost: **$0.70**, including two failed runs and two broken-CUDA hosts.

> **Conclusion.** The student recovers all but **1.45 points** of a model 2.5x
> its size and beats both previous 0.8B correctors. **Capability at this task
> compresses well** — the 2B teacher was not using its extra parameters for
> anything the 0.8B body cannot hold. This is what the Vercel serving layer and
> the client-side highlighter page ship.

## 7. kev — an open-weights stand-in for Jev

[`jaredpalmer/kev`](https://github.com/jaredpalmer/kev) is an open-weights
reconstruction of Jev's architecture (LoRA + pointer head on a Qwen backbone,
block-causal branch mask, softmax over option spans) serving the same
`/v1/systemone` contract. Put in the chooser seat on the frozen 100, over
Hunspell's pool, on the 4-vCPU box:

| System | Overall | Conditional | vs Hunspell top-1 |
|---|---:|---:|---:|
| Hunspell top-1 | 60.0% | — | — |
| **`kev-4b`** | **79.0%** | **94.0%** | **+19 pp (20 rescues, 1 break)** |
| *Hunspell coverage@8* | *84.0%* | — | *ceiling* |

(`kev-0.5b` and `kev-0.6b` scored *below* the baseline — see
[Part II](#b-negative-results--systems-that-worked-and-were-worse-than-nothing).)

`kev-4b` gets 79 of the 84 in-list items right. Its five misses are the same
near-neighbour confusions M3 catalogued for the fine-tuned picker, two of them
(`dialy→diary`, `Frence→France`) verbatim.

> **Conclusion.** The ceiling holds from a third independent direction, and
> this time with the sharpest instrument: a 94%-conditional chooser still only
> reaches 79% overall. **The Jev row is still not reproduced** — kev is a third
> party's reconstruction on different candidate lists — but the gap between
> kev-4b's 79% and Jev's ~91% is the gap between an 84%-coverage Hunspell pool
> and a 93%-coverage classic union, not a gap between choosers.

---

# Part II — what failed, and why

Three different kinds of failure, kept apart because they mean different
things. A run that never produced a number is not the same as a system that
produced a number and the number was bad.

## A. Dead ends — no result at all

| # | What | Why it failed |
|---|---|---|
| **Exp 3** | The 75% campaign (E0-E6) | **Defunded, not disproven.** Written and superseded the same day by the frozen-encoder pilot, which took the budget. Its largest dependency — a D-real set of ≥2,000 authentic contextual errors — was never priced and does not exist. |
| **Exp 4** | Frozen ModernBERT + selector head | **Two compounding bugs.** The 3854-d encoder features (including `c*t` and `\|c-t\|`) were fed in unnormalized at lr `1e-3` and overflowed — H2 set `nan_seen=true` after ~1 step. Then a bash status-capture bug turned a *recoverable* NaN into a fatal abort, so H3 never started. |
| **Exp 7** | gemma-4-E2B q4_0 on GPU + DSPy | **Abandoned in flight.** Started 2026-09-11 and still marked *running*. No pod id, no GPU type, no commit and no numbers were ever recorded. |

> **Conclusion.** None of these says anything about its hypothesis. Exp 4 in
> particular left the actual question — do fixed pretrained representations
> support candidate selection — completely unanswered; what it established was
> operational (normalize your features; don't let a status-capture bug promote
> a NaN to a fatal). Two of the three were killed by process, not by the idea.

## B. Negative results — systems that worked, and were worse than nothing

Each of these ran to completion and produced a trustworthy number. The number
was at or below the do-nothing baseline, which means **reranking with them is
worse than not reranking at all.**

| System | Overall | Baseline on same rows | Why it failed |
|---|---:|---:|---|
| **MiniCPM5-1B** (judge, index) | **39.0%** | 59.0% | Cannot follow the forced-choice format reliably at 1B. Also 4x slower than gemma — it lost on both axes at once. |
| **Qwen3.5-0.8B** (judge, index) | 60.0% | 59.0% | Zero lift. It reproduces Hunspell's ranking rather than improving on it — an 0.8B model prompted zero-shot has no signal the candidate generator did not already have. |
| **`kev-0.5b`** (chooser) | **50.0%** | 60.0% | 5 rescues against 15 breaks. |
| **`kev-0.6b`** (chooser) | **58.0%** | 60.0% | 11 rescues against 13 breaks. |

The kev failures have a clean mechanism. The wrong picks are not near-misses —
they are Hunspell's *word-split artifacts*:

```
anthing    gold=anything    picked=an-thing     list=[anting, anything, anteing, an thing, an-thing]
concentrat gold=concentrate picked=concent rat  list=[concentrate, concent rat, concent-rat, …]
perents    gold=parents     picked=pe rents     list=[repents, percents, parents, pe rents, …]
frends     gold=friends     picked=f rends      list=[fends, rends, friends, fiends, trends]
```

The pointer head scores what an option *means in context*, not how far it is
from the typo. Nothing in the architecture carries an edit-distance prior; it
has to come from the backbone's own sense of what is a word. At 0.5-0.6B there
is not enough backbone to notice that `concent rat` is not English. At 4B there
is — the same architecture, same prompt, same lists, jumps to 94% conditional.

> **Conclusion.** A model that is merely *present* does not help; below a
> capability threshold it actively destroys the generator's ranking. The
> threshold is real and it is somewhere between 0.6B and 4B for this task. Two
> separate lines of work (MiniCPM5 at 1B, kev at 0.6B) found it independently.
> **Always report the baseline on the same rows** — three of these four look
> respectable until you notice what they are being compared against.

## C. Missed targets — worked, aimed higher

| What | Target | Got | Why the gap |
|---|---:|---:|---|
| **Exp 2** (87M reranker) | 75% | 64.82% | The pool, not the model: `64.82% = 80.89% × 80.14%`. It was already converting 80% of what was reachable. |
| **Exp 8** (distillation) | >90% | 87.30% | The teacher's own ceiling was 88.75%. The student could not exceed what it was distilled from. |
| **M5** (0.8B QLoRA) | ≥ M4's 84% | 82% | A regression against the LoRA it replaced — QLoRA's quantized base cost ~2 points at 0.8B. |
| **Sentence-rewrite mode** | — | 73.0% strict | **Not a spelling failure.** 73 of 100 rewrites reflow BEA's pre-tokenised punctuation; 15 of the 27 strict errors are punctuation attachment on otherwise-correct corrections. Punctuation-insensitive it scores 88.0%. |
| **q4_0 as a free lunch** | parity with bf16 | −2 to −3 pp | Consistent in one direction, zero wins across 200 paired examples. The speedup is real; the "free" was not. |

> **Conclusion.** Three of these five are the same lesson in different clothes:
> **a system cannot beat the thing that feeds it.** A reranker cannot beat its
> candidate pool, a student cannot beat its teacher, and a strict scorer will
> punish you for an artifact of the benchmark's tokenisation rather than for
> being wrong. Diagnose the ceiling before spending on the model.

---

# What the whole arc says

1. **Reranking Hunspell is capped at ~81%, and we hit the cap three times.**
   Exp 2 converted 80.14% of what was reachable; gemma's index mode converted
   100% of it on a small sample; kev-4b converted 94%. Three independent
   systems, same wall.
2. **The two ways out both work.** Answer freely (gemma open, 90.0%) or train
   the model to emit the correction directly (M6, 91%). Neither needs a
   candidate generator at inference time.
3. **Dropping the candidate pipeline is nearly free and enormously fast.**
   M3 → M4: one point of accuracy for a 53x latency win.
4. **Small models can carry it — above a threshold.** A 2B teacher distills
   into an 0.8B student at a 1.45-point loss and serves on CPU sub-second. But
   below ~1B, prompted zero-shot, the same idea scores *below* the baseline.
5. **Most of these numbers are n=100.** A 95% CI near 40-80% is roughly ±8-10
   pp. Exactly one system — exp 2's 87M reranker, at 64.82% — has ever been run
   on all 68,429 errors, and it is not the one with the best headline.

## Known gaps

- **Two different "frozen 100" sets** are in play. Hunspell top-1 scores 59% on
  the exp 5/6 sample and 60% on the milestone sample, so the LLM-judge rows and
  the milestone rows are not strictly the same denominators.
- **The best headline (90.0%) and the only full-benchmark number (64.82%) are
  not comparable**, and nothing here has closed that gap: the trained reranker
  has never been given the small-sample treatment, and no LLM-judge or
  fine-tuned corrector has ever been run at full scale.
- **`kev-8b` was not run** (16GB bf16 does not fit the 15GB box).
- **Aspell-in-the-pool** (measured ceiling 87.7%) remains out of scope by
  decision.
- **The Luna ~94% row** has never been examined the way the Jev row now has.
