# Experiments and learnings

The goal here is to create a spelling suggestion engine that is accurate and cheap to run. 

I used the BEA-60K dataset which contains sentences with mistakes and their corrections. It is not great - some are incorrect, some are not typos but a totally diff word, so on. But was easy to use. Evaluation on a subset of 100, seed 1337. 

The rest of the doc contains various experiments.

TODO: MENTION EVERYWHERE THE COST AND IF THE PERF IS ON CPU VS GPU. 

# Part I — what worked

## 1. The baselines

| System | n | Overall | What it is |
|---|---:|---:|---|
| **Hunspell top-1** | 68,429 | **53.67%** | Take Hunspell's first suggestion. The do-nothing baseline. |
| **Aspell top-1** | 68,429 | **60.56%** | The external baseline the project set out to beat. |
| *Hunspell oracle@10* | 68,429 | *80.34%* | What a perfect chooser could reach given Hunspell's first ten. |
| *Hunspell oracle@16* | 68,429 | *80.89%* | Proves adding hunspell options does not help |

Follow-up idea: adding Aspell's suggestions to the pool lifts the ceiling to **87.7%** on a 6,000-error sample. Try models that only choose options with this set. 

> **Conclusion.** **Any** system that can only pick from Hunspell's list is capped at ~81%, however good the picker.

## 2. Trained a 27/87M model to choose the correct hunspell suggestion

| Experiment | n | Overall | Conditional |
|---|---:|---:|---|
| **Exp 1** — 28M byte reranker, 10 slots | not recorded | **62.56%** | 77.84% |
| **Exp 2** — 87M byte reranker, 16 slots | **68,429** | **64.82%** | 80.14% |
(conditional means among cases where Hunspell had the coreect proposal in Top 10)

Exp 1 met its goal: beat Aspell. 

Exp 2 (suggested by ASTRA) scaled the model 3x but failed to improve significantly. On the test data the small model had 75% accuracy. So, it was already overfitting. 

**Conclusion** We need more data to train a small LLM from scratch. Otherwise it memorizes. 

TODO: Cost & setup. 

## 3. Use an open-source LLM

| Model | Mode | Overall | Note |
|---|---|---:|---|
| **gemma-4-E2B-it** | index | 83.0% | 100% conditional — it never once picked wrong *when gold was on the list*. |
| **gemma-4-E2B-it** | **open** | **90.0%** | List shown as a hint only; free to answer any word.|
| **gemma-4-E2B-it** | beam | 88.0% | No Hunspell list at all, look at next token probabilities |
| **gemma-4-E2B-it** | sentence rewrite | 88.0% (73.0% strict) | Strict score wrecked by punctuation reflow, not spelling. |
| gemma-4-E2B q4_0 | open / index / sentence | 87.0% / 81.0% / 86.0% | q4_0 costs 2-3 points, consistently. |

(The two models that scored at or below the baseline are in
[Part II](#b-negative-results--systems-that-worked-and-were-worse-than-nothing).)

> **Conclusion.** Gemma's index mode scored
> **100% conditional**. I tried next to ignore Hunspell options: they put a cailing at 81% and increased the token count. Open mode pretty good.

TODO: gemma open accuracy and latency on CPU

## 4. External API models

| System | Acc@1 | What it is |
|---|---:|---|
| **Luna** freeform, first try | **~94%** | Large hosted model, asked to correct the word directly. |
| **Jev Choice** | **~91%** | TypeSafe's hosted decision model, used as a chooser over classic ~100-candidate lists (93% gold-in-list). |

## 5. Fine-tuning Qwen — picker, then direct corrector

All on the 100-sample subset, Acc@1 ignore case, where Hunspell top-1 = 60%.

| Milestone | What was trained | Acc@1 | Latency (3090) |
|---|---|---:|---:|
| **M3** | Qwen3.5-0.8B LoRA **picker** over candidate lists | **85%** (classic mix) / **87%** (choosing from a big list forced to include the correct option) | ~5.65 s/typo |
| **M4** | Qwen3.5-0.8B LoRA **direct corrector** — no candidates at all | **84%** | **0.107 s** |
| **M6** | **Qwen3.5-2B** QLoRA direct corrector | **91%** | 0.095 s |

Quantized to Q4_K_M for CPU serving via llama.cpp: M4 84% → 86%, M6 91% →
**87%** at 384 ms p50 on CPU.

TODO: Specify which train /evaluation was on CPU / GPU. Also the training costs. 

TODO: 0.8B studen accuracy and latency on CPU. Mention we deployed to vercel.

> **Conclusion.** **Dropping the candidate set entirely cost one point of
> accuracy and bought a 53x latency win.** A candidate generator turns out to
> be a liability at inference time, not an asset.

## 6. Distillation — 2B teacher into an 0.8B student

| | Acc@1 (n=2,000) | Acc@1 (frozen 100) |
|---|---:|---:|
| Teacher (2B Q4) | **88.75%** | 87% |
| **Student (0.8B, distilled)** | **87.30%** | **89%** |
| Student Q4_K_M on CPU | — | 88% |

Total cost: **$0.70**, including two failed runs and two broken-CUDA hosts.

TODO: explain why we dropped 90 to 88.75. 

So, a 0.8B params model at Q4 can choose the right spelling correction 87% of times. 

> **Conclusion.** The student recovers all but **1.45 points** of a model 2.5x
> its size and beats both previous 0.8B correctors. **Capability at this task
> compresses well** — the 2B teacher was not using its extra parameters for
> anything the 0.8B body cannot hold.

## 7. kev — an open-weights stand-in for Jev

TODO: MOVE AFTER JEV

[`jaredpalmer/kev`](https://github.com/jaredpalmer/kev) is an open-weights
reconstruction of Jev's architecture.

| System | Overall | Conditional | vs Hunspell top-1 |
|---|---:|---:|---:|
| Hunspell top-1 | 60.0% | — | — |
| **`kev-4b`** | **79.0%** | **94.0%** | **+19 pp (20 rescues, 1 break)** |
| *Hunspell coverage@8* | *84.0%* | — | *ceiling* |

TODO: Above we say hunspell caps at 81, not 84. 
TODO: kev is not evaluated on the same comined list on which jev was evaluated. do that. 

---

# Part II — what failed, and why

## A. Dead ends — no result at all

| # | What | Why it failed |
|---|---|---|
| **Exp 3** | The 75% campaign (E0-E6) | **Defunded, not disproven.** Written and superseded the same day by the frozen-encoder pilot, which took the budget. Its largest dependency — a D-real set of ≥2,000 authentic contextual errors — was never priced and does not exist. |
| **Exp 4** | Frozen ModernBERT + selector head | Did not manage to train. |
| **Exp 7** | gemma-4-E2B q4_0 on GPU + DSPy | **Abandoned in flight.** Took too long.  |

## B. Negative results — systems that worked, and were bad

| System | Overall | Baseline on same rows | Why it failed |
|---|---:|---:|---|
| **MiniCPM5-1B** (judge, index) | **39.0%** | 59.0% | Cannot follow the forced-choice format reliably at 1B. Also 4x slower than gemma — it lost on both axes at once. |
| **Qwen3.5-0.8B** (judge, index) | 60.0% | 59.0% | Zero lift. It reproduces Hunspell's ranking rather than improving on it — an 0.8B model prompted zero-shot has no signal the candidate generator did not already have. |
| **`kev-0.5b`** (chooser) | **50.0%** | 60.0% | 5 rescues against 15 breaks. |
| **`kev-0.6b`** (chooser) | **58.0%** | 60.0% | 11 rescues against 13 breaks. |

---

# What the whole arc says

1. **Reranking Hunspell is capped at ~81%, and we hit the cap three times.**
   Exp 2 converted 80.14% of what was reachable; gemma's index mode converted
   100% of it on a small sample; kev-4b converted 94%. Three independent
   systems, same wall.
2. **The two options that work.** Answer freely (gemma open, 90.0%) or train
   the model to emit the correction directly (M6, 91%). 
4. **Small models can learn better from teacher than from examples** A 2B teacher distills
   into an 0.8B student at a 1.45-point loss and serves on CPU sub-second.

