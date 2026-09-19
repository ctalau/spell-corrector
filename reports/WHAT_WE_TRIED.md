# Experiments and learnings

The goal here is to create a spelling suggestion engine that is accurate and cheap to run. 

I used the BEA-60K dataset which contains sentences with mistakes and their corrections. It is not great - some are incorrect, some are not typos but a totally diff word, so on. But was easy to use. Evaluation on a subset of 100, seed 1337. 

The rest of the doc contains various experiments.

## How to read the numbers

Two things decide whether a number in this doc means anything: **what hardware
it was measured on** and **how many examples it covers**. Both are stated in
every table below.

| Shorthand | What it is | Cost |
|---|---|---|
| **local CPU box** | 4 vCPU Intel Xeon @2.10GHz, 15GB RAM, no GPU | $0 |
| **3090** | Runpod Community RTX 3090 | $0.22/hr |
| **L40S** | Runpod Community L40S | $0.79/hr |
| **hosted API** | someone else's model behind an endpoint | not metered by us |

Accuracy and latency travel separately: accuracy is a property of the model and
quantization, latency is a property of the box. Where a number is a **frozen
100** result it carries roughly ±8-10 pp of binomial noise; n=2,000 carries
±1.5 pp; the only full-benchmark numbers (n=68,429) are Hunspell, Aspell and
Exp 2.

**Total GPU spend across the whole project: roughly $7.** Everything else ran
on the local CPU box for free.

# Part I — what worked

## 1. The baselines

Measured on the **local CPU box**, full benchmark, $0. Per-query latency was
never recorded for these — they are batch runs of a C library.

| System | n | Overall | What it is |
|---|---:|---:|---|
| **Hunspell top-1** | 68,429 | **53.67%** | Take Hunspell's first suggestion. The do-nothing baseline. |
| **Aspell top-1** | 68,429 | **60.56%** | The external baseline the project set out to beat. |
| *Hunspell oracle@10* | 68,429 | *80.34%* | What a perfect chooser could reach given Hunspell's first ten. |
| *Hunspell oracle@16* | 68,429 | *80.89%* | Proves adding hunspell options does not help |

Follow-up idea: adding Aspell's suggestions to the pool lifts the ceiling to **87.7%** on a 6,000-error sample. Try models that only choose options with this set. 

> **Conclusion.** **Any** system that can only pick from Hunspell's list is capped at ~81%, however good the picker.

## 2. Trained a 27/87M model to choose the correct hunspell suggestion

| Experiment | n | Overall | Conditional | Trained on | Cost |
|---|---:|---:|---:|---|---:|
| **Exp 1** — 28M byte reranker, 10 slots | not recorded | **62.56%** | 77.84% | GPU pod, details not recorded | not recorded |
| **Exp 2** — 87M byte reranker, 16 slots | **68,429** | **64.82%** | 80.14% | **L40S**, 3.74 pod-hours | **$2.95** |

(conditional means among cases where Hunspell had the coreect proposal in Top 10)

Exp 1 met its goal: beat Aspell. 

Exp 2 (suggested by ASTRA) scaled the model 3x but failed to improve significantly. On the test data the small model had 75% accuracy. So, it was already overfitting. 

**Conclusion** We need more data to train a small LLM from scratch. Otherwise it memorizes. 

**Cost & setup.** Exp 2 is the only model ever trained *and* benchmarked at full
scale: pod `xwvjd2w980kk00`, Community L40S at $0.79/hr, 3.74 pod-hours to DONE
= **$2.95**, plus roughly **$2 more burned across the eight preceding failed
pods** (bootstrap and OOM deaths, catalogued in the experiment's RUN_NOTES). So
~$5 all-in for the one full-benchmark number in this repository. Training and
the 68,429-error benchmark both ran on the pod's GPU. **Inference latency was
never recorded for either byte reranker**, on CPU or GPU — the one number this
section is missing. Exp 1's cost and sample size were not recorded at all.

## 3. Use an open-source LLM

Everything in this section ran on the **local CPU box** — no GPU, no training,
$0. `gemma-4-E2B-it` is ~10.2GB in bf16 (despite the "E2B" name) and fits in
15GB RAM with little headroom; the q4_0 GGUF is served through `llama.cpp` at
`-t 4`.

| Model | Mode | Overall | p50 latency (CPU) | Note |
|---|---|---:|---:|---|
| **gemma-4-E2B-it** | index | 83.0% | 632 ms | 100% conditional — it never once picked wrong *when gold was on the list*. |
| **gemma-4-E2B-it** | **open** | **90.0%** | (unusable — see below) | List shown as a hint only; free to answer any word.|
| **gemma-4-E2B-it** | beam | 88.0% | 14,786 ms | No Hunspell list at all, look at next token probabilities |
| **gemma-4-E2B-it** | sentence rewrite | 88.0% (73.0% strict) | 5,980 ms | Strict score wrecked by punctuation reflow, not spelling. |
| gemma-4-E2B q4_0 | open / index / sentence | 87.0% / 81.0% / 86.0% | 960 / 744 / 2,118 ms | q4_0 costs 2-3 points, consistently. |

(The two models that scored at or below the baseline are in
[Part II](#b-negative-results--systems-that-worked-and-were-bad).)

> **Conclusion.** Gemma's index mode scored
> **100% conditional**. I tried next to ignore Hunspell options: they put a cailing at 81% and increased the token count. Open mode pretty good.

**Open mode's accuracy and latency on CPU.** Accuracy is **90.0%** (n=100) —
the best number in the repository. Latency for that same run is **not usable**:
its p50 was 10.2 s, but the box was paging the 10.2GB bf16 checkpoint off disk
at 5-6MB/s throughout (confirmed live via `/proc/<pid>/io`), and the beam-mode
run on a warm page cache did strictly more work per query. The honest CPU
latency for open mode is the **q4_0 figure: 960 ms p50** (p99 1,288 ms), at
87.0% — i.e. the deployable version of open mode is sub-second on 4 vCPUs and
costs 3 points. Wall clock for a warm-cache 100-example q4_0 run is ~4 minutes;
the bf16 sentence-mode run took 693 s for the same 100.

## 4. External API models

Hosted, closed, not metered by us; neither was run on our hardware, so there is
no CPU/GPU number to give. Both come from the milestone-4 comparison table.

| System | Acc@1 | What it is |
|---|---:|---|
| **Luna** freeform, first try | **~94%** | Large hosted model, asked to correct the word directly. |
| **Jev Choice** | **~91%** | TypeSafe's hosted decision model, used as a chooser over classic ~100-candidate lists (93% gold-in-list). |

## 5. kev — an open-weights stand-in for Jev

[`jaredpalmer/kev`](https://github.com/jaredpalmer/kev) is an open-weights
reconstruction of Jev's architecture, placed here directly after the Jev row it
exists to bracket.

Frozen 100, **local CPU box**, **$0** — no pod was rented. bf16 for 4b, fp32 for
the small checkpoints.

| System | Overall | Conditional | vs Hunspell top-1 | p50 latency (CPU) |
|---|---:|---:|---:|---:|
| Hunspell top-1 | 60.0% | — | — | — |
| **`kev-4b`** | **79.0%** | **94.0%** | **+19 pp (20 rescues, 1 break)** | 5,623 ms |
| `kev-0.6b` | 58.0% | 69.0% | −2 pp | 320 ms |
| `kev-0.5b` | 50.0% | 59.5% | −10 pp | 252 ms |
| *Hunspell coverage@8 on these 100 items* | *84.0%* | — | *ceiling* | — |

5.6 s is a 4B backbone prefilling on 4 vCPUs, not a product number; on a GPU
this is one forward pass over ~150 tokens.

**The two ceilings are not in conflict.** Section 1's ~81% is Hunspell
*oracle@16* over the full 68,429 errors (80.89%). The 84.0% here is
*coverage@8* over the **frozen 100** — a different, 100-item sample carrying
±8-10 pp. Uncapping the list does not move it: gold is in the pool for 60% of
these items at rank 1, 81% by rank 4, 84% by rank 8, and 84% however far you go
(the longest pool this set produces is 15 words). Both numbers say the same
thing — a chooser restricted to Hunspell tops out around 80-84% — and the
full-benchmark 81% is the one to quote.

**Still unrun: kev on the same combined list Jev was scored over.** Jev's ~91%
was measured against classic ~100-candidate unions with 93% gold-in-list; kev
was measured against Hunspell alone, which saturates at 84% coverage on this
set. That is why the 12-point gap between them is **candidate recall, not
chooser quality** — on conditional accuracy, kev-4b's 94.0% is in the
neighbourhood of the ~98% Jev's 91%/93% implies. Closing this properly needs
the BM25 ∪ dense union that this repository does not own, so the comparison
stays open, and nothing here licenses editing Jev's 91% or claiming it was
confirmed or refuted. (`kev-8b` is also unrun: 16GB in bf16 does not fit the
15GB box, and a pod was not judged worth it.)

## 6. Fine-tuning Qwen — picker, then direct corrector

All on the 100-sample subset, Acc@1 ignore case, where Hunspell top-1 = 60%.
**All three were trained on a Runpod Community RTX 3090 at $0.22/hr**, and the
Acc@1 column is a **GPU** (NF4/bf16 + adapters) evaluation on that same pod.

| Milestone | What was trained | Acc@1 (GPU) | Latency (3090) | Train wall | Pod cost |
|---|---|---:|---:|---:|---:|
| **M3** | Qwen3.5-0.8B LoRA **picker** over candidate lists | **85%** (classic mix) / **87%** (choosing from a big list forced to include the correct option) | ~5.65 s/typo | ~1.9 h | ~$0.55-0.70 |
| **M4** | Qwen3.5-0.8B LoRA **direct corrector** — no candidates at all | **84%** | **0.107 s** | ~1.27 h | ~$0.33-0.40 |
| **M6** | **Qwen3.5-2B** QLoRA direct corrector | **91%** | 0.095 s | 77.1 min | ~$0.48-0.55 |

**~$1.40-1.65 of GPU for all three milestones**, pods terminated after artifact
sync each time.

Quantized to Q4_K_M and re-scored on the **local CPU box** via `llama.cpp`
(`-t 8`), on the same frozen 100: M4 84% (GPU) → **86%** at p50 **~0.22 s**;
M6 91% (GPU) → **87%** at p50 **384 ms** (~3.1 GiB RSS, ~1.2 GiB on disk).
M3's picker was the one that did *not* survive the move: ~7.6 s per typo on
CPU at 20 candidates and ~39 s at 100, which is what killed the picker approach
as a serving path.

> **Conclusion.** **Dropping the candidate set entirely cost one point of
> accuracy and bought a 53x latency win.** A candidate generator turns out to
> be a liability at inference time, not an asset.

## 7. Distillation — 2B teacher into an 0.8B student

| | Acc@1 (n=2,000) | Acc@1 (frozen 100) | Hardware | p50 |
|---|---:|---:|---|---:|
| Teacher (2B Q4) | **88.75%** | 87% | 3090, NF4 | 0.318 s |
| **Student (0.8B, distilled)** | **87.30%** | **89%** | 3090, NF4 | 0.310 s |
| Student Q4_K_M on CPU | 86.60% | 88% | local CPU box, `-t 4` | **0.597 s** |

Training: 6,024 steps / 3 epochs, **96.5 minutes on a community RTX 3090 at
$0.22/hr**, peak VRAM 12.2 GiB. Total cost: **$0.70**, including two failed runs
and two broken-CUDA hosts. The CPU rows used `-t 4` where M4-M6 used `-t 8`, so
that 0.597 s is not comparable to section 6's CPU latencies — the accuracy is.

**The 0.8B student on CPU, and where it runs.** Q4_K_M, 505 MiB on disk,
**86.60% casefold at n=2,000 and 88.0% on the frozen 100, p50 ~0.60 s** on 4
vCPUs. Quantization costs 0.7 points against the same student in NF4 on GPU.
That artifact is what we **deployed to Vercel**: `api/correct.js` serves the
GGUF through `node-llama-cpp` (prebuilt native bindings, so no glibc/ABI
mismatch with the Lambda runtime), 2048MB of function memory, the model bundled
via `includeFiles` and loaded once per cold start as a module-level singleton.
One deployment-specific trap is recorded in that file: the prompt must be built
with `SpecialTokensText`, because a plain JS string tokenizes the chat
delimiters as literal text and costs ~6 accuracy points (86.0% → 80.7% on a
300-example slice of the test set).

**Why the target moved from 90% to 88.75%.** The 90-91% figure came from M6's
write-up, and it is a **100-row number** — ±6 pp. When the same 2B adapter was
re-scored here on 2,000 held-out rows by the same harness that scored the
student, it measured **88.75%**. A student cannot systematically exceed the
signal it is trained on, so 90% was never reachable from this teacher; the
target was set against noise.

Worse, on the **identical frozen 100**, the same adapter scores **87.0%** here
against M6's 91%. Greedy decoding is deterministic, so that is not sampling
noise: the raw generations differ in **93 of 100 rows** while the verdict flips
on only 4 — the two harnesses decode differently throughout. Left-padded batched
generation was the natural suspect (Qwen3.5 runs linear attention with
recurrent state in 18 of its 24 layers, and such layers do not always respect an
attention mask) and the Q4 run **refuted** it: sequential Q4_K_M scored *below*
batched NF4, by about the quantization tax, when the hypothesis predicts well
above. The remaining candidate is the stack (transformers 5.18.dev +
bitsandbytes 0.50+ here, against whatever M6 ran), which would mean M6's 91% is
not reproducible on a current stack. Worth an hour before any 91% is quoted
again.

So, a 0.8B params model at Q4 can choose the right spelling correction ~87% of times, sub-second on 4 vCPUs. 

> **Conclusion.** The student recovers all but **1.45 points** of a model 2.5x
> its size and beats both previous 0.8B correctors. **Capability at this task
> compresses well** — the 2B teacher was not using its extra parameters for
> anything the 0.8B body cannot hold.

---

# Part II — what failed, and why

## A. Dead ends — no result at all

| # | What | Why it failed | Cost |
|---|---|---|---:|
| **Exp 3** | The 75% campaign (E0-E6) | **Defunded, not disproven.** Written and superseded the same day by the frozen-encoder pilot, which took the budget. Its largest dependency — a D-real set of ≥2,000 authentic contextual errors — was never priced and does not exist. | $0 — never ran |
| **Exp 4** | Frozen ModernBERT + selector head | Did not manage to train. Died on an H2 NaN with its central question unanswered. | GPU pod, not recorded |
| **Exp 7** | gemma-4-E2B q4_0 on GPU + DSPy | **Abandoned in flight.** Took too long. | GPU pod, not recorded |

## B. Negative results — systems that worked, and were bad

All four ran on the **local CPU box** at $0.

| System | Overall | Baseline on same rows | p50 latency (CPU) | Why it failed |
|---|---:|---:|---:|---|
| **MiniCPM5-1B** (judge, index) | **39.0%** | 59.0% | 2,183 ms | Cannot follow the forced-choice format reliably at 1B. Also 4x slower than gemma — it lost on both axes at once. |
| **Qwen3.5-0.8B** (judge, index) | 60.0% | 59.0% | 541 ms | Zero lift. It reproduces Hunspell's ranking rather than improving on it — an 0.8B model prompted zero-shot has no signal the candidate generator did not already have. |
| **`kev-0.5b`** (chooser) | **50.0%** | 60.0% | 252 ms | 5 rescues against 15 breaks. Picks Hunspell's word-split artifacts (`an-thing`, `concent rat`) over real words. |
| **`kev-0.6b`** (chooser) | **58.0%** | 60.0% | 320 ms | 11 rescues against 13 breaks. Same failure mode. |

---

# What the whole arc says

1. **Reranking Hunspell is capped at ~81%, and we hit the cap three times.**
   Exp 2 converted 80.14% of what was reachable; gemma's index mode converted
   100% of it on a small sample; kev-4b converted 94%. Three independent
   systems, same wall.
2. **The two options that work.** Answer freely (gemma open, 90.0%) or train
   the model to emit the correction directly (M6, 91%). 
3. **Small models can learn better from teacher than from examples.** A 2B teacher distills
   into an 0.8B student at a 1.45-point loss and serves on CPU sub-second.
4. **The whole thing cost about $7 of GPU.** Exp 2 is ~$5 of that; the four
   fine-tuning and distillation milestones together are ~$2.10. Every LLM-judge
   and kev result was free, on 4 vCPUs.
5. **Beware 100-row numbers.** M6's 91% became 88.75% at n=2,000 under the same
   harness. Nothing in Part I except Hunspell, Aspell and Exp 2 has been
   measured against the 68,429-error denominator.
