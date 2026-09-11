# Experiment 6 — LLM judge on CPU: four answer modes, then q4_0 + llama.cpp

| | |
|---|---|
| **Status** | **completed** (local CPU box only; no pod was used and no bootstrap script exists for this track) |
| **When** | Date not recorded. Local box: 4 vCPU Intel Xeon @2.10GHz, 15GB RAM, no GPU. |
| **Headline result** | Best measured system in the repository: `google/gemma-4-E2B-it` bf16, **open mode**, **90.0% overall** on a fixed **n=100** BEA-60K sample. Beam mode 88.0%, sentence mode 88.0% punctuation-insensitive (73.0% strict), index mode 83.0%. On the q4_0 GGUF via llama.cpp: open 87.0%, index 81.0%, sentence 86.0% punctuation-insensitive at a p50 of 2,118ms. |
| **Cost** | $0 in GPU rental. Costs are wall clock: a warm-cache 100-example q4_0 run is ~4 minutes; the bf16 sentence-mode run took 693s for 100 examples. |
| **What it settled** | (1) The forced-choice format is the binding constraint, not the model — 17 of the 100 sampled errors have no gold candidate anywhere in Hunspell's list, and letting the model answer freely recovers 58.8% (open) to 64.7% (beam) of them. (2) **q4_0 is 2-3 points worse than bf16, consistently in one direction** — zero wins across 200 index/open examples — in exchange for a trustworthy 2.8x speedup in sentence mode. (3) Sentence mode's strict score is dominated by a single artifact: 73 of 100 rewrites reflow BEA's pre-tokenised punctuation, and 15 of its 27 strict errors are punctuation attachment on otherwise-correct corrections. (4) The Hunspell pre-pass memo turns an 822s per-invocation cost into 0.3s. |
| **What it left open** | Everything is n=100 on one fixed sample; ±8-10 pp binomial noise. The two recommended prompt fixes (an explicit `<corrected_word>` field, or marking the correction inside the rewrite) are specified and unrun. The prompt-lookup speedup is validated on a 12-example micro-benchmark only — the full 100-example rerun was interrupted. Beam mode and prompt-lookup are not portable to the llama.cpp backend. |
| **Artifacts** | [`reports/llm_judge_cpu/`](../../llm_judge_cpu/) — one directory per model/mode (`gemma-4-e2b`, `-open`, `-beam`, `-sentence`, `-q4-index`, `-q4-open`, `-q4-sentence`, `qwen3.5-0.8b`, `minicpm5-1b`), each with `results.json`, `predictions_sample100.jsonl` and latency histograms. |
| **Predecessor** | [Experiment 5](../05-llm-judge-index/README.md) — the index-mode-only harness these three models were first measured with. The three shared models' `results.json` files are identical between the two directories. |

> The bug post-mortems in this document (the beam-mode edit-distance scoring flaw,
> the `llama-server` blocked-pipe stall, gemma-4's default thinking mode returning
> empty `content`) are kept in full below. So is the finding that **MiniCPM5-1B
> scores below Hunspell top-1 alone** — i.e. reranking with it is worse than not
> reranking. One wording caveat: the text calls the trained baseline "the trained
> ~28M byte-level reranker", but the numbers it compares against are
> [experiment 2's 87M model](../02-byte-reranker-87m/README.md).

---

A separate track from the trained ~28M byte-level reranker (`reports/experiments/02-byte-reranker-87m/README.md`):
instead of a purpose-trained model, a general-purpose small instruction-tuned LLM is
prompted zero-shot with Hunspell's numbered suggestion list and asked to pick the
best one. Code: `spelling_reranker/llm_judge_cpu.py`, `scripts/llm_judge_bea60k_cpu.py`.

## Setup

- **Benchmark**: BEA-60K (locked, never trained/tuned on) via `scripts/download_bea60k.py`.
  68,429 word-level errors; Hunspell flags 67,877 of them (99.19%) and returns at
  least one suggestion for 67,844 -- that eligible set is what both models were
  sampled from.
- **Candidates shown to the model**: up to 8 Hunspell suggestions, in Hunspell's own
  order (`--max-candidates 8`).
- **Prompt**: sentence with the typo marked `<TYPO>...</TYPO>`, a numbered candidate
  list, instruction to reply with only the candidate number. Greedy decoding,
  `max_new_tokens=8`, thinking mode disabled where the chat template supports it.
- **Sampling**: seed 1337, same shuffle of the eligible set for both models, so they
  are scored on directly comparable examples.
  1. **Fixed sample**: 100 examples -- accuracy + latency.
  2. **Timed run**: as many examples as fit in a 5-minute wall-clock budget --
     throughput/latency at scale.
- **Models**: `Qwen/Qwen3.5-0.8B`, `google/gemma-4-E2B-it`, and `openbmb/MiniCPM5-1B`.
- **Environment -- CPU, not GPU.** `RUNPOD_KEY` was not available in this session, so
  this ran on the local box instead: 4 vCPU (Intel Xeon @2.10GHz), 15GB RAM, no GPU,
  `torch==2.14.0+cpu`, `transformers==5.17.0`. Qwen and gemma loaded via the
  `AutoModelForMultimodalLM` class (both are multimodal releases; text-only prompts
  were used here); MiniCPM5-1B is a plain `LlamaForCausalLM` and loaded via
  `AutoModelForCausalLM`. All three in bfloat16 with `low_cpu_mem_usage=True`.
  Checkpoint sizes: Qwen ~1.7GB, MiniCPM5-1B ~2.2GB, gemma-4-E2B-it **~10.2GB** in
  bf16 despite the "E2B" (effective-2B) name -- it fit in 15GB RAM without swapping,
  but left little headroom. Qwen's log also reports that `causal_conv1d` and
  `flash-linear-attention` (its hybrid linear-attention layers' optimized kernels)
  are not installed, so it ran on the slower reference PyTorch path; those kernels
  are CUDA-oriented, so this is expected on CPU and is not a bug in the harness.
  MiniCPM5-1B ships a hybrid Think/No-Think chat template; `enable_thinking=False`
  was passed (supported by its template) to keep it in fast/direct-answer mode,
  consistent with the other two models.
- **Latency measurement**: wall-clock around each `model.generate()` call only
  (excludes the one-time model load and a discarded warmup call). Single request at a
  time (batch size 1), to measure realistic per-query latency rather than
  batched throughput.

## Results

| System | Overall accuracy | Conditional accuracy&#42; | Hunspell top-1 (same sample) | p50 latency | p99 latency |
|---|---|---|---|---|---|
| Hunspell top-1 (full BEA-60K, `reports/bea60k/results.json`) | 53.7% | -- | -- | -- | -- |
| Qwen3.5-0.8B, 100-sample | **60.0%** | 72.3% | 59.0% | 541ms | 823ms |
| gemma-4-E2B-it, 100-sample | **83.0%** | 100.0% | 59.0% | 632ms | 825ms |
| MiniCPM5-1B, 100-sample | **39.0%** | 47.0% | 59.0% | 2183ms | 2772ms |
| Qwen3.5-0.8B, timed 5min (n=529) | **56.5%** | 68.6% | 56.1% | 530ms | 823ms |
| gemma-4-E2B-it, timed 5min (n=486) | **78.8%** | 95.3% | 56.0% | 605ms | 830ms |
| MiniCPM5-1B, timed 5min (n=139) | **37.4%** | 45.6% | 56.8% | 2146ms | 2787ms |

&#42; Conditional accuracy = accuracy among examples where the gold correction was
actually one of the candidates shown (83/100 on the fixed sample for all three
models, since all three were shown the same examples).

Full machine-readable results:
`reports/llm_judge_cpu/{qwen3.5-0.8b,gemma-4-e2b,minicpm5-1b}/results.json`,
per-example predictions in `predictions_sample100.jsonl` / `predictions_timed.jsonl`,
and latency histograms (JSON/CSV/PNG) alongside them.

### Latency histograms (100-example fixed sample, ms)

| Bin | Qwen3.5-0.8B | gemma-4-E2B-it | MiniCPM5-1B |
|---|---|---|---|
| 400-600ms | 62 | 32 | -- |
| 600-800ms | 33 | 66 | -- |
| 800-1000ms | 5 | 2 | -- |
| 1500-2000ms | -- | -- | 9 |
| 2000-3000ms | -- | -- | 90 |
| 3000-5000ms | -- | -- | 1 |

Qwen and gemma land in a tight, unimodal 400-1000ms band per call; MiniCPM5-1B is a
full order of magnitude to the right, in an equally tight 1.5-3s band -- there is no
overlap between the two groups at all on this box. Both are consistent with
single-request, short-generation (≤8 tokens) latency rather than throughput-oriented
batched serving. See the `*_latency_histogram.png` files for the full-resolution
charts (fixed-sample and timed-run separately).

## Reading the results

- **gemma-4-E2B-it and Qwen3.5-0.8B beat Hunspell top-1 and the trained 28M
  reranker's Hunspell-top-1 baseline outright**, and gemma-4-E2B-it's *conditional*
  accuracy (100% / 95.3%) is higher than the trained reranker's conditional accuracy
  (80.1%, from `reports/bea60k/results.json`) on a much smaller sample -- worth
  treating as a promising signal rather than a settled comparison, given n=100-529
  vs the trained reranker's full-benchmark n=68,429.
- **MiniCPM5-1B is a clear outlier, in the wrong direction.** Its accuracy (39.0% /
  37.4%) sits *below* Hunspell top-1 alone (59.0% / 56.8% on the same samples) --
  i.e. on this task, in this configuration, letting MiniCPM5-1B rerank actively hurts
  versus just taking Hunspell's rank-0 suggestion. Spot-checking
  `predictions_sample100.jsonl` rules out a parsing bug (it returns varied,
  in-range candidate numbers, not a stuck default); this looks like a genuine
  capability gap for this specific model/prompt/task combination, not a harness
  issue. It is also ~4x slower per call (p50 2183ms vs 530-630ms) despite being
  closer in nominal parameter count to Qwen3.5-0.8B than to gemma-4-E2B-it.
- **gemma-4-E2B-it noticeably outperforms both smaller models on accuracy** (83.0%
  vs 60.0% vs 39.0% overall; 100% vs 72.3% vs 47.0% conditional) while its per-call
  latency is only modestly higher than Qwen's (~90-100ms more at p50) and lower than
  MiniCPM5-1B's. Given gemma-4-E2B-it's much larger checkpoint (~10.2GB vs ~1.7-2.2GB
  bf16), the accuracy edge over Qwen is not a surprising trade, but the latency gap
  is smaller than the parameter-count gap would suggest -- consistent with "E2B"
  effective-compute design intent. MiniCPM5-1B, the intermediate-sized model,
  landing both least accurate *and* slowest is the most interesting result here.
- **Caveat -- this is a CPU run, not the GPU run originally planned.** `RUNPOD_KEY`
  was unavailable in this session; latency numbers here reflect a 4-vCPU box with no
  GPU and (for Qwen) missing optimized kernels for its hybrid attention layers.
  Absolute latency on a GPU pod would very likely be substantially lower for all
  three models, and their relative latency ranking could plausibly differ once
  GPU-optimized kernels are available (Qwen's reference-path fallback in particular
  is a known slowdown that a GPU run would remove; MiniCPM5-1B's slowness has not
  been root-caused and could equally be CPU-specific or could persist on GPU).
  Accuracy numbers should be unaffected by the compute backend (greedy decoding is
  deterministic modulo floating-point non-associativity).
- **Sample sizes are small.** 100 (fixed) and 139-529 (timed) examples are enough to
  see clear accuracy gaps between these three models, but not enough to treat these
  percentages as precise; a 95% CI on a 100-sample binomial proportion near 40-80%
  is roughly ±8-10pp.

## Follow-up: how much does the forced-choice format itself cost?

gemma-4-E2B-it's index-mode result above (83.0% overall / 100% conditional) is
capped by construction: 17 of the 100 sampled errors never had the gold correction
anywhere in Hunspell's candidate list, so no forced choice among those candidates
could ever get them right. Two follow-up modes test the same fixed 100-sample
subset with that constraint loosened or removed entirely:

- **`--answer-mode open`** (`build_open_messages`/`parse_open_word`): same Hunspell
  candidates shown as a hint, but the model may write any word instead of being
  restricted to picking one.
- **`--answer-mode beam`** (`build_generative_messages`/`beam_word_candidates`/
  `select_by_edit_distance_and_probability`): no Hunspell candidates shown at all.
  The model's own beam search (width 3) generates up to 3 candidate words; each is
  cut at its first word boundary and scored `logprob - weight * edit_distance(word,
  typo)` (weight 1.0, so edit distance is a prior *weighted against* the model's own
  probability, never an absolute decider on its own), and the top-scoring survivor
  is the answer. A candidate identical to the typo itself is dropped before scoring
  -- see the flaw-and-fix note below for why. This replaces Hunspell as the
  candidate generator entirely, using only the LLM plus a classic noisy-channel-style
  edit-distance prior.

| | Index (pick from list) | Open (list as hint, free answer) | Beam (no list, self-generated + edit-distance rerank) |
|---|---|---|---|
| Overall accuracy | 83.0% | **90.0%** | 88.0% |
| Conditional accuracy (gold was offered, 83/100) | 100% | 96.4% (80/83) | 92.8% (77/83) |
| Accuracy when gold was *not* offered (17/100) | 0% (impossible by construction) | 58.8% (10/17) | **64.7% (11/17)** |

Both follow-ups beat index mode overall by removing its hard ceiling. Open mode
wins on total accuracy (90.0%), but beam mode -- despite getting *no* Hunspell hint
at all -- recovers the most of the previously-unreachable cases (64.7% vs 58.8%),
at the cost of being weaker on the "easy" gold-in-pool subset (92.8% vs 96.4%,
unsurprising since it never sees Hunspell's list to fall back on). A characteristic
open-mode recovery: typo "thursty" in "I had the worst thursty I have ever had" --
Hunspell's only candidates were "thirsty" and "hurst" (both wrong; the context
calls for the noun "thirst", not the adjective "thirsty"), and gemma-4-E2B-it
produced "thirst" directly from context despite it never appearing in the
candidate list.

**A genuine flaw, found and fixed, not glossed over.** The first beam-mode run
(87.0% overall, 91.6% conditional) had 2 of its 13 wrong answers echoing the typo
completely unchanged ("ugry" -> "ugry" instead of "ugly"; "Miken" -> "Miken"
instead of "McCain"). The scoring formula was structurally responsible:
`edit_distance(word, typo)` is 0 when a beam candidate equals the typo itself, so
the formula rewarded *not correcting at all* whenever the logprob gap to a real
correction was small (for "ugry": logprob -0.73 for the echoed typo vs -0.70 for
"ugly" -- nearly tied on probability, but the +1 edit distance was enough to flip
it). Fixed in `select_by_edit_distance_and_probability` by dropping any candidate
identical to the typo (case-insensitive) before scoring, rather than just leaving
it to compete on edit distance -- echoing the typo is not a correction, so it
should never be a candidate, not merely a disadvantaged one. Rerunning with the
fix: overall accuracy 87.0% -> **88.0%**, conditional 91.6% -> **92.8%**. "ugry" now
resolves to "ugly" cleanly (0 wrong answers echo the typo, down from 2). "Miken"
still resolves incorrectly (now to "Mike" rather than the typo itself) -- correctly
so, since "McCain" is too many edits away from "Miken" to recover with this
method; that one was never the scoring bug's fault. Full predictions with all
surviving beam candidates and their scores per example:
`reports/llm_judge_cpu/{gemma-4-e2b-open,gemma-4-e2b-beam}/predictions_sample100.jsonl`.

**Latency is not comparable across these three rows and is deliberately left out
of the table.** The open-mode run happened to hit a period of unusually slow disk
I/O on this box (`low_cpu_mem_usage=True` loads the already-bf16 checkpoint via
mmap, so the *first* touch of each weight page during a forward pass faults it in
from disk rather than RAM; confirmed live via `/proc/<pid>/io` at only ~5-6MB/s
sustained), driving its median latency to 10.2s -- an I/O artifact, not a property
of open-mode generation. The beam-mode run's page cache was warm (0 disk reads
observed via the same `/proc/<pid>/io` check), so its 14.8s median is a real, if
expensive, measurement of beam-width-3 generation cost on this CPU box -- roughly
23x the index-mode run's 632ms median, more than the naive 3x-the-beams estimate
would suggest, likely from Gemma4's hybrid sliding/full-attention layers and
shared-KV bookkeeping not being especially optimized for batched beam search on
CPU. Neither number should be read as "mode X is N times slower than mode Y" in
general; both are one-off measurements on a noisy shared box.

## Follow-up: rewrite the whole sentence instead of answering with an index

A fourth answer mode (`--answer-mode sentence`, `build_sentence_messages` /
`parse_corrected_sentence` / `extract_corrected_word`) keeps everything index mode
does -- the same fixed 100-example sample, the same Hunspell top-8 candidate list in
the same order, greedy decoding, no beam search -- and changes only the answer
format: instead of replying with a candidate *number*, the model replies with the
whole sentence, typo replaced by the corrected word, wrapped in
`<corrected_sentence></corrected_sentence>`. The correction is then recovered by
aligning the rewritten sentence against the original on whitespace tokens (the same
tokenisation BEA-60K's own error extraction uses), so the scored unit stays a single
word and the number stays comparable with the other three modes.

Two harness details this mode forced:

- **Per-example generation budget.** The other modes generate ≤8 tokens; a whole
  sentence needs far more. The budget is sized per example as the sentence's own
  token count + 32, capped at 320 (mean 53 tokens on this sample).
- **Deletions, multi-token spans and missing tags are recorded, not swallowed.**
  `replacement_span_tokens` says how many output tokens landed in the typo's slot
  (1 = clean replacement, 0 = the model deleted the word, >1 = the anchors on one
  side did not survive because the model also reworded/repunctuated its neighbours).

### Results (same 100-example sample, gemma-4-E2B-it, this box)

| | Index (candidate number) | Sentence (whole-sentence rewrite) |
|---|---|---|
| Overall accuracy, strict | 83.0% | **73.0%** |
| Overall accuracy, ignoring surrounding punctuation | 83.0% | **88.0%** |
| Overall accuracy, ignoring punctuation *and* case | 84.0% | **90.0%** |
| Conditional accuracy (gold was offered, 83/100) | 100% | 83.1% |
| Accuracy when gold was *not* offered (17/100) | 0% (impossible by construction) | 23.5% |
| Format failures (no `<corrected_sentence>` tag) | -- | 0/100 |
| p50 latency | 783ms | 5,980ms |
| p99 latency | 901ms | 16,280ms |
| Wall clock for the 100 examples | 78s | 693s |

The index-mode column is a **rerun on this box**, not the earlier table's numbers, so
the latency comparison is like-for-like on the same hardware and the same day. It
reproduced the original run's 83.0%/100% accuracy exactly.

### Where the 27 strict errors come from

The strict number is misleading on its own, and the reason is a single, systematic
artifact rather than 27 independent judgement failures:

- **15 of 27 are punctuation attachment.** BEA-60K text is pre-tokenised
  (`Now I must buy it on the interenet .`, `I do n't have a clue about anthing .`).
  Told to rewrite the sentence, gemma-4-E2B-it writes *natural* English and
  re-attaches the punctuation: `... on the internet.` The recovered token is then
  `internet.` where gold is `internet`. The correction itself is right in every one
  of these 15 cases. **73 of the 100 rewritten sentences come back reflowed this way**
  despite the prompt explicitly saying to keep every other word, its spelling,
  capitalisation and punctuation exactly as given -- this is not an occasional slip,
  it is the model's default behaviour on pre-tokenised input.
- **2 of 27 are case normalisation**: `englsh` -> `english` (gold `English`),
  `Goog` -> `good` (gold `Good`, and the rewrite mangled the opening into
  `Go good summer vacations !`). Same root cause: the model is rewriting prose, not
  performing a constrained substitution.
- **10 of 27 are genuine word choices**, and most are cases the other modes miss too:
  `commonder` -> `commoner` (index mode got `commander` -- the one real regression
  besides `Goog`), `Miken` -> `Mike` (gold `McCain`, unreachable), `thursty` ->
  `thirsty` (gold `thirst`; index also wrong, open mode got it), `Thanx` -> `Thank`
  (gold `Thanks`), `pollusions` -> `pollutions` (gold `pollutants`), `catacumbas` ->
  `catacombs` (gold `catacomb`), `vacab` -> `vocab` (gold `vocabulary`), `dialoging`
  echoed unchanged (gold `talking` -- a lexical rewrite, not a spelling fix). Two of
  the ten are **BEA gold noise**, where the model's answer is arguably better than
  the reference: `crimbimg` -> `climbing` (gold is `climbimg`, itself misspelled) and
  `studiant` -> `student` (gold is `studant`).

Comparing index-right/sentence-wrong pairs directly: 14 examples flipped from right
to wrong, and **12 of those 14 are punctuation-only**. Only `commonder` and `Goog`
are real degradations. In the other direction, sentence mode gets 4 examples index
mode misses, all of them cases where the gold correction was never in Hunspell's
list -- rewriting the sentence lets the model leave the candidate list, which
forced-choice index mode cannot do (`restrunt` -> `restaurant`, `sespend` ->
`suspension`, `wetty` -> `wet`, `crimbimg` -> `climbing`).

**So the honest reading is punctuation-insensitive: 88.0%.** On that basis sentence
mode beats index mode (83.0%) and ties beam mode (88.0%), and sits just under open
mode (90.0%); ignoring case as well, sentence and open mode are level at 90.0%.
Sentence mode's *conditional* accuracy (83.1% strict) is well below index mode's
100%, but that gap too is mostly the same artifact -- the model rarely disagrees
with a gold correction that Hunspell offered, it just re-punctuates around it.

**The cost is latency, and it is large.** p50 goes 783ms -> 5,980ms (7.6x), p99
901ms -> 16,280ms (18x), worst case 23.4s on the sample's longest sentence. That is
inherent, not an artifact: the model emits ~53 tokens instead of ≤8, at a measured
~126ms per generated token on this 4-vCPU box, and the whole-sentence output makes
per-call latency scale with sentence length rather than staying flat. All 100 calls
landed above 2s and 75 of them above 5s -- there is no overlap at all with index
mode's tight 683-916ms band.

### Revised prompt options

Ranked by what they fix, given that the dominant failure is output-format fidelity
rather than spelling judgement:

1. **Sentence rewrite plus an explicit word field** (recommended). Keep the rewrite
   as the model's reasoning surface but add a second tag it must fill with just the
   replacement token: `<corrected_sentence>...</corrected_sentence><corrected_word>internet</corrected_word>`,
   and score the word field. This removes alignment entirely -- no anchors, no span
   heuristics, no punctuation attachment -- while keeping whatever benefit writing
   the sentence gives. Expected to convert most of the 15 punctuation errors
   directly into correct answers, at a few extra tokens of latency.
2. **Keep the correction marked inside the rewrite**:
   `<corrected_sentence>Now I must buy it on the <corrected>internet</corrected> .</corrected_sentence>`.
   Same robustness benefit as (1) with one tag pair instead of two, and it forces the
   model to point at the slot it changed, which also catches the `Goog` -> `Go good`
   class of mangled rewrite. Slightly more likely to be dropped by a small model
   mid-sentence than a trailing field is.
3. **Teach the tokenisation explicitly, with a one-shot example.** State that the
   input is pre-tokenised and that spacing must be reproduced byte-for-byte, and show
   one input/output pair that keeps ` .` and `do n't` intact. Cheapest change, no
   format risk, but it fights the model's strong prior toward natural prose -- worth
   testing precisely because 73/100 reflowed under an instruction that already said
   this in words.
4. **Add a case/inflection guardrail** to whichever of the above is chosen: keep the
   original capitalisation unless the word is a proper noun, and keep the word's
   number/tense unless context demands otherwise. Targets `english`/`good` (2 cases)
   and arguably `Thank`/`catacombs` (2 more).
5. **Shrink the rewrite to a window.** Ask for only the corrected word plus two words
   of context on each side rather than the whole sentence. Recovers most of the
   latency (output length stops scaling with sentence length) while keeping the
   "write it in context" framing. Weaker than (1)/(2) on fidelity, and long-range
   context stops being in the *output* though it is still in the prompt.
6. **Non-prompt alternative, for completeness**: constrain decoding to copy the input
   tokens verbatim outside the typo slot. This eliminates the entire failure class by
   construction rather than by instruction, but it is a decoding change, not a prompt
   change, and it gives up the sentence-level fluency signal that lets the model
   escape Hunspell's list.

Options (1) and (2) are the ones worth running next; both are cheap, and either
would let the strict number be read directly instead of through a
punctuation-insensitive lens.

## Follow-up: making it fast -- prompt-lookup decoding, then q4_0 + llama.cpp

Sentence mode's 5,980ms median (above) made iteration painful, so the cost was
measured rather than guessed. Fitting latency against generated-token count over
the 100-example run gives:

```
latency ≈ 0.64s fixed + 215ms per generated token   (bf16, transformers, 4 vCPU)
```

It is entirely decode-bound. Nothing ever hit the per-example token cap, so the
budget was never the constraint -- the token *count* was: ~29 generated tokens per
call versus ~1 in index mode. Two thirds of that is avoidable overhead: the tag
names `<corrected_sentence>` and `</corrected_sentence>` cost 5 tokens each (10 of
the 29), and the sentence body is a near-verbatim copy of text already sitting in
the prompt, regenerated one token at a time at full model cost.

### Fix 1: prompt-lookup speculative decoding (output-identical)

Because the answer copies the prompt, transformers' `prompt_lookup_num_tokens`
applies directly: draft N tokens by matching the tail of the generation against the
prompt, verify the whole draft in one forward pass, keep the prefix greedy decoding
would have produced anyway. It is a speed setting, not a behaviour setting -- the
output is identical to plain greedy by construction.

Measured in-process on 10-12 examples of the same sample:

| Variant | Mean latency/call | Output identical to greedy |
|---|---|---|
| greedy (baseline) | 6.95s | -- |
| lookup, draft 5 | 3.58s | 10/10 |
| **lookup, draft 10** | **3.29s** | 10/10 |
| lookup, draft 10, 3-grams | 3.20s | 10/10 |
| lookup, draft 16 | 3.31s | 10/10 |
| lookup, draft 24 | 3.36s | 10/10 |

Gains plateau at a draft length of 10, which is the default; every setting tried
reproduced greedy output exactly. Also tried and rejected: prefilling the opening
tag into the assistant turn and stopping at `</` saves the 10 tag tokens (6.48s) but
leaves a dangling `</` in the recovered word, and it is redundant once lookup is on
-- lookup drafts the tag tokens from the prompt too. **These numbers are from a
12-example micro-benchmark; the full 100-example rerun was interrupted, so the
harness default is validated but not yet re-scored end-to-end at this setting.**

### Fix 2: q4_0 GGUF served by llama.cpp (the current default for new runs)

`--backend llama-cpp` (`spelling_reranker/llama_cpp_backend.py`) serves
`google/gemma-4-E2B-it-qat-q4_0-gguf` -- Google's **quantization-aware-trained**
q4_0 build, 3.35GB of text weights -- through `llama-server`, over its
OpenAI-compatible endpoint. The GGUF's own chat template renders the prompt and the
server's `/tokenize` sizes the per-example budget, so nothing is reconstructed from
a second tokenizer.

Unlike fix 1, **quantization changes the model's answers**, so all three portable
answer modes were re-scored from scratch on the same fixed 100-example sample rather
than inheriting the bf16 numbers:

| Mode | bf16 strict | q4_0 strict | bf16 punct-insensitive | q4_0 punct-insensitive | bf16 p50 | q4_0 p50 |
|---|---|---|---|---|---|---|
| Index | 83.0% | 81.0% | 83.0% | 81.0% | 783ms&#42; | 744ms |
| Open | 90.0% | 87.0% | 90.0% | 87.0% | (not comparable)&#42;&#42; | 960ms |
| Sentence | 73.0% | 66.0% | 88.0% | 86.0% | 5,980ms | **2,118ms** |

&#42; Same-box bf16 index rerun, not the earlier table's 632ms from another session.
&#42;&#42; The bf16 open-mode run hit the disk-I/O stall documented above (10.2s median),
so no honest speedup ratio can be quoted against it.

**The trustworthy speedup is sentence mode's 2.8x** (5,980ms -> 2,118ms, both clean
measurements on this box), and it beats fix 1's ~3.2s as well. Index mode gets
nothing, for a good reason: it generates about one token, so its latency is almost
all prompt processing, which quantization barely helps on CPU.

**What q4_0 costs in accuracy is small but one-directional.** Per-example
agreement with bf16 is 97% (index), 96% (open), 88% (sentence), and the discordant
pairs go almost entirely one way:

| Mode | bf16 wrong -> q4_0 right | bf16 right -> q4_0 wrong |
|---|---|---|
| Index | 0 | 2 (`commonder`->`commoner`, `dalls`->`dells`) |
| Open | 0 | 3 (adds `shadowig`->`shadow`) |
| Sentence | 2 | 9 |

Sentence mode's 9 losses are again mostly the punctuation artifact, not worse
spelling: `stupid?`, `colorful,`, `comfortable,`, `afraid,` are all correct
corrections carrying adjacent punctuation, which is why the punctuation-insensitive
number only moves 88.0% -> 86.0%. Its 2 wins are real (`Thanx` -> `Thanks`, which
bf16 got wrong, and `Tenpura` -> `Tempura` without a stray quote). Zero wins on
index and open across 200 examples is the honest signal here: q4_0 is slightly
worse, by roughly 2-3 points, and n=100 puts that comfortably inside binomial noise
in magnitude even though the direction is consistent.

**Memory**, measured live rather than estimated:

| | Peak RSS | Of which mmapped weights | Of which anonymous |
|---|---|---|---|
| bf16 / transformers | 6.12 GB | 4.77 GB | 0.87 GB |
| q4_0 / llama-server (4096 ctx) | 4.81 GB | 3.27 GB | 1.54 GB |

The bf16 figure is far below the ~10.2GB the checkpoint occupies on disk because
`low_cpu_mem_usage=True` mmaps it and gemma-4-E2B-it's vision/audio towers are never
touched by a text-only prompt. q4_0 saves 1.5GB of weights but spends some of it
back on KV cache and compute buffers sized for a 4096-token context; a smaller `-c`
would recover most of that.

**One harness bug worth recording**, since it looked exactly like a model failure:
`llama-server` was first spawned with `stdout=subprocess.PIPE` and nothing draining
it, so the server blocked the moment the 64KB pipe buffer filled -- presenting as a
119-second call returning an empty string. Server output now goes to a log file.
Separately, gemma-4 defaults to thinking mode under llama.cpp, spending the entire
budget on a chain of thought and returning empty `content`; the server is now started
with `--reasoning off` (the equivalent of the transformers path's
`enable_thinking=False`), and a response carrying only `reasoning_content` raises
rather than being scored as if the reasoning were the answer.

### Not portable to this backend

Beam mode stays on the transformers backend and is rejected with an explicit error
on `--backend llama-cpp`: it needs per-token logprobs and beam search, which
`llama-server` does not expose. Prompt-lookup decoding is likewise transformers-only
here -- llama.cpp's speculative decoding wants a draft model.

### Side effect: the Hunspell pre-pass

Independent of the model, every invocation re-queried Hunspell for all 68,429 BEA
errors before the first model call -- about 13.5 minutes, deterministic, identical
every run, to score 100 examples. That per-typo memo is now persisted between runs
(`data/bea60k/hunspell_suggestions.json`, untracked), keyed on the dictionary's
.dic/.aff hashes so an updated dictionary misses the cache instead of silently
scoring the locked benchmark against different suggestions. Warm: **822s -> 0.3s**,
which is what makes a full 100-example q4_0 run finish in about four minutes wall
clock instead of twenty.

## Reproducing

```bash
python scripts/download_bea60k.py
python scripts/llm_judge_bea60k_cpu.py \
    --model-id Qwen/Qwen3.5-0.8B --model-name qwen3.5-0.8b \
    --bea-dir data/bea60k --output reports/llm_judge_cpu/qwen3.5-0.8b
python scripts/llm_judge_bea60k_cpu.py \
    --model-id google/gemma-4-E2B-it --model-name gemma-4-e2b \
    --bea-dir data/bea60k --output reports/llm_judge_cpu/gemma-4-e2b
python scripts/llm_judge_bea60k_cpu.py \
    --model-id openbmb/MiniCPM5-1B --model-name minicpm5-1b \
    --bea-dir data/bea60k --output reports/llm_judge_cpu/minicpm5-1b

# Open-answer ablation (same 100-sample subset, candidates shown only as a hint):
python scripts/llm_judge_bea60k_cpu.py \
    --model-id google/gemma-4-E2B-it --model-name gemma-4-e2b-open \
    --bea-dir data/bea60k --output reports/llm_judge_cpu/gemma-4-e2b-open \
    --answer-mode open --skip-timed

# Beam-search ablation (same 100-sample subset, no Hunspell candidates at all):
python scripts/llm_judge_bea60k_cpu.py \
    --model-id google/gemma-4-E2B-it --model-name gemma-4-e2b-beam \
    --bea-dir data/bea60k --output reports/llm_judge_cpu/gemma-4-e2b-beam \
    --answer-mode beam --beam-width 3 --edit-distance-weight 1.0 --skip-timed

# Whole-sentence-rewrite ablation (same 100-sample subset, Hunspell top-8 shown,
# greedy decoding, no beam search):
python scripts/llm_judge_bea60k_cpu.py \
    --model-id google/gemma-4-E2B-it --model-name gemma-4-e2b-sentence \
    --bea-dir data/bea60k --output reports/llm_judge_cpu/gemma-4-e2b-sentence \
    --answer-mode sentence --skip-timed

# q4_0 GGUF via llama.cpp (the default backend for new runs; re-score, do not
# assume the bf16 accuracy carries over):
python scripts/llm_judge_bea60k_cpu.py \
    --backend llama-cpp \
    --gguf ~/.cache/huggingface/hub/models--google--gemma-4-E2B-it-qat-q4_0-gguf/snapshots/*/gemma-4-E2B_q4_0-it.gguf \
    --llama-server-binary /path/to/llama.cpp/build/bin/llama-server --llama-threads 4 \
    --model-id google/gemma-4-E2B-it-qat-q4_0-gguf --model-name gemma-4-e2b-q4-sentence \
    --bea-dir data/bea60k --output reports/llm_judge_cpu/gemma-4-e2b-q4-sentence \
    --answer-mode sentence --skip-timed
```

This track was built and validated directly on the local CPU box since
`RUNPOD_KEY` was not available in this session -- no pod, no bootstrap script.
For GPU execution, `scripts/llm_judge_bea60k.py` / `spelling_reranker/llm_judge.py`
/ `scripts/runpod/bootstrap_llm_judge.sh` are a separate, GPU-validated index-mode
implementation (this CPU track's "open"/"beam" answer modes and the edit-distance
reranker are not part of that one).
