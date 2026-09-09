# Experiment 2 — plan and rationale

Target: **75% overall top-1 correction accuracy on BEA-60K**, up from 62.56%.

## The constraint that shapes everything

```
overall accuracy = P(gold in candidate pool) x P(model picks gold | it is there)
```

Hunspell is fixed as the only candidate generator, so the first factor is a
ceiling the model cannot move: ~80.4% at 10 slots, ~81.2% using Hunspell's
untruncated list at 16 slots.

|                    | Experiment 1 | Needed for 75% |
|--------------------|--------------|----------------|
| Pool coverage      | 80.38%       | ~81.2% (fixed) |
| Conditional accuracy | 77.84%     | **~92.4%**     |
| Overall            | 62.56%       | 75%            |

So the entire experiment is about conditional accuracy: given that Hunspell
already offered the right word, pick it. That is a stretch — it means being
wrong on fewer than 1 in 13 solvable cases — and the run may land short.

## Diagnosis of experiment 1

The decisive finding came from the typo generator, not the model. Experiment 1
generated **exclusively edit-distance-1 typos**. Authentic misspellings are not:

| Edit distance | Wikipedia common-misspellings list | Experiment 1 training data |
|---|---|---|
| 1 | 72.6% | 100% |
| 2 | 25.0% | 0% |
| 3+ | 2.4% | 0% |

Roughly a quarter of real errors were a shape the model had never seen — and
they are the hard quarter, because Hunspell's own top-1 degrades sharply with
edit distance, which is precisely where a reranker is supposed to add value.

Supporting problems:

* **Clean training context.** A spell corrector reads text the writer has not
  corrected, so neighbouring words are frequently misspelled too. Training only
  on clean WikiText taught the model to trust context it will not get.
* **Trivial-example dominance.** Hunspell already ranks the answer first for
  ~81% of synthetic typos; those examples teach the model only to agree with
  Hunspell.
* **Data starvation.** 235k examples, 3 epochs.
* **A pooling bug masquerading as a memory requirement.** Candidate pooling
  materialised a `[batch, cands, seq, dim]` tensor (~1 GB per batch), which is
  what drove the 22 GiB peak and capped throughput.

## Changes

| # | Change | Expected effect |
|---|--------|-----------------|
| 1 | Typo generator composes 1-3 edits and adds phonetic/orthographic rules (doubling, silent letters, reduced vowels, suffix confusion) | Largest. Closes the ED>=2 blind spot. |
| 2 | Noisy-context augmentation (`p=0.25`) | Removes the clean-context mismatch. |
| 3 | Gold-index balancing (cap slot 0 at 65%) | Concentrates training on decidable cases. |
| 4 | 16 candidate slots, Hunspell list untruncated | +~0.7pp ceiling. |
| 5 | 4M training examples (17x) | Data was the binding constraint. |
| 6 | 87M parameters (from 28M) | Context modelling is now the constraint. |
| 7 | `cand*typo` and `abs(cand-typo)` head features | Direct candidate/typo comparison. |
| 8 | Auxiliary masked-byte objective, decayed out over the first 60% of training | The reranking signal alone teaches almost no English; this gives the encoder a language-modelling gradient inside the single run. |
| 9 | `bmm` candidate pooling, numpy collation, length-bucketed batching | Throughput and VRAM, enabling 5 and 6. |

## Benchmark integrity

The typo generator is calibrated against Wikipedia's public
"Lists of common misspellings" (~4.3k authentic pairs), **not** BEA-60K.
`scripts/calibrate_typo_model.py` reproduces the comparison and writes
`reports/typo_calibration.json`. `tests/test_dataset.py` fails the build if any
training-construction file references the benchmark at all.

Choices such as `gold0_fraction` and `context_noise_prob` were fixed a priori
from the structure of the task, not tuned against BEA.

## Calibration result

| | Authentic (Wikipedia) | Synthetic (this generator) |
|---|---|---|
| ED1 | 72.55% | 73.28% |
| ED2 | 25.01% | 21.80% |
| ED3+ | 2.44% | 4.92% |
| Hunspell gold at rank 0 | 80.11% | 81.08% |
| Gold absent from pool | 5.31% | 8.22% |
