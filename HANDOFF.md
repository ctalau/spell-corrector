# Handoff — Experiment 2: 28M → 87M Hunspell reranker, targeting 75% on BEA-60K

Branch: `claude/model-training-75-percent-k0bqon`
Last commit at handoff: `09b7978`

---

## 1. The goal and why it is hard

Target: **75% overall top-1 correction accuracy on BEA-60K**, up from experiment 1's 62.56%.

The governing identity:

```
overall accuracy = P(gold is in the candidate pool) x P(model picks gold | it is there)
62.56%           =            80.38%                x            77.84%
```

**Hunspell is fixed as the only candidate generator** (an explicit user
decision — Aspell was ruled out). That pins the first factor at a hard ceiling
of ~81%, so the entire experiment rides on the second factor:

|                      | Experiment 1 | Needed for 75% |
|----------------------|--------------|----------------|
| Pool coverage        | 80.38%       | ~81% (fixed)   |
| Conditional accuracy | 77.84%       | **~92.4%**     |
| Overall              | 62.56%       | 75%            |

**Be honest about this in the writeup: 92.4% conditional is a stretch and the
run may land short (a 72-75% result is a realistic outcome).** Do not present a
near-miss as a success. Measured for reference on a 6,000-error BEA sample: a
Hunspell ∪ Aspell pool would have raised the ceiling to 87.7%, which is the
single biggest lever available — but it is out of scope by user decision.

## 2. The core diagnosis

Experiment 1's decisive defect was in the **training data, not the model**: the
typo generator produced **exclusively edit-distance-1 typos**. Authentic
misspellings are ~73% ED1 / ~25% ED2 / ~2% ED3+, and Hunspell's own top-1
accuracy collapses as edit distance grows (60.2% → 46.6% → 4.9% on BEA). So a
quarter of real errors — precisely the quarter where a reranker earns its keep —
was a shape the model had never seen.

## 3. What changed (all committed)

| # | Change | Where |
|---|--------|-------|
| 1 | Typo generator composes 1-3 edits + phonetic/orthographic rules (doubling, silent letters, reduced vowels, suffix confusion) | `spelling_reranker/typo_gen.py` |
| 2 | Noisy-context augmentation, p=0.25 | `data_build.iter_examples` |
| 3 | Gold-index balancing, slot 0 capped at 65% | `data_build.iter_examples` |
| 4 | 16 candidate slots; Hunspell list no longer truncated at 10 | `byte_encoding.py`, `candidates.py` |
| 5 | ~2.1M training examples (vs 235k) | build defaults |
| 6 | 87M params (vs 28M) | `configs/model_87m.yaml` |
| 7 | Head gains `cand*typo` and `abs(cand-typo)` features | `model.ScoringHead` |
| 8 | Auxiliary masked-byte objective, weight 0.20, decayed to 0 by 60% of training | `model.py`, `scripts/train.py` |
| 9 | `bmm` candidate pooling, numpy collation, length-bucketed batching | `model.py`, `dataset.py` |

### Benchmark integrity — important

The typo generator is calibrated against **Wikipedia's public common-misspellings
list** (`data/wikipedia_misspellings.txt`, vendored, CC BY-SA), *not* BEA-60K.
An earlier draft calibrated against BEA statistics; the repo's own leak test
caught it. `tests/test_dataset.py::test_training_construction_does_not_reference_locked_benchmark`
fails the build if any training-construction file so much as mentions the
benchmark. **Keep it that way.** `gold0_fraction` and `context_noise_prob` were
fixed a priori from the structure of the task, never tuned against BEA.

Calibration result (`reports/typo_calibration.json`):

| | Authentic (Wikipedia) | Synthetic |
|---|---|---|
| ED1 | 72.55% | 73.28% |
| ED2 | 25.01% | 21.80% |
| ED3+ | 2.44% | 4.92% |

## 4. How to run it

The controlling sandbox has **outbound HTTPS only — SSH to the pod is blocked**.
The pod is therefore self-driving: `launch.py` sets the container entrypoint to
`bootstrap.sh`, which clones, sets up, runs everything, and serves progress and
artifacts over Runpod's HTTP proxy.

```bash
python scripts/runpod/launch.py \
  --name spell-corrector-exp2 --disk-gb 100 \
  --branch claude/model-training-75-percent-k0bqon \
  --target-train 3000000 --target-valid 60000
```

Monitor (pod id from launch output):

```
https://<pod-id>-8000.proxy.runpod.net/run.log
https://<pod-id>-8000.proxy.runpod.net/STATUS   # experiment | SUCCESS | FAILED: ...
https://<pod-id>-8000.proxy.runpod.net/DONE     # appears when finished
```

Pull results back, then **always** terminate:

```bash
python scripts/runpod/fetch_artifacts.py <pod-id> --dest .
python scripts/runpod/terminate.py --all
```

`RUNPOD_KEY` is in the environment. **Pods bill per second for as long as they
exist** — terminate on every exit path, including failures.

Phase timings observed on an L40S / 28 vCPU pod:

| Phase | Time |
|-------|------|
| Setup (apt+pip+corpus, parallel) | 15 s |
| Unit tests | 40 s |
| Preflight (configs on GPU) | 5 s |
| Typo calibration | ~3.5 min |
| Data build | ~19 min |
| Sanity train | ~1 min |
| Full train (87M, ~8.1k steps) | ~3-4 h (unverified) |
| BEA-60K benchmark | ~15 min (unverified) |

## 5. Where the run had got to at handoff

Pod `3gxtgufkzcvl40`, commit `09b7978`. Preflight passed on GPU; in the typo
calibration / data build phase. **Check `STATUS` first.** If it says `SUCCESS`,
fetch artifacts and write up. If `FAILED`, read `run.log` for the phase and
`boot.err` for early bootstrap failures.

Prior run produced **2,081,171 train / 20,773 validation** examples from 46,825
words and 410,555 unique typos.

## 6. Bugs already fixed — do not reintroduce

Eight pods failed before the pipeline held. Each was a distinct real defect:

1. **`dockerStartCmd` swallowed** by the image's init → use `dockerEntrypoint`.
2. **`PATH` lacks Python/git** once the entrypoint is overridden → `bootstrap.sh`
   resolves `python3` explicitly and installs git/curl.
3. **`setuptools<60` is wrong on Python 3.12** (no `distutils`; fails with
   `BackendUnavailable`). Install `python3-hunspell` from apt instead. The
   `setuptools<60` trick is only for Python ≤3.11 (the local box).
4. **`pyarrow` import race** — the corpus download ran concurrently with the pip
   install providing pyarrow → import is now lazy.
5. **`set -u` aborts** while sourcing the image's `.bashrc` → `set -o pipefail` only.
6. **raw.githubusercontent caches branch paths** for minutes → `launch.py` pins
   an exact commit SHA. This also makes runs reproducible.
7. **Parquet schema drift** — pandas infers dtypes per chunk, so a chunk whose
   `cand_i` is entirely null infers `null` not `string` and the next chunk fails
   with `Table schema does not match schema used to create file`. Killed a build
   13% in. Fixed by pinning `example_schema()`.
8. **`vocab_size` 280 vs `MASK_ID` 280** in `model_87m.yaml` — an anchored `sed`
   silently missed the line because it carries a trailing comment. Died as an
   async CUDA gather assert on the full train's first batch, *after* the 23-min
   data build. The sanity train missed it because it uses the 28M config.

Lesson from #8, now enforced: **`scripts/preflight.py` builds every training
config on the real GPU and takes an optimizer step with byte masking live,
before the data build.** A sanity check that exercises a different config than
the real run is not a sanity check.

## 7. Known issues, not yet fixed

- **Task #8 — typo-table sharding depends on CPU count.** `build_typo_table`
  shards by worker count and seeds each shard from its index, so the dataset
  differs between machines (28 vCPU gave 46,825 words/410,555 entries; 112 gave
  46,819/410,594). Same distribution, not byte-identical. The determinism test
  passes only because it pins `workers=1`. Fix: fixed shard size independent of
  CPU count.
- **Task #9 — example target sits at 91% of typo-table capacity**
  (410,555 typos × `max_uses_per_typo=8` = 3.28M vs a 3M target), so generation
  decays from 8.4k/s to <2.5k/s and stops short (2.08M). Efficiency only; the
  build is bounded by `max_passes` and self-terminates. Fix: target ~70% of
  capacity — raise `max_uses_per_typo`, widen the vocabulary, or lower the target.
- `nproc` inside a Runpod container reports the **host's** cores (112) not the
  allocation (28). `data_build.available_cpus()` handles this; anything new that
  sizes a pool must use it, not `os.cpu_count()`.

## 8. Writing up the result

```bash
python scripts/summarize_experiment.py
```

prints the headline table, whether 75% was met, the conditional accuracy that
would have been needed, and splits the residual error into "gold outside pool"
(unreachable without changing candidate generation) vs "model picked wrong"
(reachable by better reranking). Put the numbers in `reports/EXPERIMENT.md`
alongside `reports/training_loss.png`. `reports/EXPERIMENT2_PLAN.md` holds the
rationale and the full run configuration.

Report the achieved number plainly, including if it misses 75%. The decomposition
above is the useful part of the result either way: it says whether the remaining
gap is candidate generation (needs Aspell or a dictionary-search generator, both
currently out of scope) or reranking (needs more model/data).

## 9. Budget

Runpod balance was $9.70 at session start; roughly **$1.30** spent across nine
pods (eight short-lived failures plus the current run). A full training run
costs ~$3 at $0.79/hr. Check with:

```bash
curl -s -H "Authorization: Bearer $RUNPOD_KEY" -H "Content-Type: application/json" \
  -d '{"query":"query{myself{clientBalance currentSpendPerHr}}"}' \
  https://api.runpod.io/graphql
```
