# CLAUDE.md — spell-corrector

## Environment / secrets

- **`RUNPOD_KEY`** is available as an environment variable in this session. It is a
  Runpod API key. Use it for both APIs:
  - REST: `curl -H "Authorization: Bearer $RUNPOD_KEY" https://rest.runpod.io/v1/pods`
  - GraphQL: `https://api.runpod.io/graphql` with the same bearer token.
  Never print the key, commit it, or bake it into a script — always read it from the env.
- GPU work runs on Runpod. Helper scripts live in `scripts/runpod/`.
  **Always terminate the pod when the run finishes** (`scripts/runpod/terminate.py`),
  pods bill per second while they exist.

## Local box

4 vCPU, no GPU. Enough for unit tests, data diagnostics and Hunspell/Aspell work,
not for training.

`pip install hunspell==0.5.5` builds from source and fails on modern
setuptools (`AttributeError: install_layout`). On Python 3.11 and older:

```bash
pip install "setuptools<60" wheel
pip install --no-build-isolation hunspell==0.5.5
```

That workaround does **not** transfer to Python 3.12: it has no `distutils`
for old setuptools to patch, and the build dies with
`BackendUnavailable: Cannot import 'setuptools.build_meta'`. On 3.12 (which is
what the Runpod images ship) install the distro package instead — same 0.5.5,
already compiled for the running interpreter:

```bash
apt-get install -y python3-hunspell
```

## Benchmark

BEA-60K (`scripts/download_bea60k.py`) is a **locked** benchmark: never train,
validate or tune on it. It is downloaded, never committed.

This lockout is strict and test-enforced for the **byte-level reranker** track
(`spelling_reranker/`, `scripts/build_training_data.py`;
`tests/test_dataset.py::test_training_construction_does_not_reference_locked_benchmark`
fails the build if any of those files even mentions BEA).

The **LLM direct-correct track** (milestones 3-7, `scripts/distill_2b_to_student.py`
and the retired `train_direct_qlora.py` copies under `artifacts/spell_slm_m*/`)
runs a narrower rule instead, in place since milestone 4: a fixed seed-1337
100-example holdout is carved out of BEA-60K and never trained or tuned on;
every other BEA-60K word error is fair game for training data (milestones 4-6
capped this at `max_bea_typos: 20000`; milestone 7's distillation run lifts
that cap and uses the full remaining pool). This is a deliberate, user-approved
exception scoped to that experiment line only — it does not relax the
byte-level reranker's rule above, and any accuracy number this track reports
against the **full** BEA-60K benchmark (rather than its own 100-sample holdout)
is no longer meaningful once BEA text has been trained on.

## Milestone 7 — 2B → 0.8B distillation

Teacher: `Qwen/Qwen3.5-2B` + milestone-6 QLoRA adapters
(`artifacts/spell_slm_m6/qwen35_2b_direct_qlora/`, 91% Acc@1 on the frozen
BEA-100), loaded 4-bit NF4. Student: fresh `Qwen/Qwen3.5-0.8B` QLoRA, 4-bit
NF4, trained on teacher-generated labels over the BEA-60K training pool
(sequence-level distillation — the student learns to reproduce the teacher's
corrections, not necessarily BEA's raw gold). Pipeline:
`scripts/distill_2b_to_student.py {prepare-data,label-teacher,train,eval}`,
driven on Runpod by `scripts/runpod/bootstrap_distill.sh` /
`setup_distill.sh`. `train` runs in per-epoch rounds, evaluating Acc@1 against
the locked 100-sample holdout after every round, and stops as soon as it
exceeds 90% or a round budget is exhausted — see
`artifacts/spell_slm_m7/qwen35_0_8b_distill_qlora/distill_progress.jsonl` for
the round-by-round loss/accuracy trace.
