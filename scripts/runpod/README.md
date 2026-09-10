# Runpod helpers

`RUNPOD_KEY` must be in the environment. Pods bill for as long as they exist —
**always** finish with `terminate.py`.

```bash
python scripts/runpod/launch.py                 # byte-level reranker (default)
python scripts/runpod/launch.py --experiment frozen --branch <this-branch>
# ... pull artifacts back ...
python scripts/runpod/terminate.py --all        # stop paying
```

Self-driving pods clone the repo, run `scripts/runpod/setup.sh`, then either
`run_experiment.sh` (byte model) or `run_frozen_experiment.sh` when
`--experiment frozen` / `--config configs/train_frozen_modernbert.yaml` is set.

Frozen encoder: A5000-first, 100 GB disk, **cu124/py3.11** image so hunspell
builds, then 200k-subset cache + H1/H2/H3. Do not pass the cu128/py3.12 torch
2.8 image until hunspell works on 3.12.

```bash
python scripts/runpod/launch.py \
  --experiment frozen \
  --config configs/train_frozen_modernbert.yaml
```

See [reports/FROZEN_ENCODER.md](../../reports/FROZEN_ENCODER.md).

LLM-judge / Gemma-4 (`--bootstrap-path scripts/runpod/bootstrap_llm_judge.sh`)
also defaults to that **cu124 / py3.11** image. `setup_llm_judge.sh` then
installs `torch==2.5.1+cu124` (so `torch.distributed.tensor.DTensor` exists),
force-installs `nvidia-cudnn-cu12` into the venv, and writes
`/workspace/llm-judge-env.sh` (`LD_LIBRARY_PATH` for `libcudnn.so.9`).
Bootstrap sources that file. Do not pass a cu128 image or cu128 wheels:
Community hosts with a CUDA 12.4 driver fall back to CPU. See
[reports/EXPERIMENT_LLM_JUDGE.md](../../reports/EXPERIMENT_LLM_JUDGE.md).

## Where the setup time went

The first experiment spent roughly 25 minutes before a single optimizer step.
This layout removes most of it:

| Cost | Before | Now |
|------|--------|-----|
| torch wheel download | ~3 min | 0 — the pod image ships CUDA torch |
| apt, pip, corpus download | serial, ~4 min | concurrent, ~90 s |
| `pip install hunspell` | fails on modern setuptools, then a debug cycle | pinned `setuptools<60` + `--no-build-isolation` |
| training-data build | ~18 min for 240k examples | Hunspell is called once per *unique typo* rather than once per example, so the cost no longer scales with dataset size |
| stale committed parquet | pulled via Git LFS, then rebuilt anyway | clone with `GIT_LFS_SKIP_SMUDGE=1` |

Clone on the pod with:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 -b <branch> <url> /workspace/spell-corrector
```
