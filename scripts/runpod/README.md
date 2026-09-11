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

## GPU llama.cpp track (q4_0 gemma-4-E2B, `launch_gpu_llama.py`)

A serving benchmark rather than a training run: build llama.cpp with CUDA,
serve Google's QAT q4_0 GGUF with **full GPU offload**, and measure prefill /
decode throughput, a concurrency sweep and a $/1,000-corrections table.

```bash
python scripts/runpod/launch_gpu_llama.py --branch <this-branch>
# monitor, fetch, then ALWAYS:
python scripts/runpod/fetch_artifacts.py <pod-id> --dest .
python scripts/runpod/terminate.py --all
```

Two things about it differ from the other launchers:

* **The image is a CUDA `-devel` one** (`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`).
  `-DGGML_CUDA=ON` needs `nvcc`, which the runtime images do not ship.
* **GPU choice is a cheapest-first walk, not a preference list handed to the
  API.** The q4_0 weights are 3.35GB so every card from an 8GB RTX 3070 up
  fits; the only variable is price, and on the community cloud nearly every
  cheap type sits at "Low" stock. `launch_gpu_llama.py` therefore creates the
  pod one GPU type at a time in price order and records every attempt — the
  failures included — so the write-up can say which GPU was *obtainable*, not
  which was theoretically cheapest. Tesla V100 stays last despite its price
  (sm_70, no bf16).

The pod serves `RUNINFO.json` (chosen GPU, price, attempt log, llama.cpp
commit, server command, startup seconds) alongside the usual `run.log` /
`STATUS` / `DONE`, and `llama_server.log` directly so a partial GPU offload is
visible without fetching anything.

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
