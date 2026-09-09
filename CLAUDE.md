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

`pip install hunspell==0.5.5` fails on modern setuptools
(`AttributeError: install_layout`). Install it as:

```bash
pip install "setuptools<60" wheel
pip install --no-build-isolation hunspell==0.5.5
```

## Benchmark

BEA-60K (`scripts/download_bea60k.py`) is a **locked** benchmark: never train,
validate or tune on it. It is downloaded, never committed.
