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
