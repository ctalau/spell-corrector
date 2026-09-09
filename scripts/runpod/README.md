# Runpod helpers

`RUNPOD_KEY` must be in the environment. Pods bill for as long as they exist —
**always** finish with `terminate.py`.

```bash
python scripts/runpod/launch.py                 # create pod, print ssh command
# ... ssh in, clone the repo to /workspace/spell-corrector ...
bash scripts/runpod/setup.sh                    # ~2 min bootstrap
nohup bash scripts/runpod/run_experiment.sh > /workspace/run.log 2>&1 &
# ... pull artifacts back ...
python scripts/runpod/terminate.py --all        # stop paying
```

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
