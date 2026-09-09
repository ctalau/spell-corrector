#!/usr/bin/env bash
# Fast pod bootstrap. Target: usable box in ~2 minutes, not ~15.
#
# What makes it fast:
#   * the pod image already ships CUDA torch, so torch is never reinstalled
#     (that alone is a ~2.5 GB download);
#   * apt and pip run concurrently -- they contend for nothing;
#   * the WikiText download starts immediately, in parallel with both;
#   * `hunspell==0.5.5` is built with setuptools<60. Modern setuptools removes
#     `install_layout` and the build dies with a confusing AttributeError.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/spell-corrector}"
cd "$REPO_DIR"

log() { printf '\n=== %s (%ss) ===\n' "$1" "$SECONDS"; }

log "apt + pip + data download in parallel"

(
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends \
    hunspell libhunspell-dev hunspell-en-us aspell aspell-en
) >/tmp/apt.log 2>&1 &
APT_PID=$!

(
  python -m pip install -q --upgrade pip
  # Everything except torch (already in the image) and hunspell (needs libs
  # from the apt job, so it is installed after the wait below).
  python -m pip install -q \
    numpy pandas pyarrow tqdm pyyaml safetensors matplotlib requests pytest
) >/tmp/pip.log 2>&1 &
PIP_PID=$!

(
  python scripts/download_sources.py
) >/tmp/data.log 2>&1 &
DATA_PID=$!

wait "$APT_PID" || { echo "apt failed:"; tail -30 /tmp/apt.log; exit 1; }
log "apt done"

# Needs libhunspell-dev from the apt job above.
python -m pip install -q "setuptools<60" wheel
python -m pip install -q --no-build-isolation hunspell==0.5.5

wait "$PIP_PID" || { echo "pip failed:"; tail -30 /tmp/pip.log; exit 1; }
log "pip done"

wait "$DATA_PID" || { echo "source download failed:"; tail -30 /tmp/data.log; exit 1; }
log "wikitext download done"

log "verifying"
python - <<'PY'
import torch
from spelling_reranker.hunspell import default_engine
from spelling_reranker.model import ByteSpellingReranker, count_parameters
from spelling_reranker.config import load_yaml, model_config_from_mapping

print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())
print("hunspell suggest('recieve') ->", default_engine().suggest("recieve"))
cfg = model_config_from_mapping(load_yaml("configs/model_87m.yaml"))
print("params", f"{count_parameters(ByteSpellingReranker(cfg)):,}")
PY

log "setup complete"
