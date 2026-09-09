#!/usr/bin/env bash
# Fast pod bootstrap. Target: usable box in ~2 minutes, not ~15.
#
# What makes it fast:
#   * the pod image already ships CUDA torch, so torch is never reinstalled
#     (that alone is a ~2.5 GB download);
#   * apt and pip run concurrently -- they contend for nothing;
#   * the WikiText download starts immediately, in parallel with both.
#
# The Python Hunspell binding comes from apt (python3-hunspell), not pip.
# `pip install hunspell==0.5.5` builds from source and fails on any modern
# setuptools, and the usual workaround -- pinning setuptools<60 -- is itself
# broken on Python 3.12, which has no distutils for old setuptools to patch.
# The distro package is the same 0.5.5, already compiled against the running
# interpreter. The pip build is kept only as a fallback.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/spell-corrector}"
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
cd "$REPO_DIR"

log() { printf '\n=== %s (%ss) ===\n' "$1" "$SECONDS"; }
die() { echo "SETUP FAILED: $1"; exit 1; }

log "apt + pip + data download in parallel"

(
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends \
    hunspell libhunspell-dev hunspell-en-us aspell aspell-en python3-hunspell
) >/tmp/apt.log 2>&1 &
APT_PID=$!

(
  "$PYTHON" -m pip install -q --upgrade pip
  # Everything except torch, which the image already provides.
  "$PYTHON" -m pip install -q \
    numpy pandas pyarrow tqdm pyyaml safetensors matplotlib requests pytest
) >/tmp/pip.log 2>&1 &
PIP_PID=$!

( "$PYTHON" scripts/download_sources.py ) >/tmp/data.log 2>&1 &
DATA_PID=$!

wait "$APT_PID" || { tail -30 /tmp/apt.log; die "apt"; }
log "apt done"

wait "$PIP_PID" || { tail -30 /tmp/pip.log; die "pip"; }
log "pip done"

if ! "$PYTHON" -c "import hunspell" 2>/dev/null; then
  log "apt binding not importable; falling back to building hunspell from pip"
  "$PYTHON" -m pip install -q "setuptools<81" wheel
  "$PYTHON" -m pip install -q --no-build-isolation hunspell==0.5.5 \
    || die "no usable hunspell binding"
fi

wait "$DATA_PID" || { tail -30 /tmp/data.log; die "wikitext download"; }
log "wikitext download done"

log "verifying"
"$PYTHON" - <<'PY' || exit 1
import torch
from spelling_reranker.hunspell import default_engine
from spelling_reranker.model import ByteSpellingReranker, count_parameters
from spelling_reranker.config import load_yaml, model_config_from_mapping

print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())
engine = default_engine()
assert engine.suggest("recieve"), "hunspell returned no suggestions"
print("hunspell suggest('recieve') ->", engine.suggest("recieve"))
cfg = model_config_from_mapping(load_yaml("configs/model_87m.yaml"))
print("params", f"{count_parameters(ByteSpellingReranker(cfg)):,}")
PY

log "setup complete"
