#!/usr/bin/env bash
# Fast pod bootstrap. Target: usable box in ~2 minutes, not ~15.
#
# What makes it fast:
#   * the pod image already ships CUDA torch, so torch is never reinstalled
#     (that alone is a ~2.5 GB download);
#   * apt and pip run concurrently -- they contend for nothing;
#   * the WikiText download starts immediately, in parallel with both.
#
# The Hunspell *library* comes from apt (libhunspell-dev + dictionaries).
# The Python binding must be pip-installed into $PYTHON: apt python3-hunspell
# is compiled for distro Python (/usr/bin/python3), while Runpod images put
# torch on a different interpreter (often /usr/local/bin/python).
# hunspell==0.5.5 still uses distutils; on Python 3.11 pin setuptools<60 and
# disable build isolation so pip uses that copy.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/spell-corrector}"
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
cd "$REPO_DIR"

log() { printf '\n=== %s (%ss) ===\n' "$1" "$SECONDS"; }
die() { echo "SETUP FAILED: $1"; exit 1; }

log "CUDA check ($PYTHON)"
nvidia-smi || echo "nvidia-smi unavailable"
"$PYTHON" - <<'PY' || die "CUDA not available"
import torch
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
assert torch.cuda.is_available(), (
    "CUDA is not available; refusing to continue. "
    f"torch={torch.__version__} torch.version.cuda={torch.version.cuda}"
)
print("gpu", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())
PY

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

log "hunspell pip binding ($PYTHON)"
"$PYTHON" -m pip install -q "setuptools<60" wheel cython
"$PYTHON" -m pip install -q --no-build-isolation hunspell==0.5.5 \
  || die "hunspell pip install into ${PYTHON}"
"$PYTHON" -c "import hunspell; print(hunspell.__file__)" || die "import hunspell ($PYTHON)"

wait "$DATA_PID" || { tail -30 /tmp/data.log; die "wikitext download"; }
log "wikitext download done"

log "verifying"
nvidia-smi || echo "nvidia-smi unavailable"
"$PYTHON" -c "import hunspell; print(hunspell.__file__)" || die "import hunspell ($PYTHON)"
"$PYTHON" - <<'PY' || die "verify"
import hunspell
import torch
from spelling_reranker.hunspell import default_engine
from spelling_reranker.model import ByteSpellingReranker, count_parameters
from spelling_reranker.config import load_yaml, model_config_from_mapping

print("hunspell", hunspell.__file__)
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is not available; refusing to continue. "
        f"torch={torch.__version__} torch.version.cuda={torch.version.cuda}"
    )
print("gpu", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())
engine = default_engine()
assert engine.suggest("recieve"), "hunspell returned no suggestions"
print("hunspell suggest('recieve') ->", engine.suggest("recieve"))
cfg = model_config_from_mapping(load_yaml("configs/model_87m.yaml"))
print("params", f"{count_parameters(ByteSpellingReranker(cfg)):,}")
PY

log "setup complete"
