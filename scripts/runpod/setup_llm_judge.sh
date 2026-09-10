#!/usr/bin/env bash
# Pod setup for the LLM-judge experiment only: hunspell + a lean Python venv
# with transformers/accelerate. Skips the WikiText download and the heavier
# pandas/pyarrow stack that scripts/runpod/setup.sh installs for the trained
# reranker's training-data pipeline -- this experiment builds no training data.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/spell-corrector}"
IMAGE_PYTHON="${IMAGE_PYTHON:-$(command -v python3 || command -v python)}"
VENV="${VENV:-/workspace/venv}"
export IMAGE_PYTHON
cd "$REPO_DIR"

log() { printf '\n=== %s (%ss) ===\n' "$1" "$SECONDS"; }
die() { echo "SETUP FAILED: $1"; exit 1; }

case "$VENV" in
  ""|"/"|"/usr"|"/usr/bin"|"/workspace") die "refusing to rm -rf VENV=$VENV" ;;
esac

use_venv_python() {
  PYTHON="$VENV/bin/python"
  export PYTHON
  export PATH="$VENV/bin:$PATH"
  mkdir -p /workspace
  printf '%s\n' "$PYTHON" > /workspace/python-interpreter
}

pip_install() {
  local log="$1"; shift
  echo "+ $PYTHON -m pip install $*" | tee "$log"
  if ! "$PYTHON" -m pip install "$@" >>"$log" 2>&1; then
    echo "----- $log -----"
    tail -80 "$log"
    return 1
  fi
  cat "$log"
  return 0
}

log "CUDA check ($IMAGE_PYTHON)"
nvidia-smi || echo "nvidia-smi unavailable"
"$IMAGE_PYTHON" - <<'PY' || die "CUDA not available"
import torch
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA is not available; refusing to continue"
print("gpu", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())
PY

log "apt: hunspell"
(
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends \
    git curl ca-certificates hunspell libhunspell-dev hunspell-en-us
  pyver="$("$IMAGE_PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  apt-get install -y -qq --no-install-recommends \
    "python${pyver}-venv" "python${pyver}-dev" \
    || apt-get install -y -qq --no-install-recommends python3-venv python3-dev
) >/tmp/apt.log 2>&1 || { tail -30 /tmp/apt.log; die "apt"; }
log "apt done"

log "venv $VENV --system-site-packages"
rm -rf "$VENV"
"$IMAGE_PYTHON" -m venv --system-site-packages "$VENV" || die "venv"
use_venv_python
echo "venv python=$PYTHON ($($PYTHON -V 2>&1))"
if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
  "$PYTHON" -m ensurepip --upgrade || die "ensurepip"
fi

log "CUDA check (venv, image torch via system site packages)"
"$PYTHON" - <<'PY' || die "CUDA not available in venv"
import torch
print("torch", torch.__version__, "cuda avail", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA is not available in the venv"
print("gpu", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())
PY

log "pip deps + hunspell + transformers into $VENV"
pip_install /tmp/pip-upgrade.log --upgrade pip || die "pip upgrade"
pip_install /tmp/pip-deps.log \
  numpy matplotlib requests pytest \
  || die "pip deps"
pip_install /tmp/pip-build.log "setuptools<60" wheel cython || die "pip build deps"
pip_install /tmp/hunspell-pip.log --no-build-isolation --force-reinstall hunspell==0.5.5 \
  || die "hunspell pip install into ${PYTHON}"
"$PYTHON" -m pip show hunspell || die "pip show hunspell"
pip_install /tmp/pip-transformers.log -U "transformers>=4.57" "accelerate>=0.34" \
  || die "pip transformers/accelerate"

log "hunspell import check"
"$PYTHON" - <<'PY' || die "import hunspell"
import hunspell
print("hunspell", hunspell.__file__)
PY

log "verifying"
"$PYTHON" - <<'PY' || die "verify"
import torch, transformers
from spelling_reranker.hunspell import default_engine
print("torch", torch.__version__, "cuda avail", torch.cuda.is_available())
print("transformers", transformers.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; refusing to continue")
print("gpu", torch.cuda.get_device_name(0))
engine = default_engine()
assert engine.suggest("recieve"), "hunspell returned no suggestions"
print("hunspell suggest('recieve') ->", engine.suggest("recieve"))
PY

log "setup complete (PYTHON=$PYTHON)"
