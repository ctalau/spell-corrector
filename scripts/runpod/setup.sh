#!/usr/bin/env bash
# Fast pod bootstrap. Target: usable box in ~2 minutes, not ~15.
#
# What makes it fast:
#   * the pod image already ships CUDA torch, so torch is never reinstalled
#     (that alone is a ~2.5 GB download);
#   * apt and the WikiText download run concurrently;
#   * project deps and hunspell go into a venv, not distro Python.
#
# The Hunspell *library* comes from apt (libhunspell-dev + dictionaries).
# The Python binding is pip-installed into /workspace/venv. apt
# python3-hunspell and `pip install` into /usr/bin/python3 are both
# untrustworthy on these images:
#   * apt's module is compiled for distro Python (often 3.10 on Ubuntu 22.04)
#     while /usr/bin/python3 is 3.11 and ships torch;
#   * upgrading pip then installing into Debian Python lands wheels in
#     site-packages, which that interpreter does not import (it searches
#     dist-packages). pip exits 0; `import hunspell` still fails.
# The venv is created with --system-site-packages so image torch/CUDA stay
# visible. hunspell==0.5.5 still uses distutils; on Python 3.11 pin
# setuptools<60 and disable build isolation. On 3.12+ that pin fails
# (pkgutil.ImpImporter / no distutils); skip it, try apt python3-hunspell,
# and die naming the py3.11 cu124 image if import still fails.
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
print("cuda available", torch.cuda.is_available())
assert torch.cuda.is_available(), (
    "CUDA is not available; refusing to continue. "
    f"torch={torch.__version__} torch.version.cuda={torch.version.cuda}"
)
print("gpu", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())
PY

log "apt + data download in parallel"

(
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends \
    hunspell libhunspell-dev hunspell-en-us aspell aspell-en
  pyver="$("$IMAGE_PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  apt-get install -y -qq --no-install-recommends \
    "python${pyver}-venv" "python${pyver}-dev" \
    || apt-get install -y -qq --no-install-recommends python3-venv python3-dev
) >/tmp/apt.log 2>&1 &
APT_PID=$!

( "$IMAGE_PYTHON" scripts/download_sources.py ) >/tmp/data.log 2>&1 &
DATA_PID=$!

wait "$APT_PID" || { tail -30 /tmp/apt.log; die "apt"; }
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
nvidia-smi || echo "nvidia-smi unavailable"
"$PYTHON" - <<'PY' || die "CUDA not available in venv"
import sys
import torch
print("executable", sys.executable)
print("torch", torch.__version__, "file", torch.__file__)
print("torch.version.cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
assert torch.cuda.is_available(), (
    "CUDA is not available in the venv; refusing to continue. "
    f"torch={torch.__version__} torch.version.cuda={torch.version.cuda}"
)
print("gpu", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())
PY

log "pip deps + hunspell into $VENV"
pip_install /tmp/pip-upgrade.log --upgrade pip || die "pip upgrade"
# Everything except torch, which the image already provides.
pip_install /tmp/pip-deps.log \
  numpy pandas pyarrow tqdm pyyaml safetensors matplotlib requests pytest pytest-timeout \
  "transformers>=4.48,<5" tokenizers huggingface_hub accelerate \
  || die "pip deps"

PY312="$("$PYTHON" -c 'import sys; print("1" if sys.version_info >= (3, 12) else "0")')"
if [ "$PY312" = "1" ]; then
  # hunspell==0.5.5 still uses distutils. setuptools<60 restores that on
  # 3.11 but has no distutils to patch on 3.12 (ImpImporter / BackendUnavailable).
  log "Python 3.12+: skip setuptools<60; try distro python3-hunspell"
  apt-get install -y -qq --no-install-recommends python3-hunspell \
    || echo "apt python3-hunspell unavailable"
  if ! "$PYTHON" - <<'PY'
import hunspell
print("hunspell", hunspell.__file__)
PY
  then
    die "hunspell==0.5.5 cannot build on Python 3.12 (setuptools<60 / ImpImporter). Use runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04. The cu128/py3.12 torch 2.8 image is not supported until hunspell is fixed."
  fi
else
  pip_install /tmp/pip-build.log "setuptools<60" wheel cython || die "pip build deps"
  pip_install /tmp/hunspell-pip.log --no-build-isolation --force-reinstall hunspell==0.5.5 \
    || die "hunspell pip install into ${PYTHON}"
  "$PYTHON" -m pip show hunspell || die "pip show hunspell"
fi

log "hunspell import check"
"$PYTHON" - <<'PY' || die "import hunspell"
import sys
import hunspell
print("executable", sys.executable)
print("path:")
for p in sys.path:
    print(f"  {p}")
print("hunspell", hunspell.__file__)
PY

wait "$DATA_PID" || { tail -30 /tmp/data.log; die "wikitext download"; }
log "wikitext download done"

log "verifying"
nvidia-smi || echo "nvidia-smi unavailable"
"$PYTHON" - <<'PY' || die "verify"
import sys
import hunspell
import torch
from spelling_reranker.hunspell import default_engine
from spelling_reranker.model import ByteSpellingReranker, count_parameters
from spelling_reranker.config import load_yaml, model_config_from_mapping

print("executable", sys.executable)
print("path:")
for p in sys.path:
    print(f"  {p}")
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

log "setup complete (PYTHON=$PYTHON)"
