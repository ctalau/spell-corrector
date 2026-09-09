#!/usr/bin/env bash
# Full experiment on the pod: data -> sanity -> train -> benchmark.
#
# Run under nohup/tmux; the training leg is hours long:
#   nohup bash scripts/runpod/run_experiment.sh > /workspace/run.log 2>&1 &
# Fail fast through the critical path (data build, training); diagnostics and
# post-training steps are allowed to fail without discarding a trained model.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/spell-corrector}"
TARGET_TRAIN="${TARGET_TRAIN:-4000000}"
TARGET_VALID="${TARGET_VALID:-60000}"
CONFIG="${CONFIG:-configs/train_full.yaml}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR"
# Length bucketing makes batch memory vary a lot, which fragments the caching
# allocator; expandable segments let it reuse those blocks.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

log() { printf '\n########## %s (t=%ss) ##########\n' "$1" "$SECONDS"; }

# Sample GPU memory alongside training so the report can quote a real peak.
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
  --format=csv -l 5 > artifacts/vram_sampler.csv 2>/dev/null &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null || true' EXIT

die() { echo "EXPERIMENT FAILED: $1"; exit 1; }

# Refuse CPU fallback before any expensive work. A driver/image CUDA mismatch
# (e.g. host 12.4 vs cu128 torch) makes torch.cuda.is_available() False while
# nvidia-smi still works; training would then silently run on CPU for hours.
log "CUDA check"
nvidia-smi || echo "nvidia-smi unavailable"
"${PYTHON:-python}" - <<'PY' || die "CUDA not available"
import torch
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
assert torch.cuda.is_available(), (
    "CUDA is not available; refusing to train on CPU. "
    f"torch={torch.__version__} torch.version.cuda={torch.version.cuda}"
)
print("device", torch.cuda.get_device_name(0))
PY

log "unit tests"
"${PYTHON:-python}" -m pytest tests/ -q || die "unit tests"

# Before anything expensive: build every training config on the real GPU and
# take an optimizer step with all code paths live. A config error otherwise
# surfaces only after the 20+ minute data build.
log "preflight (configs on GPU)"
"${PYTHON:-python}" scripts/preflight.py || die "preflight"

log "typo generator calibration (public misspelling list)"
# Diagnostic only: never let it gate the run.
"${PYTHON:-python}" scripts/calibrate_typo_model.py || echo "calibration skipped"

log "building training data (target ${TARGET_TRAIN})"
"${PYTHON:-python}" scripts/build_training_data.py \
  --target-train "$TARGET_TRAIN" \
  --target-valid "$TARGET_VALID" \
  --seed 1337 || die "data build"

log "sanity train (must reach a finite loss before the real run)"
"${PYTHON:-python}" scripts/train.py --config configs/train_sanity.yaml || die "sanity train"

log "full train"
"${PYTHON:-python}" scripts/train.py --config "$CONFIG" || die "full train"

# Past this point the model exists and is worth keeping, so a failure here is
# reported but does not throw the run away.
log "downloading benchmark"
"${PYTHON:-python}" scripts/download_bea60k.py || echo "BENCHMARK DOWNLOAD FAILED"

log "aspell baseline"
"${PYTHON:-python}" scripts/benchmark_aspell.py || true

log "benchmark"
"${PYTHON:-python}" scripts/benchmark_bea60k.py \
  --model artifacts/model \
  --output reports/bea60k || echo "BENCHMARK FAILED"

log "done"
"${PYTHON:-python}" - <<'PY'
import json
from pathlib import Path
res = json.loads(Path("reports/bea60k/results.json").read_text())
for key in (
    "n_word_errors", "hunspell_top1", "hunspell_oracle_at_slots",
    "model_conditional_accuracy", "model_overall_success", "aspell_top1",
):
    value = res.get(key)
    print(f"{key:32} {value:.4f}" if isinstance(value, float) else f"{key:32} {value}")
PY
