#!/usr/bin/env bash
# Full experiment on the pod: data -> sanity -> train -> benchmark.
#
# Run under nohup/tmux; the training leg is hours long:
#   nohup bash scripts/runpod/run_experiment.sh > /workspace/run.log 2>&1 &
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/spell-corrector}"
TARGET_TRAIN="${TARGET_TRAIN:-4000000}"
TARGET_VALID="${TARGET_VALID:-60000}"
CONFIG="${CONFIG:-configs/train_full.yaml}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR"

log() { printf '\n########## %s (t=%ss) ##########\n' "$1" "$SECONDS"; }

# Sample GPU memory alongside training so the report can quote a real peak.
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
  --format=csv -l 5 > artifacts/vram_sampler.csv 2>/dev/null &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null || true' EXIT

log "unit tests"
python -m pytest tests/ -q

log "typo generator calibration (public misspelling list)"
python scripts/calibrate_typo_model.py

log "building training data (target ${TARGET_TRAIN})"
python scripts/build_training_data.py \
  --target-train "$TARGET_TRAIN" \
  --target-valid "$TARGET_VALID" \
  --seed 1337

log "sanity train (must reach a finite loss before the real run)"
python scripts/train.py --config configs/train_sanity.yaml

log "full train"
python scripts/train.py --config "$CONFIG"

log "downloading benchmark"
python scripts/download_bea60k.py

log "aspell baseline"
python scripts/benchmark_aspell.py || true

log "benchmark"
python scripts/benchmark_bea60k.py \
  --model artifacts/model \
  --output reports/bea60k

log "done"
python - <<'PY'
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
