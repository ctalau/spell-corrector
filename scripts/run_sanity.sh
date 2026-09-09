#!/usr/bin/env bash
# Cheap sanity training job. Intended for a 16-24GB CUDA pod after unit tests.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"

echo "== environment =="
command -v nvidia-smi >/dev/null && nvidia-smi || echo "nvidia-smi not found (CPU?)"
"$PYTHON" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("bf16", torch.cuda.is_bf16_supported())
PY

echo "== unit tests =="
"$PYTHON" -m pytest tests/ -q

if [[ ! -f data/processed/train.parquet ]]; then
  echo "data/processed/train.parquet missing. Build data first:"
  echo "  $PYTHON scripts/download_sources.py"
  echo "  $PYTHON scripts/build_training_data.py"
  exit 1
fi

echo "== sanity train =="
"$PYTHON" scripts/train.py --config configs/train_sanity.yaml

echo "== validation smoke =="
"$PYTHON" scripts/evaluate_validation.py \
  --config configs/train_sanity.yaml \
  --model artifacts/model \
  --max-examples 128

echo "sanity complete. See artifacts/train_summary.json and reports/sanity/"
