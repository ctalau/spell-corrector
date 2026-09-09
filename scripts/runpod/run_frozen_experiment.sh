#!/usr/bin/env bash
# Frozen ModernBERT encoder + selector-head experiment.
#
# Smoke-cache 10k, full 200k + D-pair cache, train H1 then H2 then H3.
# Periodic 30-minute checkpoints evaluate a fixed BEA-60K 1k subset; an arm
# stops when overall and conditional accuracy both fail to improve by >1 pp.
# Does not run the full BEA benchmark. Serve artifacts over HTTP via bootstrap.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/spell-corrector}"
TARGET_TRAIN="${TARGET_TRAIN:-400000}"
TARGET_VALID="${TARGET_VALID:-40000}"
CONFIG="${CONFIG:-configs/train_frozen_modernbert.yaml}"
if [ -z "${PYTHON:-}" ] && [ -x /workspace/venv/bin/python ]; then
  PYTHON=/workspace/venv/bin/python
fi
PYTHON="${PYTHON:-python}"
export PYTHON
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

log() { printf '\n########## %s (t=%ss) ##########\n' "$1" "$SECONDS"; }
die() { echo "FROZEN EXPERIMENT FAILED: $1"; exit 1; }

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
print("device", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())
PY

log "frozen extras (transformers>=4.48,<5 so ModernBERT imports on torch 2.4)"
"${PYTHON:-python}" -m pip install -q "transformers>=4.48,<5" tokenizers huggingface_hub accelerate \
  || die "transformers install"

log "transformers / AutoModel import check"
"${PYTHON:-python}" - <<'PY' || die "transformers cannot import torch/AutoModel"
import torch
import transformers
from transformers import AutoModel, AutoTokenizer
from transformers.models.modernbert.modeling_modernbert import ModernBertModel

print("torch", torch.__version__)
print("transformers", transformers.__version__)
major = int(transformers.__version__.split(".", 1)[0])
assert major < 5, (
    f"transformers 5.x requires PyTorch >= 2.5; this image has {torch.__version__}. "
    "Pin transformers>=4.48,<5 (ModernBERT landed in 4.48)."
)
assert AutoModel is not None and AutoTokenizer is not None
assert ModernBertModel is not None
print("AutoModel / ModernBertModel import ok")
PY

log "unit tests"
"${PYTHON:-python}" -m pytest tests/ -q || die "unit tests"

log "building training data if needed (target ${TARGET_TRAIN} / ${TARGET_VALID})"
if [ ! -f data/processed/train.parquet ] || [ ! -f data/processed/validation.parquet ]; then
  "${PYTHON:-python}" scripts/download_sources.py || die "download sources"
  "${PYTHON:-python}" scripts/build_training_data.py \
    --target-train "$TARGET_TRAIN" \
    --target-valid "$TARGET_VALID" \
    --seed 1337 || die "data build"
else
  echo "reusing data/processed/{train,validation}.parquet"
fi

log "BEA-60K download (1k monitoring subset only; not a full benchmark)"
"${PYTHON:-python}" scripts/download_bea60k.py || die "BEA download"

log "smoke cache 10000"
"${PYTHON:-python}" scripts/cache_frozen_features.py \
  --config "$CONFIG" --smoke-examples 10000 || die "smoke cache"

log "full 200k + D-pair cache + BEA-1k features"
"${PYTHON:-python}" scripts/cache_frozen_features.py \
  --config "$CONFIG" --bea-limit 1000 || die "full cache"

train_arm() {
  local arm="$1"
  log "train H ${arm}"
  if "${PYTHON:-python}" scripts/train_frozen_selector.py --config "$CONFIG" --arm "$arm"; then
    return 0
  fi
  local rc=$?
  if [ "$arm" = "mlp" ] && [ "$rc" -eq 2 ]; then
    echo "H3 unstable (non-finite loss); one predeclared retry at 3e-4"
    "${PYTHON:-python}" scripts/train_frozen_selector.py --config "$CONFIG" --arm mlp --lr 3e-4 \
      || die "H3 retry"
    return 0
  fi
  die "train ${arm} rc=${rc}"
}

train_arm scalar
train_arm linear
train_arm mlp

log "evaluate D-pair (H0 + H1/H2/H3)"
mkdir -p artifacts/frozen/eval
"${PYTHON:-python}" scripts/evaluate_frozen_selector.py \
  --config "$CONFIG" --arm hunspell --split d-pair \
  --output artifacts/frozen/eval/h0-dpair || die "eval H0"
for arm in scalar linear mlp; do
  "${PYTHON:-python}" scripts/evaluate_frozen_selector.py \
    --config "$CONFIG" --arm "$arm" --split d-pair \
    --checkpoint "artifacts/frozen/heads/${arm}" \
    --output "artifacts/frozen/eval/${arm}-dpair" || echo "eval ${arm} failed"
done

log "BEA-1k of the selected H3 checkpoint (not full BEA)"
"${PYTHON:-python}" scripts/evaluate_frozen_selector.py \
  --config "$CONFIG" --split bea --bea-limit 1000 \
  --checkpoint artifacts/frozen/heads/mlp \
  --output artifacts/frozen/eval/mlp-bea1k || echo "BEA-1k eval failed"

log "done"
"${PYTHON:-python}" - <<'PY'
import json
from pathlib import Path
root = Path("artifacts/frozen/eval")
for name in ("h0-dpair", "scalar-dpair", "linear-dpair", "mlp-dpair", "mlp-bea1k"):
    path = root / name / "metrics.json"
    if not path.is_file():
        print(f"{name:16} MISSING")
        continue
    m = json.loads(path.read_text())
    print(
        f"{name:16} overall={m.get('overall_accuracy')} "
        f"cond={m.get('conditional_accuracy')} n={m.get('n')}"
    )
PY
