#!/usr/bin/env bash
# Pod entrypoint for the 0.8B-on-a-3090 throughput hill-climb.
#
# The pod is not self-driving this time. A hill-climb decides step N+1 from
# step N's number, so the pod's job is to get a serving stack and the model
# into a ready state and then wait for instructions on a control port:
#
#   https://<pod-id>-8000.proxy.runpod.net/run.log      setup progress
#   https://<pod-id>-8000.proxy.runpod.net/STATUS       one word
#   https://<pod-id>-8000.proxy.runpod.net/READY        appears when prepped
#   https://<pod-id>-8001.proxy.runpod.net/status       the control plane
#
# Port 8001 needs CONTROL_TOKEN (?token=...), because a Runpod proxy URL is
# public. Port 8000 is a read-only directory listing and carries no secrets.
#
# The base image is vllm/vllm-openai, so vLLM, torch and CUDA are already
# installed and matched -- pip-installing vLLM onto a plain pytorch image is
# ten minutes of wheel downloads and a coin-flip on the torch version.
#
# Set at pod creation by scripts/runpod/launch_throughput.py:
#   REPO_URL, REPO_BRANCH, REPO_COMMIT, HF_MODEL, CHOSEN_GPU, GPU_PRICE_USD_HR,
#   CONTROL_TOKEN, QUANT_SCHEME, QUANT_ALGORITHM, QUANT_SAMPLES
#
# It also downloads BEA-60K and rebuilds the M7 held-out splits, so the control
# plane can score a checkpoint as well as time it. BEA-60K is never committed.
#
# Deliberately no `set -u`: this sources the image's profile scripts, which
# reference unset variables, and under `set -u` that aborts before the first
# status write -- indistinguishable from "the pod did nothing".
set -o pipefail

OUT=/workspace/out
mkdir -p "$OUT"
export DEBIAN_FRONTEND=noninteractive
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
for profile in /etc/profile.d/*.sh /root/.bashrc; do
    # shellcheck disable=SC1090
    [ -r "$profile" ] && . "$profile" >/dev/null 2>&1
done
true  # sourcing failures must not decide this script's exit status

PYTHON="$(command -v python3 || echo /usr/bin/python3)"
status() { echo "$1" > "$OUT/STATUS"; echo "== $(date -u +%H:%M:%S) $1"; }

# Serve progress before anything else, so a setup failure is still visible.
pgrep -f "http.server 8000" >/dev/null 2>&1 || \
    nohup "$PYTHON" -m http.server 8000 --directory "$OUT" >/dev/null 2>&1 &
sleep 1
exec > >(tee -a "$OUT/run.log") 2>&1
status "starting"

REPO_DIR=/workspace/spell-corrector
MODEL_DIR=/workspace/models
mkdir -p "$MODEL_DIR"
REPO_URL="${REPO_URL:-https://github.com/ctalau/spell-corrector}"
REPO_BRANCH="${REPO_BRANCH:-main}"
HF_MODEL="${HF_MODEL:-ctalau/qwen35-08b-spell-m7-distill}"
QUANT_SCHEME="${QUANT_SCHEME:-W4A16}"
QUANT_ALGORITHM="${QUANT_ALGORITHM:-gptq}"
QUANT_SAMPLES="${QUANT_SAMPLES:-256}"

echo "gpu:    ${CHOSEN_GPU:-unknown} (\$${GPU_PRICE_USD_HR:-?}/hr)"
nvidia-smi || echo "nvidia-smi failed"
"$PYTHON" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)" \
    || echo "vllm import failed"

status "apt"
apt-get update -qq && apt-get install -y -qq git curl >/dev/null 2>&1 || echo "apt-get failed (continuing)"

status "clone"
rm -rf "$REPO_DIR"
git clone --quiet "$REPO_URL" "$REPO_DIR" || { status "clone-failed"; sleep infinity; }
if [ -n "$REPO_COMMIT" ]; then
    git -C "$REPO_DIR" fetch --quiet origin "$REPO_COMMIT" 2>/dev/null
    git -C "$REPO_DIR" checkout --quiet "$REPO_COMMIT" || git -C "$REPO_DIR" checkout --quiet "$REPO_BRANCH"
else
    git -C "$REPO_DIR" checkout --quiet "$REPO_BRANCH"
fi
echo "commit: $(git -C "$REPO_DIR" rev-parse HEAD)"

# The control plane comes up now, before the slow parts, so its /status and
# /file endpoints can be used to watch the preparation that follows.
status "control-plane"
OUT_DIR="$OUT" REPO_DIR="$REPO_DIR" MODEL_DIR="$MODEL_DIR" \
CONTROL_TOKEN="$CONTROL_TOKEN" CONTROL_PORT=8001 SERVE_PORT=8080 \
QUANT_PYTHON=/workspace/quantvenv/bin/python \
    nohup "$PYTHON" "$REPO_DIR/scripts/runpod/throughput_control.py" \
    >> "$OUT/control.stdout.log" 2>&1 &
sleep 2

status "download-model"
"$PYTHON" - <<PY
from huggingface_hub import snapshot_download
path = snapshot_download("$HF_MODEL", local_dir="$MODEL_DIR/fp16",
                         allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"])
print("downloaded to", path)
PY
if [ ! -f "$MODEL_DIR/fp16/config.json" ]; then status "download-failed"; sleep infinity; fi
du -sh "$MODEL_DIR/fp16"

# llmcompressor must not be allowed to move torch, transformers or vLLM: this
# image's versions are matched to each other and to the driver, and a silent
# torch upgrade would leave the pod with a quantized model and no engine. But
# llmcompressor pins compressed-tensors to the exact patch (0.13.0 wants
# 0.18.0, the image has 0.17.0), and installing it --no-deps just moves the
# failure to an ImportError at quantization time -- which is what the first two
# launches of this pod did.
#
# So: a venv with --system-site-packages. torch, transformers and CUDA are
# inherited (nothing large is downloaded twice), while compressed-tensors and
# llmcompressor are installed *into the venv only*. The quantizer runs under
# that python; vLLM keeps the compressed-tensors it was built against. The
# checkpoint is the handoff between them, and W4A16 pack-quantized is stable
# across these patch versions.
status "install-llmcompressor"
QUANT_PYTHON=/workspace/quantvenv/bin/python
"$PYTHON" -m pip list --format=freeze 2>/dev/null \
    | grep -Ei '^(torch|transformers|vllm|numpy)==' > /workspace/constraints.txt
cat /workspace/constraints.txt
"$PYTHON" -m venv --system-site-packages /workspace/quantvenv
"$QUANT_PYTHON" -m pip install -q --upgrade pip >/dev/null 2>&1
"$QUANT_PYTHON" -m pip install -q --no-deps "llmcompressor==${LLMCOMPRESSOR_VERSION:-0.13.0}" \
    || echo "llmcompressor install failed"
"$QUANT_PYTHON" -m pip install -q --constraint /workspace/constraints.txt \
    "compressed-tensors==${COMPRESSED_TENSORS_VERSION:-0.18.0}" datasets loguru pydantic pynvml \
    || echo "helper install partially failed"
"$QUANT_PYTHON" -c "import llmcompressor, compressed_tensors, datasets, torch; print('venv:', llmcompressor.__version__, compressed_tensors.__version__, torch.__version__)" \
    || echo "llmcompressor does not import in the venv"
"$PYTHON" -c "import vllm, compressed_tensors; print('engine still intact:', vllm.__version__, compressed_tensors.__version__)" \
    || echo "WARNING: the engine's own imports broke"

status "quantize"
QUANT_LOG="$OUT/quantize.log"
if "$QUANT_PYTHON" "$REPO_DIR/scripts/distill/quantize_w4a16.py" \
        --model "$MODEL_DIR/fp16" --output "$MODEL_DIR/w4a16" \
        --scheme "$QUANT_SCHEME" --algorithm "$QUANT_ALGORITHM" --samples "$QUANT_SAMPLES" \
        --dump-modules "$OUT/linear_modules.json" > "$QUANT_LOG" 2>&1; then
    echo "quantization ok"
    du -sh "$MODEL_DIR/w4a16"
else
    status "quantize-failed"
    echo "quantization FAILED -- tail of $QUANT_LOG:"
    tail -40 "$QUANT_LOG"
    echo "the pod stays up: vLLM's in-flight --quantization fp8 path still serves a quantized model"
fi

# BEA-60K, for accuracy measurement only. It is a locked benchmark: the splits
# below are rebuilt with build_data.py's fixed seed and its frozen-100
# reconstruction check, so `test.jsonl` is byte-for-byte the split the Q4_K_M
# student's 86.60% was measured on. Nothing here is committed, and the control
# plane refuses to score `train`/`val`, which the student was trained on.
status "bea-splits"
if "$PYTHON" "$REPO_DIR/scripts/download_bea60k.py" --out-dir "$REPO_DIR/data/bea60k" > "$OUT/bea_download.log" 2>&1; then
    "$PYTHON" "$REPO_DIR/scripts/distill/build_data.py" \
        --bea-dir "$REPO_DIR/data/bea60k" \
        --out-dir "$REPO_DIR/data/distill" \
        --prompt "$REPO_DIR/artifacts/spell_slm_m6/direct_correct_v1.txt" \
        > "$OUT/bea_splits.log" 2>&1 \
        && cp "$REPO_DIR/data/distill/meta.json" "$OUT/split_meta.json" 2>/dev/null \
        || { echo "split build FAILED"; tail -20 "$OUT/bea_splits.log"; }
    wc -l "$REPO_DIR"/data/distill/*.jsonl 2>/dev/null
else
    echo "BEA download FAILED -- accuracy jobs will refuse; throughput jobs are unaffected"
    tail -20 "$OUT/bea_download.log"
fi

"$PYTHON" - <<PY > "$OUT/RUNINFO.json"
import json, os, subprocess
def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""
print(json.dumps({
    "gpu": os.environ.get("CHOSEN_GPU"),
    "gpu_price_usd_hr": os.environ.get("GPU_PRICE_USD_HR"),
    "nvidia_smi": sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
    "vllm": sh("python3 -c 'import vllm;print(vllm.__version__)'"),
    "torch": sh("python3 -c 'import torch;print(torch.__version__)'"),
    "commit": sh("git -C /workspace/spell-corrector rev-parse HEAD"),
    "hf_model": os.environ.get("HF_MODEL"),
    "quant_scheme": os.environ.get("QUANT_SCHEME"),
    "quant_algorithm": os.environ.get("QUANT_ALGORITHM"),
}, indent=2))
PY
cat "$OUT/RUNINFO.json"

status "ready"
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/READY"
echo "control plane: POST /job on port 8001 with the control token"
sleep infinity
