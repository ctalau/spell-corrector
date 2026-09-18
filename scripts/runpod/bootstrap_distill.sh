#!/usr/bin/env bash
# Pod entrypoint for M7: distil the 2B Q4 spelling corrector into the 0.8B Q4
# student on BEA-60K, then serve the student as Q4_K_M GGUF and score it.
#
# Runs as the container entrypoint (see bootstrap.sh for why) and reports over
# the Runpod HTTP proxy:
#
#   https://<pod-id>-8000.proxy.runpod.net/run.log
#   https://<pod-id>-8000.proxy.runpod.net/STATUS
#   https://<pod-id>-8000.proxy.runpod.net/PROGRESS        (student metrics.jsonl)
#   https://<pod-id>-8000.proxy.runpod.net/DONE
#   https://<pod-id>-8000.proxy.runpod.net/artifacts/...
#
# Set at pod creation by scripts/runpod/launch.py:
#   REPO_URL, REPO_BRANCH, REPO_COMMIT
#   EPOCHS, MICRO_BATCH, GRAD_ACCUM, KD_ALPHA, KD_T, LORA_R, TOPK, MAX_TRAIN
#   EVAL_EVERY, TARGET_ACC, LLAMA_CPP_REF, SKIP_GGUF
#
# No `set -u`: the image's profile scripts reference unset variables.
set -o pipefail

OUT=/workspace/out
mkdir -p "$OUT"
export DEBIAN_FRONTEND=noninteractive
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
for extra in /opt/conda/bin /usr/local/nvidia/bin /usr/local/cuda/bin /venv/bin /workspace/venv/bin; do
    [ -d "$extra" ] && export PATH="$extra:$PATH"
done
for profile in /etc/profile.d/*.sh /root/.bashrc; do
    # shellcheck disable=SC1090
    [ -r "$profile" ] && . "$profile" >/dev/null 2>&1
done
true

PYTHON="$(command -v python3 || command -v python || echo /usr/bin/python3)"
status() { echo "$1" > "$OUT/STATUS"; echo "== STATUS: $1"; }
status "starting"

nohup "$PYTHON" -m http.server 8000 --directory "$OUT" >/dev/null 2>&1 &
sleep 1
exec > >(tee -a "$OUT/run.log") 2>&1

REPO_DIR=/workspace/spell-corrector
LLAMA_DIR=/workspace/llama.cpp
WORK=/workspace/m7
STUDENT_OUT="$WORK/student"
TEACHER_OUT="$WORK/teacher"
RESULTS="$WORK/results"
mkdir -p "$WORK" "$RESULTS"

EPOCHS="${EPOCHS:-3}"
MICRO_BATCH="${MICRO_BATCH:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
KD_ALPHA="${KD_ALPHA:-0.5}"
KD_T="${KD_T:-2.0}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
TOPK="${TOPK:-64}"
MAX_TRAIN="${MAX_TRAIN:-0}"
EVAL_EVERY="${EVAL_EVERY:-250}"
EVAL_LIMIT="${EVAL_LIMIT:-500}"
TARGET_ACC="${TARGET_ACC:-0}"
LR="${LR:-2e-4}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-master}"
SKIP_GGUF="${SKIP_GGUF:-0}"
TEACHER_BATCH="${TEACHER_BATCH:-24}"
M6_ADAPTER="$REPO_DIR/artifacts/spell_slm_m6/qwen35_2b_direct_qlora"

echo "bootstrap starting $(date -u +%FT%TZ)"
echo "python=$PYTHON ($($PYTHON -V 2>&1))"
echo "repo=${REPO_URL:-unset} branch=${REPO_BRANCH:-unset} commit=${REPO_COMMIT:-unset}"
echo "epochs=$EPOCHS micro=$MICRO_BATCH accum=$GRAD_ACCUM kd_alpha=$KD_ALPHA kd_T=$KD_T lora_r=$LORA_R topk=$TOPK"
nvidia-smi || echo "nvidia-smi unavailable"
echo "vcpu=$(nproc) ram=$(free -g | awk '/^Mem:/{print $2}')GB disk=$(df -h /workspace | tail -1)"

sync_artifacts() {
    mkdir -p "$OUT/artifacts"
    cp -r "$RESULTS" "$OUT/artifacts/" 2>/dev/null
    mkdir -p "$OUT/artifacts/student"
    cp "$STUDENT_OUT"/{STATUS,metrics.jsonl,run_meta.json,train_summary.json} "$OUT/artifacts/student/" 2>/dev/null
    cp -r "$STUDENT_OUT/best" "$OUT/artifacts/student/" 2>/dev/null
    cp "$TEACHER_OUT/meta.json" "$OUT/artifacts/teacher_meta.json" 2>/dev/null
    cp "$REPO_DIR/data/distill/meta.json" "$OUT/artifacts/data_meta.json" 2>/dev/null
    cp "$STUDENT_OUT/metrics.jsonl" "$OUT/PROGRESS" 2>/dev/null
    cp "$STUDENT_OUT/STATUS" "$OUT/TRAIN_STATUS" 2>/dev/null
    true
}

fail() {
    status "FAILED: $1"
    echo "FAILED: $1"
    sync_artifacts
    touch "$OUT/DONE"
    # Stay up so the log stays readable; the orchestrator terminates the pod.
    sleep infinity
}

# Background artifact sync: a run that dies mid-training still hands back
# everything written so far, without waiting for a step boundary.
( while true; do sync_artifacts; sleep 60; done ) >/dev/null 2>&1 &

# ---------------------------------------------------------------- system deps
status "apt: build deps"
apt-get update -qq || fail "apt-get update"
apt-get install -y -qq --no-install-recommends \
    git git-lfs curl ca-certificates build-essential cmake pkg-config \
    libcurl4-openssl-dev || fail "apt build deps"
git lfs install --skip-repo || true

pip_install() {
    "$PYTHON" -m pip install -q "$@" && return 0
    "$PYTHON" -m pip install -q --break-system-packages "$@"
}

# `nvidia-smi` working proves nothing about CUDA inside the container: a host
# whose nvidia_uvm module never got loaded shows a healthy GPU and then fails
# every cudaInit with "CUDA unknown error". One community host did exactly that
# and burned ten minutes of pip before the failure surfaced, so the probe now
# runs before any install -- and attempts the standard repair first.
cuda_ok() {
    "$PYTHON" -c 'import sys
try:
    import torch
except Exception:
    sys.exit(3)
sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null
}

ensure_cuda() {
    cuda_ok && return 0
    echo "CUDA not visible to torch; attempting uvm repair"
    ls -la /dev/nvidia* 2>&1 | head -20
    modprobe nvidia_uvm 2>&1 | head -3
    nvidia-modprobe -u -c=0 2>&1 | head -3
    [ -e /dev/nvidia-uvm ] || mknod -m 666 /dev/nvidia-uvm c 243 0 2>&1 | head -3
    [ -e /dev/nvidia-uvm-tools ] || mknod -m 666 /dev/nvidia-uvm-tools c 243 1 2>&1 | head -3
    cuda_ok
}

status "probe: CUDA visible to the image's torch"
if ! ensure_cuda; then
    fail "CUDA unusable on this host (nvidia-smi works, torch cannot init) -- relaunch elsewhere"
fi
"$PYTHON" -c 'import torch;print("preinstalled torch", torch.__version__, torch.cuda.get_device_name(0))'

# ---------------------------------------------------------------------- repo
status "clone repo"
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 -b "${REPO_BRANCH:-main}" "${REPO_URL}" "$REPO_DIR" \
    || fail "git clone"
if [ -n "${REPO_COMMIT:-}" ]; then
    git -C "$REPO_DIR" fetch --depth 1 origin "$REPO_COMMIT" \
        && git -C "$REPO_DIR" checkout -q "$REPO_COMMIT"
fi
echo "repo at $(git -C "$REPO_DIR" rev-parse HEAD)"

status "git lfs: teacher adapters"
( cd "$REPO_DIR" && git lfs pull --include "artifacts/spell_slm_m6/qwen35_2b_direct_qlora/*" ) \
    || fail "git lfs pull (teacher adapters)"
ls -la "$M6_ADAPTER"
ADAPTER_BYTES=$(stat -c%s "$M6_ADAPTER/adapter_model.safetensors" 2>/dev/null || echo 0)
[ "$ADAPTER_BYTES" -gt 1000000 ] || fail "teacher adapter still an LFS pointer ($ADAPTER_BYTES bytes)"

# --------------------------------------------------------------------- python
# The image ships torch 2.4.0; M6's working stack was torch 2.5.1+cu124 with a
# transformers that knows qwen3_5. Same choice here -- a cu128 wheel silently
# falls back to CPU on these CUDA 12.4 community hosts.
status "pip: torch + transformers stack"
pip_install --upgrade pip setuptools wheel || true
TORCH_BEFORE="$("$PYTHON" -c 'import torch;print(torch.__version__)' 2>/dev/null)"
TORCH_MAJMIN="$("$PYTHON" -c 'import torch;v=torch.__version__.split(".");print(int(v[0])*100+int(v[1]))' 2>/dev/null || echo 0)"
if [ "${TORCH_MAJMIN:-0}" -lt 205 ]; then
    echo "upgrading torch from ${TORCH_BEFORE:-none} to 2.5.1+cu124"
    pip_install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 || fail "pip torch"
    if ! cuda_ok; then
        echo "torch 2.5.1 cannot see the GPU; reverting to the image's ${TORCH_BEFORE}"
        pip_install "torch==${TORCH_BEFORE%%+*}" --index-url https://download.pytorch.org/whl/cu124 \
            || fail "torch revert"
        ensure_cuda || fail "CUDA broken after torch revert"
    fi
else
    echo "image torch ${TORCH_BEFORE} is new enough; leaving it alone"
fi
pip_install "numpy<2.3" safetensors sentencepiece protobuf accelerate datasets pyyaml requests \
    || fail "pip base deps"
pip_install "git+https://github.com/huggingface/transformers" || fail "pip transformers"
pip_install "peft>=0.14" "bitsandbytes>=0.45" || fail "pip peft/bnb"
pip_install gguf || true

"$PYTHON" - <<'PY' || fail "torch/cuda check"
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__)
assert torch.cuda.is_available(), "CUDA not available -- wrong wheel for this host"
print("gpu", torch.cuda.get_device_name(0))
PY

# Optional fused kernels. Every one of these is a speedup, none is required:
# the run must not die because a wheel was not published for this exact
# torch/python/ABI triple, so each install is attempted and then probed.
status "pip: optional fused kernels"
( pip_install liger-kernel || echo "liger-kernel install failed" ) &
LIGER_PID=$!
( pip_install flash-linear-attention || echo "fla install failed" ) &
FLA_PID=$!
(
  # flash-attn only from a prebuilt wheel: a source build is over an hour.
  FA_VER=2.7.4.post1
  PYTAG=$("$PYTHON" -c 'import sys;print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')
  URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FA_VER}/flash_attn-${FA_VER}+cu12torch2.5cxx11abiFALSE-${PYTAG}-${PYTAG}-linux_x86_64.whl"
  echo "flash-attn wheel: $URL"
  # pip parses the *filename*, so the download has to keep the wheel's own name
  # (a /tmp/flash_attn.whl is rejected as "wrong number of parts").
  WHEEL="/tmp/$(basename "$URL")"
  curl -fsSL --max-time 900 -o "$WHEEL" "$URL" && pip_install "$WHEEL" \
      || echo "flash-attn wheel unavailable; SDPA will be used"
) &
FA_PID=$!

# ------------------------------------------------------------------ BEA + data
status "download BEA-60K"
( cd "$REPO_DIR" && "$PYTHON" scripts/download_bea60k.py ) || fail "download bea60k"
status "build distillation splits"
BUILD_ARGS=""
[ "$MAX_TRAIN" != "0" ] && BUILD_ARGS="--max-train $MAX_TRAIN"
( cd "$REPO_DIR" && "$PYTHON" scripts/distill/build_data.py $BUILD_ARGS ) || fail "build_data"
cp "$REPO_DIR/data/distill/meta.json" "$RESULTS/data_meta.json"

wait $LIGER_PID $FLA_PID $FA_PID
status "kernel probe"
"$PYTHON" - <<'PY' | tee "$RESULTS/kernels.json"
import json
info = {}
for name in ("flash_attn", "liger_kernel", "fla", "causal_conv1d", "triton"):
    try:
        mod = __import__(name)
        info[name] = str(getattr(mod, "__version__", "ok"))
    except Exception as exc:
        info[name] = f"missing ({type(exc).__name__})"
print(json.dumps(info, indent=2))
PY
sync_artifacts

# --------------------------------------------------------------------- smoke
# Ten minutes of smoke beats an hour of teacher forward followed by a typo in
# the training step.
status "smoke: teacher dump (256 rows)"
( cd "$REPO_DIR" && "$PYTHON" scripts/distill/dump_teacher_logits.py \
    --adapter-dir "$M6_ADAPTER" --out-dir "$WORK/smoke_teacher" \
    --limit 256 --batch-size 8 --topk "$TOPK" ) || fail "smoke teacher dump"

status "smoke: student train (5 steps)"
( cd "$REPO_DIR" && "$PYTHON" scripts/distill/train_student.py \
    --teacher-dir "$WORK/smoke_teacher" --out-dir "$WORK/smoke_student" \
    --max-steps 5 --micro-batch 4 --grad-accum 2 --eval-every 5 --eval-limit 32 \
    --log-every 1 --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" ) || fail "smoke train"
cp "$WORK/smoke_student/metrics.jsonl" "$RESULTS/smoke_metrics.jsonl" 2>/dev/null
rm -rf "$WORK/smoke_teacher"
sync_artifacts

# ------------------------------------------------------------------- teacher
status "teacher: dumping top-$TOPK logits over the training split"
( cd "$REPO_DIR" && "$PYTHON" scripts/distill/dump_teacher_logits.py \
    --adapter-dir "$M6_ADAPTER" --out-dir "$TEACHER_OUT" \
    --batch-size "$TEACHER_BATCH" --topk "$TOPK" ) || fail "teacher dump"
cp "$TEACHER_OUT/meta.json" "$RESULTS/teacher_meta.json"
sync_artifacts

# ------------------------------------------------------------------- student
status "student: distillation training"
TRAIN_EXTRA=""
[ "$TARGET_ACC" != "0" ] && TRAIN_EXTRA="--stop-at-acc $TARGET_ACC"
( cd "$REPO_DIR" && "$PYTHON" scripts/distill/train_student.py \
    --teacher-dir "$TEACHER_OUT" --out-dir "$STUDENT_OUT" \
    --epochs "$EPOCHS" --micro-batch "$MICRO_BATCH" --grad-accum "$GRAD_ACCUM" \
    --lr "$LR" --kd-alpha "$KD_ALPHA" --kd-temperature "$KD_T" \
    --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" \
    --eval-every "$EVAL_EVERY" --eval-limit "$EVAL_LIMIT" $TRAIN_EXTRA ) || fail "student training"
cp "$STUDENT_OUT/train_summary.json" "$RESULTS/train_summary.json" 2>/dev/null
sync_artifacts

# ---------------------------------------------------------------- GPU scoring
BEST="$STUDENT_OUT/best"
[ -d "$BEST" ] || BEST="$STUDENT_OUT/last"
for split in frozen_100 test; do
    status "eval (GPU, NF4): $split"
    ( cd "$REPO_DIR" && "$PYTHON" scripts/distill/eval_model.py \
        --split "data/distill/$split.jsonl" --adapter "$BEST" \
        --label "m7_student_nf4_$split" \
        --out-metrics "$RESULTS/metrics_student_nf4_$split.json" \
        --out-predictions "$RESULTS/predictions_student_nf4_$split.jsonl" ) \
        || fail "eval student $split"
    sync_artifacts
done

# Teacher on the same rows, so the distillation gap is measured rather than
# quoted from the milestone-6 write-up.
for split in frozen_100 test; do
    status "eval (GPU, NF4): teacher on $split"
    ( cd "$REPO_DIR" && "$PYTHON" scripts/distill/eval_model.py \
        --split "data/distill/$split.jsonl" --adapter "$M6_ADAPTER" \
        --base-model Qwen/Qwen3.5-2B --revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
        --label "m6_teacher_nf4_$split" \
        --out-metrics "$RESULTS/metrics_teacher_nf4_$split.json" \
        --out-predictions "$RESULTS/predictions_teacher_nf4_$split.jsonl" ) \
        || echo "teacher eval on $split failed (non-fatal)"
    sync_artifacts
done

if [ "$SKIP_GGUF" = "1" ]; then
    status "DONE (GGUF skipped)"
    sync_artifacts
    touch "$OUT/DONE"
    sleep infinity
fi

# ------------------------------------------------------------------ GGUF path
status "merge student LoRA into fp16 base"
( cd "$REPO_DIR" && "$PYTHON" scripts/distill/merge_adapter.py \
    --adapter "$BEST" --out-dir "$WORK/student_merged" ) || fail "merge adapter"

status "build llama.cpp ($LLAMA_CPP_REF)"
git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR" || fail "clone llama.cpp"
if [ "$LLAMA_CPP_REF" != "master" ]; then
    git -C "$LLAMA_DIR" fetch --tags --depth 50 origin || true
    git -C "$LLAMA_DIR" checkout "$LLAMA_CPP_REF" || fail "llama.cpp checkout"
fi
echo "llama.cpp at $(git -C "$LLAMA_DIR" rev-parse HEAD)"
pip_install -r "$LLAMA_DIR/requirements/requirements-convert_hf_to_gguf.txt" || true
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF \
    -DCMAKE_BUILD_TYPE=Release || fail "cmake configure"
cmake --build "$LLAMA_DIR/build" -j "$(nproc)" --target llama-quantize llama-server \
    || fail "cmake build"

status "convert to GGUF F16"
"$PYTHON" "$LLAMA_DIR/convert_hf_to_gguf.py" "$WORK/student_merged" \
    --outfile "$WORK/student-F16.gguf" --outtype f16 || fail "convert_hf_to_gguf"

status "quantize Q4_K_M"
"$LLAMA_DIR/build/bin/llama-quantize" "$WORK/student-F16.gguf" \
    "$WORK/qwen35_0_8b_distilled_q4-Q4_K_M.gguf" Q4_K_M || fail "llama-quantize"
GGUF="$WORK/qwen35_0_8b_distilled_q4-Q4_K_M.gguf"

status "check GGUF metadata (MTP / block_count)"
if ! "$PYTHON" "$REPO_DIR/scripts/distill/check_gguf.py" "$GGUF" \
        --report "$RESULTS/gguf_metadata.json"; then
    echo "metadata inconsistent; repairing"
    "$PYTHON" "$REPO_DIR/scripts/distill/check_gguf.py" "$GGUF" \
        --fix "$WORK/student-fixed.gguf" --report "$RESULTS/gguf_metadata_before_fix.json" \
        || fail "gguf repair"
    mv "$WORK/student-fixed.gguf" "$GGUF"
    "$PYTHON" "$REPO_DIR/scripts/distill/check_gguf.py" "$GGUF" \
        --report "$RESULTS/gguf_metadata.json" || fail "gguf still inconsistent after repair"
fi
ls -la "$GGUF"
cp "$GGUF" "$OUT/artifacts/" 2>/dev/null

status "serve GGUF on CPU and score"
nohup "$LLAMA_DIR/build/bin/llama-server" -m "$GGUF" -t 8 -c 1024 --jinja \
    --host 127.0.0.1 --port 8080 > "$OUT/llama_server.log" 2>&1 &
for split in frozen_100 test; do
    ( cd "$REPO_DIR" && "$PYTHON" scripts/distill/eval_gguf.py \
        --split "data/distill/$split.jsonl" --label "m7_student_q4km_$split" \
        --model-path "$(basename "$GGUF")" \
        --out-metrics "$RESULTS/metrics_student_q4km_$split.json" \
        --out-predictions "$RESULTS/predictions_student_q4km_$split.jsonl" ) \
        || fail "gguf eval $split"
    sync_artifacts
done

status "DONE"
sync_artifacts
touch "$OUT/DONE"
sleep infinity
