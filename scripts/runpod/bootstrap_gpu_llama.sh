#!/usr/bin/env bash
# Pod entrypoint for the GPU llama.cpp track: build llama.cpp with CUDA, serve
# Google's QAT q4_0 gemma-4-E2B GGUF with full GPU offload, and measure it.
#
# Why a GPU pod for a model that already runs on the local box: the CPU track
# (reports/EXPERIMENT_LLM_JUDGE_CPU.md) is bandwidth-bound and answers one
# request at a time. The question here is a serving question -- what does a
# correction cost once the weights are GPU-resident and requests are batched --
# so the run produces prefill/decode throughput, a concurrency sweep and a
# $/1,000-corrections table, not just an accuracy number.
#
# Runs as the container's *entrypoint* (see scripts/runpod/bootstrap.sh for the
# reasons: the controlling sandbox has outbound HTTPS only, and the base image's
# init swallows dockerStartCmd). Everything is reported over the HTTP proxy:
#
#   https://<pod-id>-8000.proxy.runpod.net/run.log
#   https://<pod-id>-8000.proxy.runpod.net/STATUS
#   https://<pod-id>-8000.proxy.runpod.net/DONE
#   https://<pod-id>-8000.proxy.runpod.net/RUNINFO.json
#   https://<pod-id>-8000.proxy.runpod.net/llama_server.log
#   https://<pod-id>-8000.proxy.runpod.net/artifacts/reports/gpu_llama/...
#
# Set at pod creation by scripts/runpod/launch_gpu_llama.py:
#   REPO_URL, REPO_BRANCH, REPO_COMMIT
#   CHOSEN_GPU, GPU_PRICE_USD_HR, GPU_ATTEMPT_LOG, CUDA_ARCHS
#   LLAMA_CPP_REF, GGUF_REPO, GGUF_FILE, CTX_PER_SLOT, N_PARALLEL, N_SAMPLES
#
# Deliberately no `set -u`: this script sources the image's profile scripts,
# which routinely reference unset variables, and under `set -u` that aborts
# before the first status write -- indistinguishable from "the pod did nothing".
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
true  # sourcing failures must not decide this script's exit status

PYTHON="$(command -v python3 || command -v python || echo /usr/bin/python3)"

status() { echo "$1" > "$OUT/STATUS"; }
status "starting"

# Serve progress before anything else, so a failure in setup is still visible.
nohup "$PYTHON" -m http.server 8000 --directory "$OUT" >/dev/null 2>&1 &
sleep 1

exec > >(tee -a "$OUT/run.log") 2>&1

REPO_DIR=/workspace/spell-corrector
LLAMA_DIR=/workspace/llama.cpp
MODEL_DIR=/workspace/models
LLAMA_LOG="$OUT/llama_server.log"
LLAMA_PORT="${LLAMA_PORT:-8080}"
LLAMA_BASE_URL="http://127.0.0.1:${LLAMA_PORT}"
MODEL_ALIAS="${MODEL_ALIAS:-gemma-4-e2b-q4}"

CHOSEN_GPU="${CHOSEN_GPU:-unknown}"
GPU_PRICE_USD_HR="${GPU_PRICE_USD_HR:-}"
GPU_ATTEMPT_LOG="${GPU_ATTEMPT_LOG:-[]}"
CUDA_ARCHS="${CUDA_ARCHS:-86;89}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-b6390}"
GGUF_REPO="${GGUF_REPO:-google/gemma-4-E2B-it-qat-q4_0-gguf}"
GGUF_FILE="${GGUF_FILE:-gemma-4-E2B_q4_0-it.gguf}"
CTX_PER_SLOT="${CTX_PER_SLOT:-1024}"
N_PARALLEL="${N_PARALLEL:-32}"
N_SAMPLES="${N_SAMPLES:-100}"
# llama-server's -c is the TOTAL KV context shared by every --parallel slot.
N_CTX_TOTAL=$(( CTX_PER_SLOT * N_PARALLEL ))

# write_runinfo reads some of these from os.environ rather than interpolating
# them (JSON with embedded quotes does not survive shell interpolation), so
# they have to be exported and not merely set.
export CHOSEN_GPU GPU_PRICE_USD_HR GPU_ATTEMPT_LOG CUDA_ARCHS REPO_URL REPO_BRANCH REPO_COMMIT

echo "bootstrap starting $(date -u +%FT%TZ)"
echo "python=$PYTHON ($($PYTHON -V 2>&1))"
echo "repo=${REPO_URL:-unset} branch=${REPO_BRANCH:-unset} commit=${REPO_COMMIT:-unset}"
echo "gpu=$CHOSEN_GPU price=${GPU_PRICE_USD_HR:-unknown}/hr cuda_archs=$CUDA_ARCHS"
echo "llama.cpp=$LLAMA_CPP_REF gguf=$GGUF_REPO/$GGUF_FILE"
echo "ctx_per_slot=$CTX_PER_SLOT parallel=$N_PARALLEL total_ctx=$N_CTX_TOTAL"
nvidia-smi || echo "nvidia-smi unavailable"
echo "vcpu=$(nproc) ram=$(free -g | awk '/^Mem:/{print $2}')GB"

# RUNINFO.json is written early and rewritten as facts arrive, so that a run
# that dies in the build still says which GPU it was on and what it tried.
DSPY_STATUS="not started"
JUDGE_STATUS="not started"
LLAMA_STARTUP_SECONDS=""
LLAMA_SERVER_CMD=""
write_runinfo() {
    "$PYTHON" - "$OUT/RUNINFO.json" <<PY
import json
import os
import subprocess
import sys

def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return None

try:
    attempts = json.loads(os.environ.get("GPU_ATTEMPT_LOG") or "[]")
except json.JSONDecodeError:
    attempts = []

price = (os.environ.get("GPU_PRICE_USD_HR") or "").strip()
payload = {
    "chosen_gpu": "${CHOSEN_GPU}",
    "cost_per_hr": float(price) if price else None,
    "gpu_attempts": attempts,
    "cuda_archs": "${CUDA_ARCHS}",
    "repo_commit": os.environ.get("REPO_COMMIT"),
    "repo_branch": os.environ.get("REPO_BRANCH"),
    "llama_cpp_ref": "${LLAMA_CPP_REF}",
    "llama_cpp_commit": sh("git", "-C", "${LLAMA_DIR}", "rev-parse", "HEAD"),
    "gguf_repo": "${GGUF_REPO}",
    "gguf_file": "${GGUF_FILE}",
    "ctx_per_slot": ${CTX_PER_SLOT},
    "n_parallel": ${N_PARALLEL},
    "n_ctx_total": ${N_CTX_TOTAL},
    "llama_server_cmd": "${LLAMA_SERVER_CMD}",
    "llama_server_startup_seconds": float("${LLAMA_STARTUP_SECONDS}") if "${LLAMA_STARTUP_SECONDS}" else None,
    "nvidia_smi": sh("nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
                     "--format=csv,noheader"),
    "judge_status": "${JUDGE_STATUS}",
    "dspy_status": "${DSPY_STATUS}",
    "status": open("$OUT/STATUS").read().strip() if os.path.exists("$OUT/STATUS") else None,
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PY
}
write_runinfo

# Artifacts are copied out after every step rather than once at the end: a run
# that dies in step 3 should still hand back steps 1 and 2.
sync_artifacts() {
    mkdir -p "$OUT/artifacts/reports"
    cp -r "$REPO_DIR/reports/gpu_llama" "$OUT/artifacts/reports/" 2>/dev/null
    write_runinfo
    true
}

fail() {
    status "FAILED: $1"
    echo "FAILED: $1"
    sync_artifacts
    touch "$OUT/DONE"
    # Stay up: the HTTP server is the only way to read the log, and the
    # orchestrator terminates the pod explicitly.
    sleep infinity
}

# ---------------------------------------------------------------- system deps
status "apt: build deps"
apt-get update -qq || fail "apt-get update"
apt-get install -y -qq --no-install-recommends \
    git curl ca-certificates build-essential cmake pkg-config \
    libcurl4-openssl-dev \
    hunspell libhunspell-dev hunspell-en-us \
    || fail "apt build deps"

# Newer images mark the interpreter externally-managed (PEP 668). There is no
# other consumer of this container, so overriding that is correct rather than
# wrapping a handful of packages in a venv.
pip_install() {
    "$PYTHON" -m pip install -q "$@" && return 0
    "$PYTHON" -m pip install -q --break-system-packages "$@"
}

# hunspell's Python binding is the one dependency whose install differs per
# interpreter version: <=3.11 builds from source under the setuptools<60 patch,
# while 3.12 has no distutils for that patch to apply to and needs the distro
# package instead (CLAUDE.md). Handle both -- the image is chosen, not
# guaranteed, and a wrong guess here costs the whole run.
status "hunspell python binding"
install_hunspell() {
    if "$PYTHON" -c 'import hunspell' 2>/dev/null; then
        echo "hunspell already importable"
        return 0
    fi
    local pyver
    pyver="$("$PYTHON" -c 'import sys; print("%d%02d" % sys.version_info[:2])')"
    if [ "$pyver" -lt 312 ]; then
        if pip_install "setuptools<60" wheel cython \
            && pip_install --no-build-isolation hunspell==0.5.5 \
            && "$PYTHON" -c 'import hunspell' 2>/dev/null; then
            return 0
        fi
        echo "pip hunspell build failed on python $pyver; falling back to apt"
    fi
    apt-get install -y -qq --no-install-recommends python3-hunspell || return 1
    if "$PYTHON" -c 'import hunspell' 2>/dev/null; then
        return 0
    fi
    # apt installs into the distro python's dist-packages, which an image
    # python under /opt/conda does not search. Add it explicitly rather than
    # letting the import error surface 20 minutes into the run.
    if [ -d /usr/lib/python3/dist-packages ]; then
        export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}/usr/lib/python3/dist-packages"
        "$PYTHON" -c 'import hunspell' 2>/dev/null && return 0
    fi
    return 1
}
install_hunspell || fail "hunspell python binding"
echo "hunspell ok: $("$PYTHON" -c 'import hunspell; print(hunspell.__file__)')"

status "pip deps"
pip_install --upgrade pip >/dev/null 2>&1
pip_install numpy requests huggingface_hub pytest matplotlib || fail "pip deps"

# ---------------------------------------------------------------------- clone
status "cloning"
GIT_LFS_SKIP_SMUDGE=1 git clone -q -b "${REPO_BRANCH}" "${REPO_URL}" "$REPO_DIR" || fail "clone"
cd "$REPO_DIR" || fail "cd repo"
if [ -n "${REPO_COMMIT:-}" ]; then
    git checkout -q "${REPO_COMMIT}" || fail "checkout ${REPO_COMMIT}"
fi
echo "running commit $(git rev-parse HEAD)"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$REPO_DIR/reports/gpu_llama"

# ---------------------------------------------------------- build llama.cpp
status "building llama.cpp ($LLAMA_CPP_REF) with CUDA"
which nvcc || echo "warning: nvcc not on PATH -- a runtime (non-devel) image cannot build -DGGML_CUDA=ON"
nvcc --version || true
BUILD_T0=$SECONDS
git clone -q --depth 1 --branch "$LLAMA_CPP_REF" https://github.com/ggml-org/llama.cpp "$LLAMA_DIR" \
    || git clone -q https://github.com/ggml-org/llama.cpp "$LLAMA_DIR" \
    || fail "clone llama.cpp"
if [ -n "$LLAMA_CPP_REF" ]; then
    git -C "$LLAMA_DIR" fetch -q --depth 1 origin "$LLAMA_CPP_REF" 2>/dev/null
    git -C "$LLAMA_DIR" checkout -q "$LLAMA_CPP_REF" 2>/dev/null || echo "note: staying on default branch"
fi
echo "llama.cpp at $(git -C "$LLAMA_DIR" rev-parse HEAD)"
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
    -DLLAMA_CURL=ON \
    -DCMAKE_BUILD_TYPE=Release \
    > "$OUT/llama_cmake_configure.log" 2>&1 || { tail -40 "$OUT/llama_cmake_configure.log"; fail "cmake configure"; }
cmake --build "$LLAMA_DIR/build" --config Release -j "$(nproc)" \
    --target llama-server \
    > "$OUT/llama_build.log" 2>&1 || { tail -60 "$OUT/llama_build.log"; fail "cmake build"; }
LLAMA_SERVER="$LLAMA_DIR/build/bin/llama-server"
[ -x "$LLAMA_SERVER" ] || fail "llama-server not built at $LLAMA_SERVER"
echo "llama.cpp built in $(( SECONDS - BUILD_T0 ))s"
write_runinfo

# -------------------------------------------------------------- download GGUF
status "downloading GGUF"
mkdir -p "$MODEL_DIR"
GGUF_PATH="$("$PYTHON" - <<PY || echo ""
from huggingface_hub import hf_hub_download
print(hf_hub_download(repo_id="${GGUF_REPO}", filename="${GGUF_FILE}", local_dir="${MODEL_DIR}"))
PY
)"
GGUF_PATH="$(echo "$GGUF_PATH" | tail -1)"
[ -s "$GGUF_PATH" ] || fail "GGUF download ($GGUF_REPO/$GGUF_FILE)"
echo "gguf: $GGUF_PATH ($(du -h "$GGUF_PATH" | cut -f1))"

# -------------------------------------------------------------- serve on GPU
# -ngl 99: every layer on the card. The q4_0 weights are 3.35GB, so this fits
# even the 8GB tier; anything less would silently measure a CPU/GPU hybrid.
# Server stdout goes to a FILE, never a pipe: an undrained pipe blocks
# llama-server the moment its 64KB buffer fills, which looks exactly like the
# model hanging mid-generation (see reports/EXPERIMENT_LLM_JUDGE_CPU.md).
start_server() {
    local fa_flag="$1" ctx_total="$2"
    LLAMA_SERVER_CMD="$LLAMA_SERVER -m $GGUF_PATH --host 127.0.0.1 --port $LLAMA_PORT -ngl 99 $fa_flag -c $ctx_total --parallel $N_PARALLEL --cont-batching --reasoning off --metrics --alias $MODEL_ALIAS"
    echo "starting: $LLAMA_SERVER_CMD"
    # shellcheck disable=SC2086
    nohup "$LLAMA_SERVER" -m "$GGUF_PATH" --host 127.0.0.1 --port "$LLAMA_PORT" \
        -ngl 99 $fa_flag -c "$ctx_total" --parallel "$N_PARALLEL" --cont-batching \
        --reasoning off --metrics --alias "$MODEL_ALIAS" \
        > "$LLAMA_LOG" 2>&1 &
    LLAMA_PID=$!
    local t0=$SECONDS
    for _ in $(seq 1 300); do
        if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
            echo "llama-server exited during startup; last log lines:"
            tail -30 "$LLAMA_LOG"
            return 1
        fi
        if curl -fsS "$LLAMA_BASE_URL/health" >/dev/null 2>&1; then
            LLAMA_STARTUP_SECONDS=$(( SECONDS - t0 ))
            echo "llama-server healthy after ${LLAMA_STARTUP_SECONDS}s (pid $LLAMA_PID)"
            return 0
        fi
        sleep 1
    done
    echo "llama-server did not become healthy in 300s; last log lines:"
    tail -30 "$LLAMA_LOG"
    kill "$LLAMA_PID" 2>/dev/null
    return 1
}

status "starting llama-server"
# Probe the binary that was actually built rather than assuming the flag
# spellings of whichever llama.cpp release the docs were written against.
# `--reasoning off` is the one flag that must exist: without it gemma-4 spends
# the whole budget on a chain of thought and returns empty `content`, which is
# a silently wrong answer rather than a loud failure.
SERVER_HELP="$("$LLAMA_SERVER" --help 2>&1)"
echo "$SERVER_HELP" | grep -E -- "--flash-attn|--reasoning|--parallel|--metrics|--alias|-ngl" | head -20
echo "$SERVER_HELP" | grep -q -- "--reasoning" \
    || echo "WARNING: this llama.cpp build has no --reasoning flag; gemma-4 may answer with reasoning_content only"
# `-fa` became a valued flag (on|off|auto) at some point; try the form the help
# text suggests first and keep the other as a fallback.
if echo "$SERVER_HELP" | grep -qE -- "--flash-attn\s+\[?(on|FA)"; then
    FA_FORMS=("-fa on" "-fa")
else
    FA_FORMS=("-fa" "-fa on")
fi
# A KV cache sized for 32 slots may simply not fit an 8GB card, so halving the
# context is a real fallback; whichever value wins is recorded in RUNINFO.json.
SERVER_UP=0
for fa in "${FA_FORMS[@]}"; do
    for ctx in "$N_CTX_TOTAL" "$(( N_CTX_TOTAL / 2 ))" "$(( N_CTX_TOTAL / 4 ))"; do
        if start_server "$fa" "$ctx"; then
            N_CTX_TOTAL="$ctx"
            SERVER_UP=1
            break 2
        fi
    done
done
[ "$SERVER_UP" = "1" ] || fail "llama-server would not start (see llama_server.log)"
grep -E "offloaded .*layers to GPU|Device 0:|CUDA[0-9]+ .*buffer size" "$LLAMA_LOG" | head -20
# "offloaded N/M layers to GPU" with N != M is the quiet failure mode this run
# exists to avoid: it still serves, just at CPU speed for the remainder.
OFFLOAD_LINE="$(grep -oE "offloaded [0-9]+/[0-9]+ layers to GPU" "$LLAMA_LOG" | tail -1)"
if [ -z "$OFFLOAD_LINE" ]; then
    echo "WARNING: no offload line in llama_server.log -- cannot confirm GPU offload"
elif ! echo "$OFFLOAD_LINE" | awk '{split($2,a,"/"); exit (a[1]==a[2] ? 0 : 1)}'; then
    echo "WARNING: partial offload ($OFFLOAD_LINE) -- the numbers below are a CPU/GPU hybrid"
else
    echo "full GPU offload confirmed: $OFFLOAD_LINE"
fi
write_runinfo
sync_artifacts

# ------------------------------------------------------------- 1. benchmark
status "benchmark"
"$PYTHON" scripts/benchmark_llama_server.py \
    --base-url "$LLAMA_BASE_URL" \
    --model "$MODEL_ALIAS" \
    --output reports/gpu_llama/benchmark \
    --server-log "$LLAMA_LOG" \
    --runinfo "$OUT/RUNINFO.json" \
    --startup-seconds "${LLAMA_STARTUP_SECONDS:-0}" \
    || echo "BENCHMARK FAILED (continuing)"
sync_artifacts

# ---------------------------------------------------- 2. accuracy on BEA-60K
# Evaluation only -- BEA-60K is locked, nothing here trains, tunes or selects
# on it. Both scorers attach to the server started above (--llama-base-url)
# rather than each spawning their own copy of the weights.
status "downloading BEA-60K"
"$PYTHON" scripts/download_bea60k.py || fail "BEA-60K download"

JUDGE_RC=0
for mode in index sentence; do
    status "judge: $mode"
    "$PYTHON" scripts/llm_judge_bea60k_cpu.py \
        --backend llama-cpp \
        --llama-base-url "$LLAMA_BASE_URL" \
        --model-id "$GGUF_REPO" \
        --model-name "gemma-4-e2b-q4-$mode" \
        --answer-mode "$mode" \
        --bea-dir data/bea60k \
        --output "reports/gpu_llama/judge/$mode" \
        --n-samples "$N_SAMPLES" \
        --skip-timed \
        || { echo "JUDGE FAILED: $mode"; JUDGE_RC=1; }
    sync_artifacts
done
JUDGE_STATUS=$([ "$JUDGE_RC" -eq 0 ] && echo "ok" || echo "one or more modes failed")

# ------------------------------------------------------- 3. dspy prompt search
# Least certain step by design (another agent owns that script, and it may not
# exist on this commit yet), so it is wrapped: a failure here is recorded and
# the run still finishes with the benchmark and judge results intact.
status "dspy prompt search"
# OpenAI-compatible clients refuse to start without *some* key; llama-server
# ignores it entirely.
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-local-llama-server}"
if [ -f scripts/dspy_prompt_search.py ]; then
    pip_install dspy || echo "pip install dspy failed (the script may vendor its own deps)"
    if "$PYTHON" scripts/dspy_prompt_search.py \
        --base-url "$LLAMA_BASE_URL/v1" \
        --model "$MODEL_ALIAS" \
        --output reports/gpu_llama/dspy; then
        DSPY_STATUS="ok"
    else
        DSPY_STATUS="failed (non-fatal, see run.log)"
        echo "DSPY FAILED (non-fatal)"
        touch "$OUT/DSPY_FAILED"
    fi
else
    DSPY_STATUS="skipped (scripts/dspy_prompt_search.py not present at this commit)"
    echo "$DSPY_STATUS"
fi
sync_artifacts

# ------------------------------------------------------------------- wrap up
status "collecting artifacts"
cp "$LLAMA_LOG" "$OUT/artifacts/" 2>/dev/null
sync_artifacts

if [ "$JUDGE_RC" -eq 0 ]; then
    status "SUCCESS"
else
    status "FAILED: judge step, see run.log"
fi
write_runinfo
echo "bootstrap finished $(date -u +%FT%TZ) judge=$JUDGE_STATUS dspy=$DSPY_STATUS"
touch "$OUT/DONE"

# The HTTP server (and llama-server) stay up so artifacts can still be fetched;
# the orchestrator terminates the pod explicitly. Pods bill per second.
sleep infinity
