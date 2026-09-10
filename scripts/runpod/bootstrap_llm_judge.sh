#!/usr/bin/env bash
# Pod entrypoint for the LLM-judge experiment: one or two small instruct LLMs
# prompt-select the best Hunspell suggestion for BEA-60K word errors; see
# scripts/llm_judge_bea60k.py and reports/EXPERIMENT_LLM_JUDGE.md.
#
# Gemma-4 (`google/gemma-4-E2B-it`) needs torch>=2.5 so
# `from torch.distributed.tensor import DTensor` succeeds, built for CUDA
# 12.4 (cu124). setup_llm_judge.sh installs torch==2.5.1+cu124 into the
# venv on the documented py3.11 / CUDA 12.4 image, force-installs
# nvidia-cudnn-cu12 (libcudnn.so.9) into the venv, and writes
# /workspace/llm-judge-env.sh with LD_LIBRARY_PATH. Do not launch with a
# cu128 image or cu128 wheels -- Community hosts with a CUDA 12.4 driver
# then silently fall back to CPU.
#
# Runs as the container's entrypoint (see scripts/runpod/bootstrap.sh for why:
# the controlling environment has outbound HTTPS only). Progress and results
# are served over Runpod's HTTP proxy:
#
#   https://<pod-id>-8000.proxy.runpod.net/run.log
#   https://<pod-id>-8000.proxy.runpod.net/STATUS
#   https://<pod-id>-8000.proxy.runpod.net/DONE
#   https://<pod-id>-8000.proxy.runpod.net/artifacts/reports/llm_judge/...
#
# Configured by environment variables set at pod creation:
#   REPO_URL, REPO_BRANCH, REPO_COMMIT
#   MODEL_A_ID, MODEL_A_NAME, MODEL_B_ID, MODEL_B_NAME
#     MODEL_B_ID empty/none/null/skip -> run MODEL_A only (Gemma-only path)
#   N_SAMPLES (default 100)
#   TIME_BUDGET_SECONDS (default 300). A value other than 300 skips the
#     100-sample phase unless ALSO_SAMPLE=1, so TIME_BUDGET_SECONDS=3600 is a
#     single 1-hour timed pass.
#   FULL_BEA=1 -> pass --full (non-wrapping eligible set; still honors
#     TIME_BUDGET_SECONDS).
#   ALSO_SAMPLE=1 -> keep the fixed n-samples phase even for long/full runs.
# Deliberately no `set -u`: this script sources the image's profile scripts.
set -o pipefail

OUT=/workspace/out
mkdir -p "$OUT"

export DEBIAN_FRONTEND=noninteractive
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
for extra in /opt/conda/bin /usr/local/nvidia/bin /venv/bin /workspace/venv/bin; do
    [ -d "$extra" ] && export PATH="$extra:$PATH"
done
for profile in /etc/profile.d/*.sh /root/.bashrc; do
    # shellcheck disable=SC1090
    [ -r "$profile" ] && . "$profile" >/dev/null 2>&1
done
true

PYTHON="$(command -v python3 || command -v python || echo /usr/bin/python3)"

status() { echo "$1" > "$OUT/STATUS"; }
status "starting"

nohup "$PYTHON" -m http.server 8000 --directory "$OUT" >/dev/null 2>&1 &
sleep 1

exec > >(tee -a "$OUT/run.log") 2>&1

echo "bootstrap starting $(date -u +%FT%TZ)"
echo "python=$PYTHON ($($PYTHON -V 2>&1))"
echo "repo=${REPO_URL:-unset} branch=${REPO_BRANCH:-unset}"
nvidia-smi || echo "nvidia-smi unavailable"

fail() { status "FAILED: $1"; echo "FAILED: $1"; touch "$OUT/DONE"; sleep infinity; }

if ! command -v git >/dev/null || ! command -v curl >/dev/null; then
    status "installing git"
    apt-get update -qq && apt-get install -y -qq --no-install-recommends git curl ca-certificates \
        || fail "install git/curl"
fi

status "cloning"
GIT_LFS_SKIP_SMUDGE=1 git clone -b "${REPO_BRANCH}" "${REPO_URL}" \
  /workspace/spell-corrector || fail "clone"
cd /workspace/spell-corrector
if [ -n "${REPO_COMMIT:-}" ]; then
    git checkout -q "${REPO_COMMIT}" || fail "checkout ${REPO_COMMIT}"
fi
echo "running commit $(git rev-parse HEAD)"
export REPO_DIR=/workspace/spell-corrector
export PYTHONPATH=/workspace/spell-corrector
export PYTHON

status "setup"
bash scripts/runpod/setup_llm_judge.sh || fail "setup"
if [ -x /workspace/venv/bin/python ]; then
    PYTHON=/workspace/venv/bin/python
    export PYTHON
    export PATH="/workspace/venv/bin:$PATH"
fi
# setup is a subprocess; re-apply nvidia lib dirs so `import torch` finds
# libcudnn.so.9 (venv nvidia-cudnn-cu12 is not on the default linker path).
if [ -f /workspace/llm-judge-env.sh ]; then
    # shellcheck disable=SC1091
    . /workspace/llm-judge-env.sh
    echo "sourced /workspace/llm-judge-env.sh"
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"
fi
echo "python now $PYTHON ($($PYTHON -V 2>&1))"

status "unit tests"
"$PYTHON" -m pytest tests/test_llm_judge.py tests/test_llm_judge_harness.py tests/test_byte_encoding.py -q \
    || fail "unit tests"

status "downloading BEA-60K"
"$PYTHON" scripts/download_bea60k.py || fail "BEA-60K download"

MODEL_A_ID="${MODEL_A_ID:-Qwen/Qwen3.5-0.8B}"
MODEL_A_NAME="${MODEL_A_NAME:-qwen3.5-0.8b}"
MODEL_B_ID="${MODEL_B_ID:-google/gemma-4-E2B-it}"
MODEL_B_NAME="${MODEL_B_NAME:-gemma-4-e2b}"
N_SAMPLES="${N_SAMPLES:-100}"
TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-300}"
FULL_BEA="${FULL_BEA:-0}"
ALSO_SAMPLE="${ALSO_SAMPLE:-0}"

skip_model() {
    local raw="${1:-}"
    local lowered
    lowered="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')"
    case "$lowered" in
        ""|none|null|skip) return 0 ;;
        *) return 1 ;;
    esac
}

EXTRA_ARGS=(--n-samples "$N_SAMPLES" --time-budget-seconds "$TIME_BUDGET_SECONDS")
if [ "$FULL_BEA" = "1" ]; then
    EXTRA_ARGS+=(--full)
fi
if [ "$ALSO_SAMPLE" = "1" ]; then
    EXTRA_ARGS+=(--also-sample)
elif [ "$FULL_BEA" = "1" ] || [ "$TIME_BUDGET_SECONDS" != "300" ]; then
    # Long timed runs (e.g. 3600s) and --full skip the old sample100 + 300s combo
    # unless ALSO_SAMPLE=1. Default TIME_BUDGET_SECONDS=300 keeps both phases.
    EXTRA_ARGS+=(--skip-sample)
fi

echo "llm_judge extra args: ${EXTRA_ARGS[*]}"
echo "MODEL_A=${MODEL_A_ID} MODEL_B=${MODEL_B_ID}"

RC=0
NAMES=()
run_model() {
    local model_id="$1"
    local model_name="$2"
    if skip_model "$model_id"; then
        echo "skipping empty/none model ($model_name)"
        return 0
    fi
    NAMES+=("$model_name")
    status "running $model_name"
    "$PYTHON" scripts/llm_judge_bea60k.py \
        --model-id "$model_id" \
        --model-name "$model_name" \
        --bea-dir data/bea60k \
        --output "reports/llm_judge/$model_name" \
        "${EXTRA_ARGS[@]}" \
        || { echo "MODEL FAILED: $model_name"; RC=1; }
}

run_model "$MODEL_A_ID" "$MODEL_A_NAME"
run_model "$MODEL_B_ID" "$MODEL_B_NAME"
if [ "${#NAMES[@]}" -eq 0 ]; then
    fail "no models to run (MODEL_A_ID and MODEL_B_ID are empty/none)"
fi

status "summarizing"
"$PYTHON" - "${NAMES[@]}" <<'PY' || true
import json
import sys
from pathlib import Path

names = sys.argv[1:]
summary = {}
for name in names:
    path = Path("reports/llm_judge") / name / "results.json"
    if path.is_file():
        summary[name] = json.loads(path.read_text())
    else:
        summary[name] = {"error": "no results.json"}
Path("reports/llm_judge/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
for name, res in summary.items():
    if "error" in res:
        print(f"{name}: {res['error']}")
        continue
    s100 = res.get("sample_100") or {}
    timed = res.get("timed") or res.get("timed_5min") or {}
    full = res.get("full_bea60k") or {}
    bits = [f"{name}:"]
    if s100:
        bits.append(
            f"sample100 acc={s100.get('overall_accuracy')} "
            f"p50_ms={s100.get('latency_stats', {}).get('p50_ms')}"
        )
    if timed:
        bits.append(
            f"timed n={timed.get('n_ok')} elapsed={timed.get('elapsed_seconds')} "
            f"qps={timed.get('throughput_qps')} acc={timed.get('overall_accuracy')} "
            f"p50_ms={timed.get('latency_stats', {}).get('p50_ms')} "
            f"p99_ms={timed.get('latency_stats', {}).get('p99_ms')}"
        )
    if full:
        bits.append(
            f"full_bea60k n={full.get('n_ok')} elapsed={full.get('elapsed_seconds')} "
            f"qps={full.get('throughput_qps')} acc={full.get('overall_accuracy')} "
            f"p50_ms={full.get('latency_stats', {}).get('p50_ms')} "
            f"p99_ms={full.get('latency_stats', {}).get('p99_ms')}"
        )
    print(" | ".join(bits))
PY

status "collecting artifacts"
mkdir -p "$OUT/artifacts/reports"
cp -r reports/llm_judge "$OUT/artifacts/reports/" 2>/dev/null

if [ "$RC" -eq 0 ]; then status "SUCCESS"; else status "FAILED: one or more models failed, see run.log"; fi
echo "bootstrap finished rc=$RC $(date -u +%FT%TZ)"
touch "$OUT/DONE"

sleep infinity
