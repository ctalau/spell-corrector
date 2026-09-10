#!/usr/bin/env bash
# Pod entrypoint for the LLM-judge experiment: two small instruct LLMs prompt-
# select the best Hunspell suggestion for BEA-60K word errors; see
# scripts/llm_judge_bea60k.py and reports/EXPERIMENT_LLM_JUDGE.md.
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
#   N_SAMPLES (default 100), TIME_BUDGET_SECONDS (default 300)
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
echo "python now $PYTHON ($($PYTHON -V 2>&1))"

status "unit tests"
"$PYTHON" -m pytest tests/test_llm_judge.py tests/test_byte_encoding.py -q || fail "unit tests"

status "downloading BEA-60K"
"$PYTHON" scripts/download_bea60k.py || fail "BEA-60K download"

MODEL_A_ID="${MODEL_A_ID:-Qwen/Qwen3.5-0.8B}"
MODEL_A_NAME="${MODEL_A_NAME:-qwen3.5-0.8b}"
MODEL_B_ID="${MODEL_B_ID:-google/gemma-4-E2B-it}"
MODEL_B_NAME="${MODEL_B_NAME:-gemma-4-e2b}"
N_SAMPLES="${N_SAMPLES:-100}"
TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-300}"

RC=0
for pair in "$MODEL_A_ID|$MODEL_A_NAME" "$MODEL_B_ID|$MODEL_B_NAME"; do
    model_id="${pair%%|*}"
    model_name="${pair##*|}"
    status "running $model_name"
    "$PYTHON" scripts/llm_judge_bea60k.py \
        --model-id "$model_id" \
        --model-name "$model_name" \
        --bea-dir data/bea60k \
        --output "reports/llm_judge/$model_name" \
        --n-samples "$N_SAMPLES" \
        --time-budget-seconds "$TIME_BUDGET_SECONDS" \
        || { echo "MODEL FAILED: $model_name"; RC=1; }
done

status "summarizing"
"$PYTHON" - "$MODEL_A_NAME" "$MODEL_B_NAME" <<'PY' || true
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
    s100 = res.get("sample_100", {})
    timed = res.get("timed_5min", {})
    print(
        f"{name}: sample100 acc={s100.get('overall_accuracy')} "
        f"p50_ms={s100.get('latency_stats', {}).get('p50_ms')} | "
        f"timed n={timed.get('n_ok')} elapsed={timed.get('elapsed_seconds')} "
        f"qps={timed.get('throughput_qps')} acc={timed.get('overall_accuracy')}"
    )
PY

status "collecting artifacts"
mkdir -p "$OUT/artifacts/reports"
cp -r reports/llm_judge "$OUT/artifacts/reports/" 2>/dev/null

if [ "$RC" -eq 0 ]; then status "SUCCESS"; else status "FAILED: one or more models failed, see run.log"; fi
echo "bootstrap finished rc=$RC $(date -u +%FT%TZ)"
touch "$OUT/DONE"

sleep infinity
