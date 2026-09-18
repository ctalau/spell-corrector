#!/usr/bin/env bash
# Pod entrypoint for milestone 7: distill Qwen3.5-2B (teacher, M6 QLoRA,
# already 91% Acc@1 on the frozen BEA-100) into a fresh Qwen3.5-0.8B student,
# both 4-bit NF4, on BEA-60K itself (explicit, user-approved exception to the
# locked-benchmark rule -- see CLAUDE.md's "Milestone 7" section; the
# seed-1337 100-sample holdout stays untouched throughout).
#
# Runs as the container's entrypoint. Progress and results are served over
# Runpod's HTTP proxy:
#
#   https://<pod-id>-8000.proxy.runpod.net/run.log
#   https://<pod-id>-8000.proxy.runpod.net/STATUS
#   https://<pod-id>-8000.proxy.runpod.net/distill_progress.jsonl
#   https://<pod-id>-8000.proxy.runpod.net/DONE
#   https://<pod-id>-8000.proxy.runpod.net/artifacts/...
#
# Env: REPO_URL, REPO_BRANCH, REPO_COMMIT, TARGET_ACC (default 0.90),
#      MAX_ROUNDS (default 4), LABEL_BATCH_SIZE (default 32).
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

status "setup"
bash scripts/runpod/setup_distill.sh || fail "setup"
if [ -x /workspace/venv/bin/python ]; then
    PYTHON=/workspace/venv/bin/python
    export PATH="/workspace/venv/bin:$PATH"
fi
if [ -f /workspace/llm-judge-env.sh ]; then
    # shellcheck disable=SC1091
    . /workspace/llm-judge-env.sh
    echo "sourced /workspace/llm-judge-env.sh"
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"
fi
echo "python now $PYTHON ($($PYTHON -V 2>&1))"

status "downloading BEA-60K"
"$PYTHON" scripts/download_bea60k.py || fail "BEA-60K download"

TARGET_ACC="${TARGET_ACC:-0.90}"
MAX_ROUNDS="${MAX_ROUNDS:-4}"
LABEL_BATCH_SIZE="${LABEL_BATCH_SIZE:-32}"
STUDENT_BATCH="${STUDENT_BATCH:-8}"
STUDENT_GRAD_ACCUM="${STUDENT_GRAD_ACCUM:-4}"

status "prepare-data"
"$PYTHON" scripts/distill_2b_to_student.py prepare-data \
    --bea-dir data/bea60k --out-dir data/distill || fail "prepare-data"

status "label-teacher"
"$PYTHON" scripts/distill_2b_to_student.py label-teacher \
    --train-pool data/distill/train_pool.jsonl \
    --out data/distill/train_labeled.jsonl \
    --batch-size "$LABEL_BATCH_SIZE" || fail "label-teacher"

status "train (accuracy-gated)"
STUDENT_OUT=artifacts/spell_slm_m7/qwen35_0_8b_distill_qlora
mkdir -p "$STUDENT_OUT"
: > "$STUDENT_OUT/distill_progress.jsonl"
ln -sf "$(pwd)/$STUDENT_OUT/distill_progress.jsonl" "$OUT/distill_progress.jsonl"
"$PYTHON" scripts/distill_2b_to_student.py train \
    --train-labeled data/distill/train_labeled.jsonl \
    --holdout data/distill/holdout_100.json \
    --out-dir "$STUDENT_OUT" \
    --target-acc "$TARGET_ACC" \
    --max-rounds "$MAX_ROUNDS" \
    --per-device-batch "$STUDENT_BATCH" \
    --grad-accum "$STUDENT_GRAD_ACCUM"
TRAIN_RC=$?
echo "train.py exit code: $TRAIN_RC"

status "collecting artifacts"
mkdir -p "$OUT/artifacts"
cp -r data/distill "$OUT/artifacts/" 2>/dev/null
mkdir -p "$OUT/artifacts/spell_slm_m7"
cp -r "$STUDENT_OUT" "$OUT/artifacts/spell_slm_m7/" 2>/dev/null

if [ "$TRAIN_RC" -eq 0 ]; then
    status "SUCCESS target_acc reached"
else
    status "FINISHED below target (rc=$TRAIN_RC), see distill_progress.jsonl"
fi
echo "bootstrap finished rc=$TRAIN_RC $(date -u +%FT%TZ)"
touch "$OUT/DONE"

sleep infinity
