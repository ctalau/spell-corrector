#!/usr/bin/env bash
# Pod entrypoint: clone, set up, run the whole experiment unattended.
#
# This runs as the container's start command rather than over SSH, because the
# controlling environment can only make outbound HTTPS connections. Progress and
# artifacts are served over Runpod's HTTP proxy instead:
#
#   https://<pod-id>-8000.proxy.runpod.net/run.log
#   https://<pod-id>-8000.proxy.runpod.net/DONE          (appears when finished)
#   https://<pod-id>-8000.proxy.runpod.net/artifacts/...
#
# Configured by environment variables set at pod creation:
#   REPO_URL, REPO_BRANCH, TARGET_TRAIN, TARGET_VALID, CONFIG
set -uo pipefail

OUT=/workspace/out
mkdir -p "$OUT"

# Serve progress before anything else, so even a failure in setup is visible.
nohup python -m http.server 8000 --directory "$OUT" >/dev/null 2>&1 &

exec > >(tee -a "$OUT/run.log") 2>&1

echo "bootstrap starting $(date -u +%FT%TZ)"
echo "repo=${REPO_URL:-unset} branch=${REPO_BRANCH:-unset}"
nvidia-smi || true
echo "vcpu=$(nproc) ram=$(free -g | awk '/^Mem:/{print $2}')GB"

status() { echo "$1" > "$OUT/STATUS"; }
fail() { status "FAILED: $1"; echo "FAILED: $1"; touch "$OUT/DONE"; exit 1; }

status "cloning"
# Skip LFS: the committed parquet is stale and gets rebuilt anyway.
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 -b "${REPO_BRANCH}" "${REPO_URL}" \
  /workspace/spell-corrector || fail "clone"
cd /workspace/spell-corrector
export REPO_DIR=/workspace/spell-corrector
export PYTHONPATH=/workspace/spell-corrector

status "setup"
bash scripts/runpod/setup.sh || fail "setup"

status "experiment"
TARGET_TRAIN="${TARGET_TRAIN:-3000000}" \
TARGET_VALID="${TARGET_VALID:-60000}" \
CONFIG="${CONFIG:-configs/train_full.yaml}" \
  bash scripts/runpod/run_experiment.sh
RC=$?

status "collecting artifacts"
mkdir -p "$OUT/artifacts" "$OUT/reports"
cp -r artifacts/* "$OUT/artifacts/" 2>/dev/null || true
cp -r reports/* "$OUT/reports/" 2>/dev/null || true
cp data/processed/data_stats.json data/processed/manifest.json "$OUT/" 2>/dev/null || true
# The benchmark's per-error dump is large and not needed off-box.
rm -f "$OUT/reports/bea60k/predictions.jsonl"

if [ "$RC" -eq 0 ]; then status "SUCCESS"; else status "FAILED: experiment rc=$RC"; fi
echo "bootstrap finished rc=$RC $(date -u +%FT%TZ)"
touch "$OUT/DONE"

# Stay alive so artifacts remain downloadable. The controller terminates the pod.
sleep infinity
