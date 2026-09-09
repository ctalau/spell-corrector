#!/usr/bin/env bash
# Pod entrypoint: clone, set up, run the whole experiment unattended.
#
# This runs as the container's entrypoint rather than over SSH, because the
# controlling environment can only make outbound HTTPS connections. Progress and
# artifacts are served over Runpod's HTTP proxy instead:
#
#   https://<pod-id>-8000.proxy.runpod.net/run.log
#   https://<pod-id>-8000.proxy.runpod.net/STATUS
#   https://<pod-id>-8000.proxy.runpod.net/DONE          (appears when finished)
#   https://<pod-id>-8000.proxy.runpod.net/artifacts/...
#
# Overriding the entrypoint bypasses the base image's init script, so nothing
# here may assume that PATH already points at the image's Python: resolve every
# tool explicitly and install what is missing.
#
# Configured by environment variables set at pod creation:
#   REPO_URL, REPO_BRANCH, TARGET_TRAIN, TARGET_VALID, CONFIG
# Deliberately no `set -u`: this script sources the image's profile scripts,
# and those routinely reference unset variables. Under `set -u` that aborts the
# shell before the first status write, which looks identical to "the container
# did nothing".
set -o pipefail

OUT=/workspace/out
mkdir -p "$OUT"

export DEBIAN_FRONTEND=noninteractive
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
for extra in /opt/conda/bin /usr/local/nvidia/bin /venv/bin /workspace/venv/bin; do
    [ -d "$extra" ] && export PATH="$extra:$PATH"
done
# The image's own environment, if it has one, wins for the actual work.
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

echo "bootstrap starting $(date -u +%FT%TZ)"
echo "python=$PYTHON ($($PYTHON -V 2>&1))"
echo "repo=${REPO_URL:-unset} branch=${REPO_BRANCH:-unset}"
nvidia-smi || echo "nvidia-smi unavailable"
echo "vcpu=$(nproc) ram=$(free -g | awk '/^Mem:/{print $2}')GB"

fail() { status "FAILED: $1"; echo "FAILED: $1"; touch "$OUT/DONE"; sleep infinity; }

if ! command -v git >/dev/null || ! command -v curl >/dev/null; then
    status "installing git"
    apt-get update -qq && apt-get install -y -qq --no-install-recommends git curl ca-certificates \
        || fail "install git/curl"
fi

status "cloning"
# Skip LFS: the committed parquet is stale and gets rebuilt anyway.
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 -b "${REPO_BRANCH}" "${REPO_URL}" \
  /workspace/spell-corrector || fail "clone"
cd /workspace/spell-corrector
export REPO_DIR=/workspace/spell-corrector
export PYTHONPATH=/workspace/spell-corrector
export PYTHON

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
cp -r artifacts/* "$OUT/artifacts/" 2>/dev/null
cp -r reports/* "$OUT/reports/" 2>/dev/null
cp data/processed/data_stats.json data/processed/manifest.json "$OUT/" 2>/dev/null
# The benchmark's per-error dump is large and not needed off-box.
rm -f "$OUT/reports/bea60k/predictions.jsonl"

if [ "$RC" -eq 0 ]; then status "SUCCESS"; else status "FAILED: experiment rc=$RC"; fi
echo "bootstrap finished rc=$RC $(date -u +%FT%TZ)"
touch "$OUT/DONE"

# Stay alive so artifacts remain downloadable. The controller terminates the pod.
sleep infinity
