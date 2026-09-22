#!/usr/bin/env bash
# Pod entrypoint for the unmarked-markup audit: kev-4b judges the regex candidates on one GPU.
#
# Self-driving: clone, install, judge the validation rows then the candidates, write DONE. Progress over the proxy:
#   https://<pod-id>-8000.proxy.runpod.net/run.log | STATUS | DONE | *.jsonl
# Set at pod creation by scripts/runpod/launch_kev_markup.py: REPO_URL, REPO_BRANCH, REPO_COMMIT, KEV_COMMIT, KEV_RUN.
set -o pipefail
OUT=/workspace/out
mkdir -p "$OUT"
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
PYTHON="$(command -v python3)"
status() { echo "$1" > "$OUT/STATUS"; echo "== $(date -u +%H:%M:%S) $1"; }
pgrep -f "http.server 8000" >/dev/null 2>&1 || nohup "$PYTHON" -m http.server 8000 --directory "$OUT" >/dev/null 2>&1 &
exec > >(tee -a "$OUT/run.log") 2>&1
status "starting"
echo "gpu: ${CHOSEN_GPU:-unknown} (\$${GPU_PRICE_USD_HR:-?}/hr)"
nvidia-smi || true

status "clone"
REPO_DIR=/workspace/spell-corrector
KEV_DIR=/workspace/kev
GIT_LFS_SKIP_SMUDGE=1 git clone --quiet "$REPO_URL" "$REPO_DIR" || { status "clone-failed"; sleep infinity; }
git -C "$REPO_DIR" checkout --quiet "${REPO_COMMIT:-$REPO_BRANCH}"
GIT_LFS_SKIP_SMUDGE=1 git clone --quiet https://github.com/jaredpalmer/kev "$KEV_DIR" || { status "kev-clone-failed"; sleep infinity; }
[ -n "$KEV_COMMIT" ] && git -C "$KEV_DIR" checkout --quiet "$KEV_COMMIT"
echo "commit: $(git -C "$REPO_DIR" rev-parse HEAD)  kev: $(git -C "$KEV_DIR" rev-parse HEAD)"

status "pip"
# torch stays the image's (2.8, inside kev's <2.9 pin); only kev's other dependencies are added.
"$PYTHON" -m pip install -q "transformers>=5.17,<6" "peft>=0.21" "pydantic>=2.9" accelerate lxml || { status "pip-failed"; sleep infinity; }
"$PYTHON" -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, torch.cuda.get_device_name())" \
    || { status "no-cuda"; sleep infinity; }

cd "$REPO_DIR"
for part in gold_sample candidates; do
    status "judge-$part"
    KEV_DIR="$KEV_DIR" "$PYTHON" scripts/kev/judge_unmarked.py "reports/markup_audit/$part.jsonl" \
        --run "${KEV_RUN:-jaredpalmer/kev-4b}" --device cuda --out "$OUT/judged_$part.jsonl" \
        || { status "judge-$part-failed"; sleep infinity; }
done
status "done"
date -u > "$OUT/DONE"
