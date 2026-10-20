#!/bin/bash
# One command on the external machine: unpack the code, install the Python
# environment from the bundled wheels (no internet needed), put the caches in
# place and start the job list in the background, so that it keeps running
# after the remote session closes.
#
#   tar -xzf neuro-causal-pfn-bundle.tgz && bash bundle/START.sh [jobs/<list>.txt]
#   EXISTING_VENV=~/env bash bundle/START.sh      # reuse an environment whose torch fits the GPU
#
# Afterwards:  tail -f neuro-causal-pfn/run_*.log     (progress)
#              ls neuro-causal-pfn/outputs/pfn_*/pfn.pt (finished checkpoints)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."
JOBS="${1:-jobs/$(ls "$HERE/jobs" | head -1)}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "warning: nvidia-smi not found"
if [ ! -d neuro-causal-pfn ]; then
    [ -s "$HERE/code.tgz" ] || { echo "no code.tgz in the bundle and no neuro-causal-pfn/ next to it"; exit 2; }
    tar -xzf "$HERE/code.tgz"
fi
cd neuro-causal-pfn
mkdir -p outputs data jobs
cp -rn "$HERE/outputs/." outputs/ && cp -rn "$HERE/data/." data/ && cp -n "$HERE"/jobs/*.txt jobs/
# EXISTING_VENV=~/env: an environment that already has a torch for this GPU
# (the RTX 5090 needs torch >= 2.7); otherwise a .venv is made from the wheels
export VENV="${EXISTING_VENV:-$PWD/.venv}"
if "$VENV/bin/python" -c "import neurocausalpfn, torch" 2>/dev/null; then
    EXISTING_VENV="${EXISTING_VENV:-}" bash scripts/setup_env_local.sh check
else
    W=(); [ -d "$HERE/wheels" ] && W=(WHEELS="$HERE/wheels")
    env ${W[@]+"${W[@]}"} ${EXISTING_VENV:+EXISTING_VENV="$EXISTING_VENV"} bash scripts/setup_env_local.sh
fi
[ -s "$JOBS" ] || { echo "no jobs file $JOBS"; exit 2; }
log="run_$(date +%Y%m%d_%H%M).log"
nohup bash scripts/run_jobs.sh "$JOBS" > "$log" 2>&1 &
echo "started $(grep -c -v -E '^\s*(#|$)' "$JOBS") jobs from $JOBS in the background (pid $!)"
echo "progress:  tail -f $PWD/$log"
echo "results:   $PWD/outputs/pfn_*/   (send these folders back when the log says 'jobs: ...')"
