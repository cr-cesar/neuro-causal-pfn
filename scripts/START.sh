#!/bin/bash
# One command on the external machine: unpack the code, install the Python
# environment from the bundled wheels (no internet needed), put the caches in
# place and start the job list in the background, so that it keeps running
# after the remote session closes.
#
#   tar -xzf neuro-causal-pfn-bundle.tgz && bash bundle/START.sh [jobs/<list>.txt]
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
if [ ! -x .venv/bin/python ]; then
    if [ -d "$HERE/wheels" ]; then WHEELS="$HERE/wheels" bash scripts/setup_env_local.sh; else bash scripts/setup_env_local.sh; fi
else
    bash scripts/setup_env_local.sh check
fi
[ -s "$JOBS" ] || { echo "no jobs file $JOBS"; exit 2; }
log="run_$(date +%Y%m%d_%H%M).log"
nohup bash scripts/run_jobs.sh "$JOBS" > "$log" 2>&1 &
echo "started $(grep -c -v -E '^\s*(#|$)' "$JOBS") jobs from $JOBS in the background (pid $!)"
echo "progress:  tail -f $PWD/$log"
echo "results:   $PWD/outputs/pfn_*/   (send these folders back when the log says 'jobs: ...')"
