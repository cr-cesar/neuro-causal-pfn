#!/bin/bash
# Run a list of training jobs one after another on a machine without a
# scheduler (a lab workstation with one GPU). Each non-empty, non-comment
# line of the jobs file is a set of VAR=value assignments followed by the job
# script, exactly as the qsub -v lines on Myriad, e.g.
#
#   CACHE=outputs/prior_cache/kch_e1_32d_x.npz OUT=outputs/pfn_reduced_demo SEED=0 ARCH=linear \
#     PRIOR_HYPER="two_arms=1 effect_scale=0.2:1 effect_headroom=1" scripts/train_pfn_myriad.qsub.sh
#
# A job whose OUT/pfn.pt already exists is skipped (so the list can be rerun
# after an interruption); a failed job is logged and the next one starts.
# Logs go to logs/<n>_<name>.log, the name taken from a JOBNAME=... token
# when the line has one, else from OUT. VENV names the Python environment
# (default .venv in the repository). The jobs run in the repository holding
# this script; WORKDIR=<dir> runs them in another checkout (the per-fold
# repository, whose jobs come from `python -m ncpfold.plan --emit-local`).
#
#   bash scripts/run_jobs.sh jobs/today.txt
#   WORKDIR=~/neuro-causal-pfn-perfold VENV=~/env bash scripts/run_jobs.sh jobs_perfold.txt
set -uo pipefail
JOBS="${1:?usage: bash scripts/run_jobs.sh <jobs file>}"
JOBS="$(readlink -f "$JOBS")"
cd "${WORKDIR:-$(dirname "$0")/..}"
export VENV="${VENV:-$PWD/.venv}"
[ -x "$VENV/bin/python" ] || { echo "no Python environment at $VENV (run scripts/setup_env_local.sh)"; exit 2; }
mkdir -p logs
n=0; ok=0; skipped=0; failed=0
while IFS= read -r line || [ -n "$line" ]; do
    [[ -z "${line// }" || "$line" =~ ^[[:space:]]*# ]] && continue
    n=$((n + 1))
    out=$(printf '%s\n' "$line" | sed -nE 's/(^|.*[[:space:]])OUT=("[^"]*"|[^[:space:]]+).*/\2/p' | tr -d '"')
    jobname=$(printf '%s\n' "$line" | sed -nE 's/(^|.*[[:space:]])JOBNAME=([^[:space:]]+).*/\2/p')
    name="${jobname:-$(basename "${out:-job$n}")}"
    if [ -n "$out" ] && [ -s "$out/pfn.pt" ]; then
        echo "[$n] skip, done already: $out/pfn.pt"; skipped=$((skipped + 1)); continue
    fi
    echo "[$n] $(date +%H:%M) start $name"
    if bash -c "$line" > "logs/${n}_${name}.log" 2>&1; then
        echo "[$n] $(date +%H:%M) ok    $name"; ok=$((ok + 1))
    else
        echo "[$n] $(date +%H:%M) FAILED $name (see logs/${n}_${name}.log)"; failed=$((failed + 1))
    fi
done < "$JOBS"
echo "jobs: $n, ok $ok, skipped $skipped, failed $failed"
[ "$failed" -eq 0 ]
