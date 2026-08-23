#!/usr/bin/env bash
# Table 9 experiment programme in prototype mode (CPU, synthetic masks).
#
# Runs the full Arm-A dependency chain end to end and writes a leaderboard. Use
# this as the smoke test for the orchestrator; it needs no real data or cluster.
#
#   bash scripts/run_experiments.sh            # Arm A, 3 seeds
#   bash scripts/run_experiments.sh E3 1       # a single experiment, 1 seed
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

TARGET="${1:-A}"          # an experiment id (E3), an arm letter (A) or "all"
SEEDS="${2:-3}"
OUT="${OUT:-outputs/experiments}"

echo ">> Table-9 orchestrator | target=${TARGET} | seeds=${SEEDS} | mode=prototype"

if [[ "${TARGET}" == "all" ]]; then
  python -m neurocausalpfn.experiments.runner --all --mode prototype --seeds "${SEEDS}" --out-root "${OUT}" --report
elif [[ "${TARGET}" =~ ^[A-E]$ ]]; then
  python -m neurocausalpfn.experiments.runner --arm "${TARGET}" --mode prototype --seeds "${SEEDS}" --out-root "${OUT}" --report
else
  python -m neurocausalpfn.experiments.runner --experiment "${TARGET}" --mode prototype --seeds "${SEEDS}" --out-root "${OUT}" --report
fi

echo ">> leaderboard: ${OUT}/leaderboard.md"
