#!/bin/bash -l
# One Phase-1 experiment per job, so the programme fits inside Myriad's 48-h
# wall-clock limit instead of running the whole arm in a single submission.
# Submit in dependency order, reusing the same out-root so later experiments
# can read the earlier winners:
#
#   qsub -v EXP=E1 scripts/run_experiment_myriad.qsub.sh
#   qsub -v EXP=E2 scripts/run_experiment_myriad.qsub.sh      # after E1 finishes
#   qsub -v EXP=E3,SEEDS=3 scripts/run_experiment_myriad.qsub.sh
#
# The log lands in ncp-EXP.o<jobid>; the leaderboard accumulates in
# outputs/experiments/leaderboard.{csv,md}.
#
#$ -N ncp-exp
#$ -l h_rt=24:0:0
#$ -l gpu=1
#$ -pe smp 8
#$ -l mem=6G
#$ -l tmpfs=20G
#$ -cwd
#$ -j y
set -euo pipefail

module load python3/3.11
source ~/venvs/neuro/bin/activate

EXP="${EXP:?set EXP, e.g.: qsub -v EXP=E1 scripts/run_experiment_myriad.qsub.sh}"
SEEDS="${SEEDS:-3}"
TIER="${TIER:-full}"
OUT="${OUT:-outputs/experiments}"

python -m neurocausalpfn.experiments.runner \
    --experiment "$EXP" --mode full --data-tier "$TIER" \
    --seeds "$SEEDS" --out-root "$OUT" --report
