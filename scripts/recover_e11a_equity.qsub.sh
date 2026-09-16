#!/bin/bash -l
# Recover the E11a equity breakdown from its trained checkpoints (short GPU
# job: encodes 2 x 4,119 volumes per seed, no retraining).
#
#   qsub scripts/recover_e11a_equity.qsub.sh
#
# Writes outputs/experiments/E11a/equity.json and prints the per-stratum
# table (criterion: worst-to-best root-PEHE ratio < 2x) to the job log.
#
#$ -N e11a-equity
#$ -l h_rt=2:0:0
#$ -l gpu=1
#$ -pe smp 4
#$ -l mem=8G
#$ -l tmpfs=10G
#$ -cwd
#$ -j y
set -euo pipefail

module load python3/3.11
source ~/venvs/neuro/bin/activate

OUT="${OUT:-outputs/experiments}"

python scripts/recover_e11a_equity.py --out-root "$OUT"
