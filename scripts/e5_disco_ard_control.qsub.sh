#!/bin/bash -l
# E5-disco control: ARD prior on a disconnectome-only VAE, trained from scratch,
# one seed per array task (each ~12-16 h; a 4 h wall clock would not fit a full
# 200-epoch VAE, so this asks for the training budget and backfills as a 3-task
# array). Reports the disco-only active-dimension count.
#
#   qsub scripts/e5_disco_ard_control.qsub.sh
#   # then aggregate the three seeds (login node, fast):
#   for s in 0 1 2; do cat outputs/experiments/E5_disco_ard/seed$s/active_dims_disco.json; done
#
#$ -N e5-disco-ard
#$ -l h_rt=20:0:0
#$ -l gpu=1
#$ -pe smp 8
#$ -l mem=6G
#$ -l tmpfs=20G
#$ -cwd
#$ -j y
#$ -t 1-3
set -euo pipefail

module load python3/3.11
source ~/venvs/neuro/bin/activate

OUT="${OUT:-outputs/experiments}"
ZDIM="${ZDIM:-100}"
SEED=$((SGE_TASK_ID - 1))

python scripts/e5_disco_ard_control.py --seed "$SEED" --zdim "$ZDIM" --out-root "$OUT"
