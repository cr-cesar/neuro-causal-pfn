#!/bin/bash -l
# Per-channel active latent dimensions for E5 (ARD), from frozen checkpoints.
# Short job: encodes 2 x 4,119 volumes per seed, no retraining.
#
#   qsub scripts/active_dims_per_channel.qsub.sh
#
# Writes outputs/experiments/E5/active_dims.json and prints the per-channel
# table to the job log. The disconnectome count is the chain's real dimensionality.
#
#$ -N e5-activedims
#$ -l h_rt=1:0:0
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
EID="${EID:-E5}"

python scripts/active_dims_per_channel.py --out-root "$OUT" --eid "$EID"
