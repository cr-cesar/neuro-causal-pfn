#!/bin/bash -l
# Plan B: per-fold "Giles-style" VAE-50 (the released repo has no VAE code, so
# this trains the paper's description: Giles ResNet blocks, 50 dims, batch 10,
# 16-32 epochs, early stop 4, one VAE per replica fold). SGE array = one fold
# per GPU job:
#
#   qsub scripts/train_giles_style_vae50_myriad.qsub.sh
#
# When all 10 tasks are done, score it in the certified replica:
#   qsub -hold_jid <this job id> -v FOLD_LATENTS="outputs/giles_style_vae50",OUT=outputs/giles_replica_vae50 \
#        scripts/run_giles_replica_myriad.qsub.sh
#
#$ -N giles-vae50
#$ -t 1-10
#$ -l h_rt=8:0:0
#$ -l gpu=1
#$ -pe smp 4
#$ -l mem=8G
#$ -l tmpfs=15G
#$ -cwd
#$ -j y
set -euo pipefail

module load python3/3.11
source ~/venvs/neuro/bin/activate

IMAGES="${IMAGES:-data/Full data/disconnectomes}"
OUT="${OUT:-outputs/giles_style_vae50}"
FOLD=$((SGE_TASK_ID - 1))

python scripts/train_giles_style_vae50.py \
    --images-dir "$IMAGES" --fold "$FOLD" --out "$OUT" --device auto
