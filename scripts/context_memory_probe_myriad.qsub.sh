#!/bin/bash -l
# Phase 2 audit, question 4: peak GPU memory and seconds per training step of
# the single-axis model at longer contexts (scripts/context_memory_probe.py).
# About 10 minutes of GPU.
#   qsub -v OUT=status/phase2_costs/context_probe.csv scripts/context_memory_probe_myriad.qsub.sh
#   CONTEXTS="1024 2048 3000 4119" BATCHES="4 2 1" ARCH=linear|tabicl
#
#$ -N ctx-probe
#$ -l h_rt=0:30:0
#$ -l gpu=1
#$ -pe smp 4
#$ -l mem=8G
#$ -cwd
#$ -j y
set -euo pipefail
trap 'exit 100' ERR

module load python3/3.11
source ~/venvs/neuro/bin/activate
echo "node: $(hostname -s) gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo none) job: ${JOB_ID:-?}"
# shellcheck disable=SC2086
python scripts/context_memory_probe.py --contexts ${CONTEXTS:-1024 2048 3000 4119} \
    --batch-sizes ${BATCHES:-4 2 1} --iters "${ITERS:-20}" --arch "${ARCH:-linear}" \
    --out "${OUT:?set OUT to the csv to write}"
