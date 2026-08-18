#!/bin/bash -l
# Runs the test suite on a Myriad COMPUTE node, where using several cores is
# allowed. Do not run `pytest` on a login node: Arbiter2 limits each user to
# under 6 cores / 30 GB there and applies escalating penalties.
#
#   cd ~/Scratch/neuro-causal-pfn
#   qsub scripts/run_tests_myriad.qsub.sh
#   tail -f ncp-tests.o*        # watch progress
#
#$ -N ncp-tests
#$ -l h_rt=2:0:0
#$ -pe smp 4
#$ -l mem=4G
#$ -l tmpfs=10G
#$ -cwd
#$ -j y
set -euo pipefail

module load python3/3.11
source ~/venvs/neuro/bin/activate

export OMP_NUM_THREADS=${NSLOTS:-4}
python -m pytest tests/ -q
