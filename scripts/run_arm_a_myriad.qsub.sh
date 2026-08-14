#!/bin/bash -l
# UCL Myriad (Sun Grid Engine) job script for Phase 1, Arm A.
#
# Myriad schedules with SGE (qsub), not SLURM, so the *.sbatch scripts in this
# directory do not apply there. Submit this one from the repo root, which must
# live under ~/Scratch (jobs cannot write to $HOME):
#
#   cd ~/Scratch/neuro-causal-pfn
#   qsub scripts/run_arm_a_myriad.qsub.sh
#
# Monitor with `qstat`; output lands in neuro-arm-a.o<jobid> in the repo root.
#
#$ -N neuro-arm-a
#$ -l h_rt=48:0:0
#$ -l gpu=1
#$ -pe smp 8
#$ -l mem=6G
#$ -l tmpfs=20G
#$ -cwd
#$ -j y
#
# Note: on Myriad `-l mem` is PER CORE, so smp 8 x 6G = 48G total, matching the
# SLURM template. h_rt is wall-clock; 48 h is the Myriad maximum.
set -euo pipefail

# Software comes from modules; exact versions drift, so check `module avail
# python` and `module avail cuda` after a cluster upgrade.
module purge || true
module load default-modules 2>/dev/null || true
module load python/miniconda3 2>/dev/null || module load python3/recommended
module load cuda 2>/dev/null || true

# One-off environment creation (login node, before the first submission):
#   conda env create -f env/environment.cluster.yml && conda activate neuro-causal-pfn
#   pip install -e ".[imaging,baselines,cluster]"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate neuro-causal-pfn
fi

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

# Requires the real data placed in:
#   data/lesions/         lesion masks (binary, MNI 2mm)
#   data/disconnectomes/  disconnection maps (continuous, MNI 2mm), same id per patient
#   data/atlases/         functional parcellation and Giles subdivisions
SEEDS="${SEEDS:-3}"
OUT="${OUT:-outputs/experiments}"

python -m neurocausalpfn.experiments.runner \
    --arm A --mode full --seeds "${SEEDS}" --out-root "${OUT}" --report

# Whole programme (all arms + E11 audit + E12 curriculum):
#   python -m neurocausalpfn.experiments.runner --all --mode full --seeds 3 \
#       --out-root outputs/experiments --report
# Single experiment (e.g. the backbone ablation E3):
#   python -m neurocausalpfn.experiments.runner --experiment E3 --mode full --seeds 3
