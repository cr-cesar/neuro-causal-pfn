#!/bin/bash -l
# Probe Neuro-Causal-PFN checkpoints (CPU): treatment sensitivity and accuracy
# per prior family on the checkpoint's own anatomy cache (scripts/probe_pfn.py).
#
#   CKPTS="outputs/pfn_reduced_kch_e1/pfn.pt" qsub -v CKPTS scripts/probe_pfn_myriad.qsub.sh
#   (or: bash scripts/myriad_phase1b.sh probe <pfn.pt> ...)
#
#$ -N probe-pfn
#$ -l h_rt=2:0:0
#$ -pe smp 4
#$ -l mem=8G
#$ -cwd
#$ -j y
set -euo pipefail

module load python3/3.11
source ~/venvs/neuro/bin/activate

CKPTS="${CKPTS:?set CKPTS to one or more pfn.pt paths (space-separated)}"
OUT="${OUT:-outputs/probe_pfn.csv}"
# shellcheck disable=SC2086
python scripts/probe_pfn.py --ckpt ${CKPTS} --out "$OUT" ${N_CONTEXT:+--n-context "$N_CONTEXT"}
