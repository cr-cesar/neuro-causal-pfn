#!/bin/bash -l
# Giles virtual-trial replica on the full cohort (CPU-only, no GPU needed).
#
#   qsub scripts/run_giles_replica_myriad.qsub.sh                      # ideal scenario
#   qsub -v SCENARIO=location_bias scripts/run_giles_replica_myriad.qsub.sh
#   qsub -v IMAGES="data/Full data/lesions" scripts/run_giles_replica_myriad.qsub.sh
#   qsub -v LATENTS="outputs/foo/latents.npz" ... to score encoder latents too
#   qsub -v NMF_PER_FOLD=1 ... to refit NMF per fold (paper protocol, leak-free)
#
#$ -N giles-replica
#$ -l h_rt=8:0:0
#$ -pe smp 4
#$ -l mem=4G
#$ -l tmpfs=10G
#$ -cwd
#$ -j y
set -euo pipefail

module load python3/3.11
source ~/venvs/neuro/bin/activate

SCENARIO="${SCENARIO:-ideal}"
IMAGES="${IMAGES:-data/Full data/disconnectomes}"
MODALITY="${MODALITY:-receptor}"
OUT="${OUT:-outputs/giles_replica_${SCENARIO}}"
# BUILTIN="volume nmf50_nimfa" runs Giles' exact nimfa NMF (needs `pip install
# nimfa` in the venv, icv_mask_2mm.nii.gz in data/atlases, and more memory:
# resubmit with `qsub -l mem=10G ...` — the dense per-fold matrix is ~7 GB).
BUILTIN="${BUILTIN:-volume nmf50}"

EXTRA=()
if [ -n "${LATENTS:-}" ]; then
    EXTRA+=(--latents ${LATENTS})
fi
if [ -n "${FOLD_LATENTS:-}" ]; then
    EXTRA+=(--fold-latents ${FOLD_LATENTS})
fi
if [ -n "${NMF_PER_FOLD:-}" ]; then
    EXTRA+=(--nmf-per-fold)
fi

python scripts/run_giles_replica.py \
    --images-dir "$IMAGES" \
    --atlas-dir data/atlases --modality "$MODALITY" \
    --scenario "$SCENARIO" --builtin $BUILTIN \
    --out "$OUT" ${EXTRA[@]+"${EXTRA[@]}"}
