#!/bin/bash -l
# Giles virtual-trial replica on the full cohort (CPU-only, no GPU needed).
#
#   qsub scripts/run_giles_replica_myriad.qsub.sh                      # ideal scenario
#   qsub -v SCENARIO=location_bias scripts/run_giles_replica_myriad.qsub.sh
#   qsub -v IMAGES="data/Full data/lesions" scripts/run_giles_replica_myriad.qsub.sh
#   qsub -v LATENTS="outputs/foo/latents.npz" ... to score encoder latents too
#   qsub -v NMF_PER_FOLD=1 ... to refit NMF per fold (paper protocol, leak-free)
#   qsub -v GROUP_TABLE=outputs/groups_public.csv,OUT=outputs/giles_replica_ideal_grp ... group-aware folds
#   qsub -v SUBSAMPLE=1207,SUBSAMPLE_SEED=0,OUT=... size-matched control for an external cohort
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
# GROUP_TABLE=outputs/groups_public.csv (filename, group, rank; kept out of
# git) switches to group-aware folds; GROUP_MODE=giles|strict. Set OUT
# explicitly to keep the image-level results apart. The variable is NOT called
# GROUPS: bash pre-defines GROUPS as the caller's numeric group ids, so a job
# that never set it still passed "--groups 4527" and crashed.
if [ -n "${GROUP_TABLE:-}" ]; then
    EXTRA+=(--groups "$GROUP_TABLE" --group-mode "${GROUP_MODE:-giles}")
fi
# SUBSAMPLE=1207 (and SUBSAMPLE_SEED) runs the replica on a random subset of
# the images: the size-matched internal control for a smaller external cohort.
if [ -n "${SUBSAMPLE:-}" ]; then
    EXTRA+=(--subsample "$SUBSAMPLE" --subsample-seed "${SUBSAMPLE_SEED:-0}")
fi
# SECOND_IMAGES=<dir of the other channel, same file names> enables the
# nmf50_both builtin (BUILTIN="volume nmf50_both"): one per-fold NMF per
# channel, factors concatenated, the linear counterpart of E6 "both".
if [ -n "${SECOND_IMAGES:-}" ]; then
    EXTRA+=(--second-images-dir "$SECOND_IMAGES")
fi
# LATENT_PCA=100 reduces wide external embeddings (--latents) with a PCA
# fitted on the train side of each fold.
if [ -n "${LATENT_PCA:-}" ]; then
    EXTRA+=(--latent-pca "$LATENT_PCA")
fi
# PFN_CKPT=outputs/<run>/pfn.pt scores the trained transformer in context on
# the same folds, next to the sklearn classifiers (ONLY_PFN=1 to skip them).
if [ -n "${PFN_CKPT:-}" ]; then
    EXTRA+=(--pfn-ckpt "$PFN_CKPT")
    if [ -n "${ONLY_PFN:-}" ]; then EXTRA+=(--only-pfn); fi
fi
# CAUSALPFN=1 adds the off-the-shelf CausalPFN (fixed weights, the design's
# Tier-4 evaluator); ONLY_CAUSALPFN=1 scores it alone. Weights must already be
# in ~/.cache/causalpfn (download once on a login node).
if [ -n "${ONLY_CAUSALPFN:-}" ]; then
    EXTRA+=(--only-causalpfn)
elif [ -n "${CAUSALPFN:-}" ]; then
    EXTRA+=(--causalpfn)
fi

python scripts/run_giles_replica.py \
    --images-dir "$IMAGES" \
    --atlas-dir data/atlases --modality "$MODALITY" \
    --scenario "$SCENARIO" --builtin $BUILTIN \
    --out "$OUT" ${EXTRA[@]+"${EXTRA[@]}"}
