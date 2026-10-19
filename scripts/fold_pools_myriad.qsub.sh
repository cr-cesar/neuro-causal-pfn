#!/bin/bash -l
# Per-fold anatomy pools for the Neuro-Prior (CPU, seconds to minutes): one
# cache per fold with the training-side rows of the cohort cache and the fold
# encoder's latents (scripts/build_fold_pool_cache.py).
#
#   qsub -v CACHE=outputs/prior_cache/uclh_disco_x.npz,FOLD_DIR=<perfold>/outputs_perfold_pub/E1/E1/seed0/folds,\
#        OUT_DIR=outputs/prior_cache/fold_pools/e1_s0,FOLDS="0 1" scripts/fold_pools_myriad.qsub.sh
#   EXTRA_CACHE=<kch cache> EXTRA_LATENTS="<npz per fold, {fold} in the path>" appends a second cohort.
#
#$ -N fold-pools
#$ -l h_rt=0:30:0
#$ -l mem=8G
#$ -cwd
#$ -j y
set -euo pipefail
trap 'exit 100' ERR

module load python3/3.11
source ~/venvs/neuro/bin/activate

CACHE="${CACHE:?set CACHE to the cohort's anatomy cache}"
FOLD_DIR="${FOLD_DIR:?set FOLD_DIR to the folder of fold<k>.npz files}"
OUT_DIR="${OUT_DIR:?set OUT_DIR}"
ARGS=()
if [ -n "${EXTRA_CACHE:-}" ]; then ARGS+=(--extra-cache "$EXTRA_CACHE" --extra-latents "${EXTRA_LATENTS:?set EXTRA_LATENTS with EXTRA_CACHE}"); fi
# shellcheck disable=SC2086
python scripts/build_fold_pool_cache.py --cache "$CACHE" --fold-dir "$FOLD_DIR" --out-dir "$OUT_DIR" \
    --folds ${FOLDS:-0 1 2 3 4 5 6 7 8 9} ${ARGS[@]+"${ARGS[@]}"}
