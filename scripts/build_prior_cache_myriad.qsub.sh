#!/bin/bash -l
# Precompute the Neuro-Prior anatomy cache of a covariate pool (CPU):
#   qsub -v LESIONS=<lesion dir>,LATENTS=<latents npz>,OUT=outputs/prior_cache/pool.npz scripts/build_prior_cache_myriad.qsub.sh
#
#$ -N prior-cache
#$ -l h_rt=6:0:0
#$ -pe smp 4
#$ -l mem=8G
#$ -l tmpfs=10G
#$ -cwd
#$ -j y
set -euo pipefail
# a failed step exits 100: SGE then keeps this job in error state and the jobs
# holding on it (-hold_jid) stay queued instead of starting on missing inputs
trap 'exit 100' ERR

module load python3/3.11
source ~/venvs/neuro/bin/activate

LESIONS="${LESIONS:?set LESIONS to the lesion directory of the pool}"
OUT="${OUT:?set OUT to the cache npz}"
ATLAS_DIR="${ATLAS_DIR:-data/atlases}"
[ -d "$LESIONS" ] || { echo "no lesion dir $LESIONS" >&2; exit 100; }
[ -z "${LATENTS:-}" ] || [ -s "$LATENTS" ] || { echo "no latents $LATENTS (the pool export failed?)" >&2; exit 100; }
mkdir -p "$(dirname "$OUT")"
python scripts/build_prior_cache.py --lesions-dir "$LESIONS" --atlas-dir "$ATLAS_DIR" \
    --modality "${MODALITY:-receptor}" ${LATENTS:+--latents "$LATENTS"} --out "$OUT"
