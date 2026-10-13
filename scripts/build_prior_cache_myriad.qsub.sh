#!/bin/bash -l
# Precompute the Neuro-Prior anatomy cache of a covariate pool (CPU):
#   qsub -v IMAGES=<image dir of the pool>,LATENTS=<latents npz>,OUT=outputs/prior_cache/pool.npz scripts/build_prior_cache_myriad.qsub.sh
#
# IMAGES must be the directory the latents were exported from (POOL_IMAGES of
# the pool block in scripts/myriad_phase1b.sh): the cache's overlaps decide
# which lesions a process can affect, the latents are the covariates, and the
# evaluation labels the scored images with the same rule, so all three must
# come from one channel. build_prior_cache.py detects the channel of IMAGES
# (binary masks = lesion, continuous maps = disconnectome), compares it with
# the latents' and stops on a mismatch; ALLOW_CHANNEL_MISMATCH=1 overrides.
# LESIONS is the former name of IMAGES and is still accepted with a warning.
# MODALITY is the atlas subdivision (receptor | genetics), not the image channel.
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

if [ -z "${IMAGES:-}" ] && [ -n "${LESIONS:-}" ]; then
    echo "warning: LESIONS is the former name of IMAGES; the cache must be built from the directory the latents were exported from" >&2
    IMAGES="$LESIONS"
fi
IMAGES="${IMAGES:?set IMAGES to the image directory of the pool (the one the latents were exported from)}"
OUT="${OUT:?set OUT to the cache npz}"
ATLAS_DIR="${ATLAS_DIR:-data/atlases}"
[ -d "$IMAGES" ] || { echo "no image dir $IMAGES" >&2; exit 100; }
[ -z "${LATENTS:-}" ] || [ -s "$LATENTS" ] || { echo "no latents $LATENTS (the pool export failed?)" >&2; exit 100; }
mkdir -p "$(dirname "$OUT")"
python scripts/build_prior_cache.py --images-dir "$IMAGES" --atlas-dir "$ATLAS_DIR" \
    --modality "${MODALITY:-receptor}" ${LATENTS:+--latents "$LATENTS"} \
    ${ALLOW_CHANNEL_MISMATCH:+--allow-channel-mismatch} --out "$OUT"
