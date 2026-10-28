#!/bin/bash -l
# F32: score per-fold representations on the extended continuous prior (CPU).
#   qsub -v REPS=<root>/E1/E1/seed0/folds,CACHE=outputs/prior_cache/uclh_disco.npz,LABEL=E1,OUT=outputs/cp_E1.csv scripts/score_continuous_prior.qsub.sh
#   VOLUME=1 scores log lesion volume from the cache instead of the latents;
#   NMF=1 refits NMF-50 per fold on the cache's images (folds taken from REPS).
#
#$ -N cp-score
#$ -l h_rt=4:0:0
#$ -pe smp 4
#$ -l mem=8G
#$ -cwd
#$ -j y
set -euo pipefail
trap 'exit 100' ERR

module load python3/3.11
source ~/venvs/neuro/bin/activate

REPS="${REPS:?set REPS to a directory with fold*.npz}"
CACHE="${CACHE:?set CACHE to the prior cache of the same cohort}"
OUT="${OUT:?set OUT to the csv}"
ATLAS_DIR="${ATLAS_DIR:-data/atlases}"
python scripts/score_continuous_prior.py --reps "$REPS" --cache "$CACHE" --atlas-dir "$ATLAS_DIR" \
    --hyper "${PRIOR_HYPER:-two_arms=1 effect_scale=0.2:1}" --processes "${PROCESSES:-20}" \
    ${VOLUME:+--volume} ${NMF:+--nmf} ${LABEL:+--label "$LABEL"} --out "$OUT"
