#!/bin/bash -l
# F34 Q11: clipping and effect loss of a theory-family hyper-prior on a prior cache (CPU):
#   qsub -v CACHE=outputs/prior_cache/kch_e1_32d_x.npz,OUT=outputs/prior_diag_x.csv,HYPER="two_arms=1 effect_scale=0.2:1 effect_headroom=1" \
#       scripts/prior_diagnostics_myriad.qsub.sh
# HYPER is space-separated key=value items (qsub -v cannot carry commas inside
# a value, so export it and pass -v HYPER instead when it has spaces).
#
#$ -N prior-diag
#$ -l h_rt=1:0:0
#$ -pe smp 2
#$ -l mem=8G
#$ -cwd
#$ -j y
set -euo pipefail
trap 'exit 100' ERR

module load python3/3.11
source ~/venvs/neuro/bin/activate

CACHE="${CACHE:?set CACHE to the prior cache npz}"
OUT="${OUT:?set OUT to the csv}"
python scripts/prior_diagnostics.py --cache "$CACHE" --atlas-dir "${ATLAS_DIR:-data/atlases}" \
    --hyper "${HYPER:-two_arms=1 effect_scale=0.2:1}" --processes "${PROCESSES:-200}" --out "$OUT"
