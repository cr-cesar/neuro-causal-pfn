#!/bin/bash -l
# F33: aggregate descriptives of one cohort's maps with the PI's script (CPU):
#   export MAPS="$HOME/Scratch/kch_qdisco/*.nii*"
#   qsub -N desc-kch -v MAPS,OUT=status/descriptives_kch scripts/disco_descriptives_myriad.qsub.sh
# MAPS is a glob (quoted, expanded by the script); ATLAS_DIR defaults to the
# repository's copy of the Giles atlases. Aggregates only.
#
#$ -N disco-desc
#$ -l h_rt=1:0:0
#$ -pe smp 2
#$ -l mem=8G
#$ -cwd
#$ -j y
set -euo pipefail
trap 'exit 100' ERR

module load python3/3.11
source ~/venvs/neuro/bin/activate

MAPS="${MAPS:?set MAPS to a quoted glob of the cohort maps}"
OUT="${OUT:?set OUT to status/descriptives_<cohort>}"
python scripts/disco_descriptives.py --maps "$MAPS" --atlas-dir "${ATLAS_DIR:-data/atlases}" \
    --out "$OUT" --threshold "${THRESHOLD:-0.5}"
