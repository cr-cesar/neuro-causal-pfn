#!/bin/bash -l
# Clinical validation of frozen representations on a cohort with real outcomes (CPU).
#
#   qsub -v COHORT=outputs/kch_cohort.csv,IDCOL=sub,IMAGES=$HOME/Scratch/kch_clean,\
#        LATENTS="outputs/latents_kch/*.npz outputs/latents_kch_disco/*.npz",\
#        OUTCOMES="nihss_total:>=5 nihss_total:>=16",COVARS="Age Sex",VOLCOL=vol_mni_ml,\
#        REGRESS="nihss_total",OUT=outputs/clinical_kch.csv scripts/clinical_validation_myriad.qsub.sh
#
#$ -N clinical
#$ -l h_rt=4:0:0
#$ -pe smp 4
#$ -l mem=8G
#$ -l tmpfs=5G
#$ -cwd
#$ -j y
set -euo pipefail

module load python3/3.11
source ~/venvs/neuro/bin/activate

COHORT="${COHORT:?set COHORT}"
IDCOL="${IDCOL:-sub}"
IMAGES="${IMAGES:?set IMAGES (the listing the latents are aligned to)}"
OUT="${OUT:-outputs/clinical_$(basename "$COHORT" .csv).csv}"

ARGS=(--cohort "$COHORT" --id-col "$IDCOL" --images-dir "$IMAGES" --out "$OUT" --with-covariates)
# shellcheck disable=SC2206
if [ -n "${LATENTS:-}" ]; then ARGS+=(--latents ${LATENTS}); fi
for o in ${OUTCOMES:-}; do ARGS+=(--outcome "$o"); done
for r in ${REGRESS:-}; do ARGS+=(--regress "$r"); done
# shellcheck disable=SC2206
if [ -n "${COVARS:-}" ]; then ARGS+=(--covariates ${COVARS}); fi
if [ -n "${VOLCOL:-}" ]; then ARGS+=(--volume-col "$VOLCOL"); fi
if [ -n "${REPEATS:-}" ]; then ARGS+=(--repeats "$REPEATS"); fi

python scripts/clinical_validation.py "${ARGS[@]}"
