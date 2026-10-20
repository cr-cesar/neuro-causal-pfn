#!/bin/bash -l
# F42: which disconnectome generator reproduces Giles's Supplementary Table 2
# (CPU). One job builds the sample's disconnectomes with every listed set of
# QuickDisco priors and compares the 16 enrolment fractions per split:
#   export PRIORS="V2.P.WB=$HOME/Scratch/fcntm_priors/priors_full_proba_3T_comp9_thr0p01.h5 V2.D.WB=$HOME/Scratch/fcntm_priors/priors_full_deter3T.h5"
#   qsub -v PRIORS,OUT=status/<day>/f42_prior_match.csv scripts/disco_prior_match_myriad.qsub.sh
# Optional: N (default 200), RECEIVED (the received maps; default data/Full data/disconnectomes),
# MAPS_DIRS ("LABEL=DIR ..." for maps built elsewhere), SAMPLE_FROM (draw the sample
# only among lesions with a map in that folder), LESIONS, ATLAS_DIR, WORK.
#
#$ -N disco-prior
#$ -l h_rt=12:0:0
#$ -pe smp 4
#$ -l mem=6G
#$ -cwd
#$ -j y
set -euo pipefail
trap 'exit 100' ERR

module load python3/3.11
source ~/venvs/neuro/bin/activate

OUT="${OUT:?set OUT to the csv to write, e.g. status/<day>/f42_prior_match.csv}"
LESIONS="${LESIONS:-data/Full data/lesions}"
RECEIVED="${RECEIVED:-data/Full data/disconnectomes}"
ATLAS_DIR="${ATLAS_DIR:-data/atlases}"
WORK="${WORK:-$HOME/Scratch/disco_prior_match}"

ARGS=()
for p in ${PRIORS:-}; do ARGS+=(--priors "$p"); done
for m in ${MAPS_DIRS:-}; do ARGS+=(--maps-dir "$m"); done
if [ -n "${SAMPLE_FROM:-}" ]; then ARGS+=(--sample-from "$SAMPLE_FROM"); fi
python scripts/disco_prior_match.py --lesions "$LESIONS" --received "$RECEIVED" \
    --atlas-dir "$ATLAS_DIR" --work "$WORK" --n "${N:-200}" --procs "${NSLOTS:-4}" \
    --out "$OUT" ${ARGS[@]+"${ARGS[@]}"}
