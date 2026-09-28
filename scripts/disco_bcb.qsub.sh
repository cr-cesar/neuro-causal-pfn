#!/bin/bash -l
# Structural disconnectome maps for a folder of binary lesion masks (MNI 2 mm)
# with the BCBtoolkit batch script: run_disco.sh passes track_vis over every
# tractogram of the bundled 2 mm atlas (streamlines 25-250 mm crossing the
# lesion), binarises the visitation maps and averages them into a map of
# disconnection probability in [0, 1], named after the lesion. One array task
# per chunk of lesions; finished lesions are skipped, so a killed array can be
# resubmitted as is.
#
#   BCB=~/Scratch/BCBToolKit                 # unpacked distribution, run_disco.sh at its root
#   N=$(ls ~/Scratch/kch_clean/*.nii* | wc -l); CH=20
#   qsub -t 1-$(( (N + CH - 1) / CH )) \
#        -v BCB=$BCB,LESIONS=$HOME/Scratch/kch_clean,OUT=$HOME/Scratch/kch_disco,CHUNK=$CH \
#        scripts/disco_bcb.qsub.sh
#
# Calibration first: run 50 training lesions and compare with the reference
# maps (scripts/compare_disconnectomes.py) before spending compute on a cohort.
#
#$ -N disco-bcb
#$ -l h_rt=6:0:0
#$ -l mem=4G
#$ -pe smp 8
#$ -l tmpfs=20G
#$ -cwd
#$ -j y
set -euo pipefail

BCB="${BCB:?root of the unpacked BCBToolKit distribution}"
LESIONS="${LESIONS:?folder of binary lesion masks on the MNI 2 mm grid}"
OUT="${OUT:?output folder for the disconnectome maps}"
CHUNK="${CHUNK:-20}"
TRACKS="${TRACKS:-$BCB/Tools/extraFiles/tracks}"
[ -f "$BCB/run_disco.sh" ] || { echo "run_disco.sh not found under $BCB"; exit 1; }
mkdir -p "$OUT" "$OUT/logs"

mapfile -t files < <(ls "$LESIONS"/*.nii "$LESIONS"/*.nii.gz 2>/dev/null | sort)
start=$(( (SGE_TASK_ID - 1) * CHUNK ))
sel=("${files[@]:$start:$CHUNK}")
[ ${#sel[@]} -gt 0 ] || { echo "task $SGE_TASK_ID: no lesions in this chunk"; exit 0; }

work="${TMPDIR:-/tmp}/disco_${JOB_ID:-x}_$SGE_TASK_ID"
mkdir -p "$work/in" "$work/tmp"
todo=0
for f in "${sel[@]}"; do
    stem=$(basename "$f"); stem="${stem%.nii.gz}"; stem="${stem%.nii}"
    [ -f "$OUT/$stem.nii.gz" ] && continue          # resume: already computed
    cp "$f" "$work/in/"; todo=$((todo + 1))
done
[ $todo -gt 0 ] || { echo "task $SGE_TASK_ID: all ${#sel[@]} lesions already done"; exit 0; }

ntrk=$(ls "$TRACKS"/*.trk 2>/dev/null | wc -l)
echo "task $SGE_TASK_ID: $todo lesions | $ntrk tractograms | ${NSLOTS:-1} cores | $(date)"
bash "$BCB/run_disco.sh" -l "$work/in" -o "$work/out" -n "${NSLOTS:-1}" -T "$TRACKS" -w "$work/tmp"

cp "$work/out"/*.nii.gz "$OUT"/
cp "$work/out"/logs/* "$OUT/logs/" 2>/dev/null || true
echo "task $SGE_TASK_ID: wrote $(ls "$work/out"/*.nii.gz | wc -l) maps to $OUT | $(date)"
