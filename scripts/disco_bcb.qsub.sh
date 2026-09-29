#!/bin/bash -l
# Structural disconnectome maps for a folder of binary lesion masks (MNI 2 mm)
# with the BCBtoolkit 5.x command line (pure Python): for every tractogram of
# the atlas the streamlines of 25-250 mm crossing the lesion are selected, the
# voxels they visit are marked, and the per-subject maps are averaged into a
# map of disconnection probability in [0, 1] named after the lesion. Each
# tractogram is read once per batch of lesions. One array task per chunk of
# lesions; finished lesions are skipped, so a killed array can be resubmitted
# as is. THRESHOLD defaults to 0 (raw probabilities, like the reference maps).
#
#   BCB=~/Scratch/BCBToolKit/bcb53/BCBToolKit_standalone   # folder holding the bcbtoolkit package
#   TRACKS=~/Scratch/BCBToolKit/tracks                       # the .trk atlas (178 subjects, 2 mm)
#   N=$(ls ~/Scratch/kch_clean/*.nii* | wc -l); CH=50
#   qsub -t 1-$(( (N + CH - 1) / CH )) \
#        -v BCB=$BCB,TRACKS=$TRACKS,LESIONS=$HOME/Scratch/kch_clean,OUT=$HOME/Scratch/kch_disco,CHUNK=$CH \
#        scripts/disco_bcb.qsub.sh
#
# Calibration first: run 50 training lesions and compare with the reference
# maps (scripts/compare_disconnectomes.py) before spending compute on a cohort.
#
#$ -N disco-bcb
#$ -l h_rt=12:0:0
#$ -l mem=8G
#$ -pe smp 4
#$ -l tmpfs=10G
#$ -cwd
#$ -j y
# strict mode only after the module/venv lines: `module load` can return
# non-zero on Myriad although the module is loaded, and set -e would then
# abort the job silently right after the module banner.
module load python3/3.11
source ~/venvs/neuro/bin/activate
set -euo pipefail

BCB="${BCB:?folder that contains the bcbtoolkit python package}"
LESIONS="${LESIONS:?folder of binary lesion masks on the MNI 2 mm grid}"
OUT="${OUT:?output folder for the disconnectome maps}"
CHUNK="${CHUNK:-50}"
TRACKS="${TRACKS:?folder of .trk tractograms}"
THRESHOLD="${THRESHOLD:-0}"
[ -d "$BCB/bcbtoolkit" ] || { echo "bcbtoolkit package not found under $BCB"; exit 1; }
mkdir -p "$OUT" "$OUT/logs"

mapfile -t files < <(ls "$LESIONS"/*.nii "$LESIONS"/*.nii.gz 2>/dev/null | sort)
start=$(( (SGE_TASK_ID - 1) * CHUNK ))
sel=("${files[@]:$start:$CHUNK}")
[ ${#sel[@]} -gt 0 ] || { echo "task $SGE_TASK_ID: no lesions in this chunk"; exit 0; }

work="${TMPDIR:-/tmp}/disco_${JOB_ID:-x}_$SGE_TASK_ID"
mkdir -p "$work/in" "$work/out"
todo=0
for f in "${sel[@]}"; do
    stem=$(basename "$f"); stem="${stem%.nii.gz}"; stem="${stem%.nii}"
    [ -f "$OUT/$stem.nii.gz" ] && continue          # resume: already computed
    cp "$f" "$work/in/"; todo=$((todo + 1))
done
[ $todo -gt 0 ] || { echo "task $SGE_TASK_ID: all ${#sel[@]} lesions already done"; exit 0; }

ntrk=$(ls "$TRACKS"/*.trk "$TRACKS"/*.bcbtrk 2>/dev/null | wc -l)
echo "task $SGE_TASK_ID: $todo lesions | $ntrk tractograms | threshold $THRESHOLD | $(date)"
cd "$BCB"
python -m bcbtoolkit disconnectome -l "$work/in" -r "$work/out" \
    --threshold "$THRESHOLD" --tracks "$TRACKS" --batch-size "$todo" \
    2>&1 | tee "$OUT/logs/task_${SGE_TASK_ID}.txt"

cp "$work/out"/*.nii* "$OUT"/
echo "task $SGE_TASK_ID: wrote $(ls "$work/out"/*.nii* | wc -l) maps to $OUT | $(date)"
