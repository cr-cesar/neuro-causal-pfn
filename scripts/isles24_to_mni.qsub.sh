#!/bin/bash -l
# Register ISLES'24 final-infarct masks to MNI152 2 mm (91x109x91), one array
# task per case. The follow-up DWI (same NCCT-space grid as the mask) drives
# an affine registration to the MNI152 T1 2 mm template; the mask follows with
# nearest-neighbour interpolation so it stays binary.
#
#   N=$(ls -d "$ISLES/derivatives"/sub-*/ | wc -l)
#   qsub -t 1-$N -v ISLES=~/Scratch/ISLES24/train,OUT=~/Scratch/ISLES24/mni2mm \
#        scripts/isles24_to_mni.qsub.sh
#   python scripts/isles24_qc.py ~/Scratch/ISLES24/train ~/Scratch/ISLES24/mni2mm
#
# Output per case: <OUT>/<sub>_lesion-msk_mni2mm.nii.gz (binary, uint8), the
# affine matrix and the registered DWI for visual QC.
#
#$ -N isles-mni
#$ -l h_rt=0:30:0
#$ -l mem=6G
#$ -l tmpfs=5G
#$ -pe smp 2
#$ -cwd
#$ -j y
set -euo pipefail

# Myriad loads the Intel compilers by default and fsl/6.0.4 requires
# compilers/gnu/4.9.2; unload the defaults first or the module fails.
module unload compilers mpi 2>/dev/null || true
module load "${FSL_MODULE:-fsl}"
export FSLOUTPUTTYPE=NIFTI_GZ

ISLES="${ISLES:?root of the ISLES24 release (contains derivatives/)}"
OUT="${OUT:-$ISLES/mni2mm}"
mkdir -p "$OUT"
REF="$FSLDIR/data/standard/MNI152_T1_2mm"
REF_BRAIN="$FSLDIR/data/standard/MNI152_T1_2mm_brain"

subs=($(ls -d "$ISLES"/derivatives/sub-*/ | xargs -n1 basename | sort))
sub="${subs[$((SGE_TASK_ID - 1))]}"
ses="$ISLES/derivatives/$sub/ses-02"
dwi=$(ls "$ses"/*_space-ncct_dwi.nii.gz | head -1)
msk=$(ls "$ses"/*_space-ncct_lesion-msk.nii.gz | head -1)
[ -f "$dwi" ] && [ -f "$msk" ] || { echo "$sub: missing dwi or mask"; exit 0; }

work="${TMPDIR:-/tmp}/$sub"; mkdir -p "$work"; cd "$work"
echo "$sub: $(basename "$dwi") + $(basename "$msk")"

# 1. same standard orientation for both (LAS and LPS cases in the release)
fslreorient2std "$dwi" dwi_std
fslreorient2std "$msk" msk_std
fslmaths msk_std -bin msk_bin -odt char

# 2. brain-extract the DWI (robust centre; b1000 keeps skull faint)
bet dwi_std dwi_brain -R -f 0.3

# 3. affine DWI -> MNI, mutual information across modalities, wide search
flirt -in dwi_brain -ref "$REF_BRAIN" -dof 12 -cost mutualinfo \
      -searchrx -90 90 -searchry -90 90 -searchrz -90 90 \
      -omat dwi2mni.mat -out dwi_mni

# 4. mask follows the same transform, nearest neighbour, template grid
flirt -in msk_bin -ref "$REF" -applyxfm -init dwi2mni.mat \
      -interp nearestneighbour -out msk_mni
fslmaths msk_mni -bin "$OUT/${sub}_lesion-msk_mni2mm" -odt char
cp dwi2mni.mat "$OUT/${sub}_dwi2mni.mat"
cp dwi_mni.nii.gz "$OUT/${sub}_dwi_mni2mm.nii.gz"

v0=$(fslstats msk_bin -V | awk '{print $2}')
v1=$(fslstats "$OUT/${sub}_lesion-msk_mni2mm" -V | awk '{print $2}')
echo "$sub: volume native ${v0} mm3 -> MNI ${v1} mm3"
