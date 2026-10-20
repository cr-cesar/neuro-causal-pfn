#!/bin/bash -l
# Latents of one image folder with every fold encoder of a per-fold
# representation, on CPU (no GPU queue): one npz per fold, in its own folder,
# because the export names a file by experiment, seed and channel only.
# Used for a second cohort joining the per-fold pools (EXTRA_LATENTS of
# scripts/build_fold_pool_cache.py).
#
#   qsub -v FOLD_ROOT=<perfold>/outputs_perfold_pub/E1/E1/seed0,IMAGES=$HOME/Scratch/kch_qdisco,\
#        OUT_ROOT=outputs/latents_pf/seed0,FOLDS="0 1" scripts/export_latents_cpu.qsub.sh
#   -> OUT_ROOT/fold<k>/*.npz; CHANNEL (default disco) picks the encoder of that channel.
#
#$ -N latent-cpu
#$ -l h_rt=8:0:0
#$ -pe smp 4
#$ -l mem=8G
#$ -cwd
#$ -j y
set -euo pipefail
trap 'exit 100' ERR

module load python3/3.11
source ~/venvs/neuro/bin/activate

FOLD_ROOT="${FOLD_ROOT:?set FOLD_ROOT to the representation folder holding fold<k>/}"
IMAGES="${IMAGES:?set IMAGES to the image folder to encode}"
OUT_ROOT="${OUT_ROOT:?set OUT_ROOT}"
CHANNEL="${CHANNEL:-disco}"
for k in ${FOLDS:-0 1 2 3 4 5 6 7 8 9}; do
    ckpt="$FOLD_ROOT/fold$k/$CHANNEL/vae_${CHANNEL/disco/disconnectome}.pt"
    [ -s "$ckpt" ] || { echo "no encoder for fold $k under $FOLD_ROOT"; exit 100; }
    echo "fold $k: $(date +%H:%M:%S)"
    python scripts/export_latents.py --checkpoints "$ckpt" --images-dir "$IMAGES" \
        --out "$OUT_ROOT/fold$k" --device cpu
done
echo "done: $(date +%H:%M:%S)"
