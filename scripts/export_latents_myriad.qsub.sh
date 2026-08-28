#!/bin/bash -l
# Latent export from frozen experiment checkpoints (short GPU job).
#
#   qsub -v CKPTS="outputs/experiments/E2/E2[w_dice=0.5]/seed0/disco/vae_disconnectome.pt" \
#        scripts/export_latents_myriad.qsub.sh
#   qsub -v CKPTS="outputs/experiments/E4/*/seed0/disco/vae_disconnectome.pt",IMAGES="data/Full data/disconnectomes" ...
#
# Then feed the npz files to the replica:
#   qsub -v LATENTS="outputs/latents/*.npz" scripts/run_giles_replica_myriad.qsub.sh
#
#$ -N latent-export
#$ -l h_rt=4:0:0
#$ -l gpu=1
#$ -pe smp 4
#$ -l mem=8G
#$ -l tmpfs=10G
#$ -cwd
#$ -j y
set -euo pipefail

module load python3/3.11
source ~/venvs/neuro/bin/activate

IMAGES="${IMAGES:-data/Full data/disconnectomes}"
OUT="${OUT:-outputs/latents}"

if [ -z "${CKPTS:-}" ]; then
    echo "set CKPTS to one or more checkpoint globs (space-separated)" >&2
    exit 1
fi

# CKPTS may contain several space-separated globs; expand them here
# shellcheck disable=SC2086
python scripts/export_latents.py \
    --checkpoints ${CKPTS} \
    --images-dir "$IMAGES" \
    --out "$OUT" --device auto
