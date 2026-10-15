#!/bin/bash -l
# F33 provenance: re-encode a latents file's first images from each candidate
# directory and compare with the stored codes (CPU, minutes):
#   export CANDIDATES="$HOME/Scratch/isles_qdisco $HOME/Scratch/isles_disco"
#   qsub -v LATENTS=outputs/latents_isles_disco/E1_seed0_disco.npz,CKPT=<vae_disconnectome.pt>,CANDIDATES \
#       scripts/which_image_dir_myriad.qsub.sh
#
#$ -N which-dir
#$ -l h_rt=0:30:0
#$ -pe smp 2
#$ -l mem=8G
#$ -cwd
#$ -j y
set -euo pipefail
trap 'exit 100' ERR

module load python3/3.11
source ~/venvs/neuro/bin/activate

LATENTS="${LATENTS:?set LATENTS to the latents npz}"
CKPT="${CKPT:?set CKPT to the checkpoint that produced it}"
CANDIDATES="${CANDIDATES:?set CANDIDATES to the directories to test (space-separated)}"
# word splitting of CANDIDATES is intended: one argument per directory
# shellcheck disable=SC2086
python scripts/which_image_dir.py --latents "$LATENTS" --checkpoint "$CKPT" \
    --candidates $CANDIDATES --limit "${LIMIT:-16}"
