#!/usr/bin/env bash
#SBATCH --job-name=neuro-causal-pfn
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
# Template for the UCL cluster. Adjust partition and account as appropriate.
# Requires the real data placed in:
#   data/lesions/         lesion masks (binary, MNI 2mm)
#   data/disconnectomes/  disconnection maps (continuous, MNI 2mm), same id per patient
#   data/atlases/         functional parcellation and Giles subdivisions
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

# Stage 1: two VAEs trained separately, each with its own loss and exporting
# their frozen latents (lesion: BCE plus Dice; disconnectome: continuous MSE).
python -m neurocausalpfn.train.train_vae --mode full --representation lesion
python -m neurocausalpfn.train.train_vae --mode full --representation disconnectome

# Stage 2: wires the real latents as the Neuro-Prior covariate and trains
# the transformer with the TabICL-style tabular encoding. fusion_mode in
# stage2_real_config chooses the variant: lesion, disconnectome or both.
python -m neurocausalpfn.train.run_stage2_real

# To resume Stage 1 if the job is interrupted, add --resume with the checkpoint
# path, for example:
#   python -m neurocausalpfn.train.train_vae --mode full --representation lesion \
#       --resume outputs/vae_full_lesion/vae_lesion_last.pt
