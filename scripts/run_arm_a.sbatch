#!/usr/bin/env bash
#SBATCH --job-name=neuro-arm-a
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
# Table 9, Arm A in full mode on the V100 nodes.
#
# Runs the generative-baseline arm end to end through the orchestrator, in the
# dependency order of section 14: E1 -> E2 (loss) -> E3 (backbone; winner
# defaults) -> E4 (dimensionality) -> E5 (ARD); E8 and E6 after E3; E9 after E6.
# The winner of each selection experiment is propagated automatically. A
# leaderboard is written to outputs/experiments/leaderboard.{csv,md}.
#
# Requires the real data placed in:
#   data/Full data/lesions/          lesion masks (binary, MNI 2mm)
#   data/Full data/disconnectomes/   disconnection maps (continuous, MNI 2mm), same id per patient
#   data/atlases/                    functional parcellation and Giles subdivisions
# (the pilot subset lives in data/Trial data/...; select it by exporting
#  NEUROCAUSAL_DATA_TIER=trial or passing --data-tier trial)
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

# Logging: set these to push to Weights & Biases; otherwise the local JSON/CSV
# backend is used automatically.
#   export WANDB_API_KEY=...     ; export NEUROCAUSAL_LOGGER=wandb
# or MLflow:
#   export MLFLOW_TRACKING_URI=...; export NEUROCAUSAL_LOGGER=mlflow

SEEDS="${SEEDS:-3}"
OUT="${OUT:-outputs/experiments}"

python -m neurocausalpfn.experiments.runner \
    --arm A --mode full --seeds "${SEEDS}" --out-root "${OUT}" --report

# To run the whole programme (all arms + E11a audit + E12 curriculum):
#   python -m neurocausalpfn.experiments.runner --all --mode full --seeds 3 \
#       --out-root outputs/experiments --report
#
# To run a single experiment (e.g. the backbone ablation E3):
#   python -m neurocausalpfn.experiments.runner --experiment E3 --mode full --seeds 3
