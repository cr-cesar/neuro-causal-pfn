#!/usr/bin/env bash
#SBATCH --job-name=neuro-causal-pfn
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
# Plantilla para el cluster de UCL. Ajusta particion y cuenta segun proceda.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

python -m neurocausalpfn.train.train_vae --mode full
# tras entrenar la Etapa 1, exportar la representacion congelada y luego:
python -m neurocausalpfn.train.train_pfn --mode full
