#!/usr/bin/env bash
#SBATCH --job-name=neuro-causal-pfn
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
# Plantilla para el cluster de UCL. Ajusta particion y cuenta segun proceda.
# Requiere los datos reales colocados en:
#   data/lesions/         mascaras de lesion (binarias, MNI 2mm)
#   data/disconnectomes/  mapas de disconexion (continuos, MNI 2mm), mismo id por paciente
#   data/atlases/         parcelacion funcional y subdivisiones de Giles
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

# Etapa 1: dos VAEs entrenados por separado, cada uno con su perdida y exportando
# sus latentes congelados (lesion: BCE mas Dice; disconnectoma: MSE continuo).
python -m neurocausalpfn.train.train_vae --mode full --representation lesion
python -m neurocausalpfn.train.train_vae --mode full --representation disconnectome

# Etapa 2: cablea los latentes reales como covariable del Neuro-Prior y entrena
# el transformer con la codificacion tabular estilo TabICL. fusion_mode en
# stage2_real_config elige la variante: lesion, disconnectome o both.
python -m neurocausalpfn.train.run_stage2_real

# Para reanudar la Etapa 1 si el trabajo se corta, anadir --resume con la ruta
# del checkpoint, por ejemplo:
#   python -m neurocausalpfn.train.train_vae --mode full --representation lesion \
#       --resume outputs/vae_full_lesion/vae_lesion_last.pt
