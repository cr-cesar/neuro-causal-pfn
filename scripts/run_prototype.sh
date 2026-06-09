#!/usr/bin/env bash
# Prueba de humo en CPU: entrena la Etapa 1 y la Etapa 2 en modo prototipo.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

echo ">> Etapa 1: autoencoder de lesion (prototipo)"
python -m neurocausalpfn.train.train_vae --mode prototype

echo ">> Etapa 2: transformer causal (prototipo)"
python -m neurocausalpfn.train.train_pfn --mode prototype
