#!/usr/bin/env bash
# Smoke test on CPU: trains Stage 1 and Stage 2 in prototype mode.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

echo ">> Stage 1: lesion autoencoder (prototype)"
python -m neurocausalpfn.train.train_vae --mode prototype

echo ">> Stage 2: causal transformer (prototype)"
python -m neurocausalpfn.train.train_pfn --mode prototype
