#!/bin/bash -l
# First real Neuro-Causal-PFN training (reduced configuration, one GPU).
#
#   VARIANT=nocurr|ctx|ctx+stages selects an E12 curriculum variant (scripts/myriad_phase1b.sh e12).
#   qsub -v CACHE=outputs/prior_cache/kch_E1_seed0.npz,OUT=outputs/pfn_reduced_kch_E1 \
#        scripts/train_pfn_myriad.qsub.sh
#   ITERS=20000 (default), SEED=0, ARCH=tabicl|linear
#
# Then score it on the Phase 1 folds (CPU):
#   qsub -v LATENTS="outputs/latents_E1/E1_seed0_disco.npz",GROUP_TABLE=outputs/groups_public.csv,\
#        PFN_CKPT=outputs/pfn_reduced_kch_E1/pfn.pt,OUT=outputs/giles_replica_pfn scripts/run_giles_replica_myriad.qsub.sh
#
#$ -N ncp-pfn
#$ -l h_rt=36:0:0
#$ -l gpu=1
#$ -pe smp 8
#$ -l mem=8G
#$ -l tmpfs=10G
#$ -cwd
#$ -j y
set -euo pipefail

module load python3/3.11
source ~/venvs/neuro/bin/activate

CACHE="${CACHE:?set CACHE to the npz from scripts/build_prior_cache.py}"
OUT="${OUT:-outputs/pfn_reduced}"
ITERS="${ITERS:-20000}"
SEED="${SEED:-0}"
ARCH="${ARCH:-tabicl}"
VARIANT="${VARIANT:-}"      # E12: nocurr | ctx | ctx+stages (empty = the pilot's reduced config)

python - <<PY
import json
from neurocausalpfn.train.train_pfn import e12_config, reduced_config, run_pfn
cfg = e12_config("$VARIANT") if "$VARIANT" else reduced_config()
cfg["out_dir"] = "$OUT"; cfg["seed"] = int("$SEED")
cfg["pfn"]["iters"] = int("$ITERS"); cfg["pfn"]["arch"] = "$ARCH"
cfg["prior"]["cache"] = "$CACHE"; cfg["prior"]["atlas_dir"] = "data/atlases"
print(json.dumps(cfg, indent=1, default=str))
model, history = run_pfn(cfg)
json.dump(history, open("$OUT/history.json", "w"))
PY
