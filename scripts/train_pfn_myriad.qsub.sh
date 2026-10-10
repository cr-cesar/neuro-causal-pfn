#!/bin/bash -l
# First real Neuro-Causal-PFN training (reduced configuration, one GPU).
#
#   VARIANT=nocurr|ctx|ctx+stages selects an E12 curriculum variant (scripts/myriad_phase1b.sh e12).
#   FAMILY=theory|giles|mixture selects the Neuro-Prior family (empty = theory, the pilot's);
#   FAMILY_WEIGHTS="0.5 0.5" sets the mixture's theory / giles weights (space-separated).
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
# a failed step exits 100: SGE then keeps this job in error state and the jobs
# holding on it (-hold_jid) stay queued instead of starting on missing inputs
trap 'exit 100' ERR

module load python3/3.11
source ~/venvs/neuro/bin/activate

CACHE="${CACHE:?set CACHE to the npz from scripts/build_prior_cache.py}"
[ -s "$CACHE" ] || { echo "no prior cache $CACHE (the cache job failed?)" >&2; exit 100; }
OUT="${OUT:-outputs/pfn_reduced}"
ITERS="${ITERS:-20000}"
SEED="${SEED:-0}"
ARCH="${ARCH:-tabicl}"
VARIANT="${VARIANT:-}"      # E12: nocurr | ctx | ctx+stages (empty = the pilot's reduced config)
FAMILY="${FAMILY:-}"        # theory | giles | mixture (empty = theory)
FAMILY_WEIGHTS="${FAMILY_WEIGHTS:-}"

python - <<PY
import json
from neurocausalpfn.train.train_pfn import e12_config, reduced_config, run_pfn, with_family
fam = "$FAMILY" or "theory"
w = [float(x) for x in "$FAMILY_WEIGHTS".split()] or None
cfg = e12_config("$VARIANT", fam, w) if "$VARIANT" else with_family(reduced_config(), fam, w)
cfg["out_dir"] = "$OUT"; cfg["seed"] = int("$SEED")
cfg["pfn"]["iters"] = int("$ITERS"); cfg["pfn"]["arch"] = "$ARCH"
cfg["prior"]["cache"] = "$CACHE"; cfg["prior"]["atlas_dir"] = "data/atlases"
print(json.dumps(cfg, indent=1, default=str))
model, history = run_pfn(cfg)
json.dump(history, open("$OUT/history.json", "w"))
PY
