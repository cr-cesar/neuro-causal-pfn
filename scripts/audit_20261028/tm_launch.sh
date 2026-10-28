#!/bin/bash
# TM (28 Oct): the trial-matched package on one fold, against G4c (C1 on the
# same code). Changes from G4c, all at once:
#   prior     theory family off; virtual-trial family with randomised
#             allocation (10 % of the processes); generic share 0.9: 72 % of
#             the processes of the trial kinds trial, trial_mlp, trial2,
#             trial2_mlp (effects -1 / 0 / +1, null share 0.3 to 0.7, 85 %
#             with TE 1 and RE 0, the rest TE 0.75 to 1 and RE 0 to 0.25) and
#             18 % linear or step without clipping, effect 0.3 to 1
#   target    every query at both arms, plus the squared error of the effect
#   head      outcome bins from -0.05 to 1.05
#   queries   256 per process
#   context   every batch's length drawn log-uniformly from 128 to 3,000
# Same pool, seed, iterations, architecture and longest context as G4c. The prior gate
# runs first and a prior that fails it does not train. GATE_ONLY=1 stops after
# the gate (CPU only). Run from the root of a clean checkout with the atlas
# at data/atlases; CACHE and OUT can be overridden, the defaults are siena's.
#
#   GATE_ONLY=1 bash scripts/audit_20261028/tm_launch.sh
#   bash scripts/audit_20261028/tm_launch.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
dirty=$(git status --porcelain --untracked-files=no | wc -l)
echo "code_commit=$(git rev-parse HEAD) cambios_sin_commit=$dirty"
[ "$dirty" -eq 0 ] || { echo "uncommitted changes: refusing to train" >&2; exit 1; }
export PYTHONPATH="$PWD/src"
# anything left over from an earlier launch must not reach this one
unset CACHE_ANCHOR_WEIGHT ITERS VARIANT BATCH AUGMENT CONTEXT_REPLACE STANDARDIZE D_MODEL N_LAYERS
command -v module >/dev/null 2>&1 && module load python3/3.11
source "${VENV:-$HOME/venvs/neuro}/bin/activate"
# SEED (default 0, G4c's) can be set for a second training of the same package
export SEED="${SEED:-0}"
export JOBNAME="${JOBNAME:-TM-f0-s$SEED}"
export CACHE="${CACHE:-$HOME/p2_bcb/pools/kch_s0/fold0.npz}"
export OUT="${OUT:-$HOME/p2_bcb/pfn_screen/TM/fold0}"
export FAMILY_WEIGHTS="0 1" ARCH=linear FAMILY=mixture
export GILES_HYPER="random_partitions=1 threshold=0.03:0.08 p_te=0.9:1 p_re=0:0.05 label_noise=0:0 bias=0:0 bias_weights=1:0:0"
export PRIOR_HYPER="gamma=0:0 generic_weight=0.9 generic_kinds=trial+trial_mlp+trial2+trial2_mlp+linear+step generic_null=0.3:0.7 generic_clean=0.85 generic_kind_weights=2:2:2:2:1:1 generic_te=0.75:1 generic_re=0:0.25 generic_scale=0.3:1 generic_noclip=1 generic_binary=1"
export LR_SCHEDULE=constant CONTEXT_POOL=susceptible CONTEXT_MAX=3000 OMP_NUM_THREADS=4
export BOTH_ARMS=1 CATE_WEIGHT=0.5 HEAD_LO=-0.05 HEAD_HI=1.05 N_QUERY=256 CURRICULUM=loguniform CONTEXT_MIN=128
[ -s "$CACHE" ] || { echo "no cache $CACHE" >&2; exit 1; }
[ -d data/atlases ] || { echo "no atlas at data/atlases" >&2; exit 1; }
mkdir -p "$OUT"
python scripts/prior_gate.py --batches 50 --contexts 128 512 3000 --out "$OUT/prior_gate.csv" \
    --expect prior.family=mixture prior.family_weights=0:1 prior.hyper.generic_weight=0.9 \
             prior.hyper_giles.bias=0:0 prior.hyper_giles.random_partitions=1 \
             pfn.both_arms=True pfn.head_lo=-0.05 pfn.head_hi=1.05 pfn.n_query=256 pfn.context_max=3000 \
             pfn.curriculum=loguniform pfn.context_min=128
[ "${GATE_ONLY:-0}" = "1" ] && { echo "GATE_ONLY: not training"; exit 0; }
bash scripts/train_pfn_myriad.qsub.sh
