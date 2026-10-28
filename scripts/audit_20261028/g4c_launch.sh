#!/bin/bash
# G4c (28 Oct): C1's launch on the current code, the control of G5 and TM.
# This is g5_launch.sh without G5's two allocation changes (bias=0:0
# bias_weights=1:0:0 in GILES_HYPER, gamma=0:0 in PRIOR_HYPER), i.e. the line
# G4c was trained with. SEED (default 0, G4c's own) gives a second training of
# the same configuration: the variation between two trainings on one code,
# which C1 against G4c could not measure (they differ in the prior code too).
# Run from the root of a clean checkout with the atlas at data/atlases; CACHE
# and OUT can be overridden, the defaults are siena's.
#
#   SEED=1 OUT=... bash scripts/audit_20261028/g4c_launch.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
dirty=$(git status --porcelain --untracked-files=no | wc -l)
echo "code_commit=$(git rev-parse HEAD) cambios_sin_commit=$dirty"
[ "$dirty" -eq 0 ] || { echo "uncommitted changes: refusing to train" >&2; exit 1; }
export PYTHONPATH="$PWD/src"
# anything left over from an earlier launch must not reach this one
unset CACHE_ANCHOR_WEIGHT ITERS VARIANT BATCH CONTEXT_MIN AUGMENT CONTEXT_REPLACE STANDARDIZE D_MODEL N_LAYERS \
    BOTH_ARMS CATE_WEIGHT HEAD_LO HEAD_HI N_QUERY CURRICULUM
command -v module >/dev/null 2>&1 && module load python3/3.11
source "${VENV:-$HOME/venvs/neuro}/bin/activate"
export SEED="${SEED:-0}"
export JOBNAME="${JOBNAME:-G4c-f0-s$SEED}"
export CACHE="${CACHE:-$HOME/p2_bcb/pools/kch_s0/fold0.npz}"
export OUT="${OUT:-$HOME/p2_bcb/pfn_screen/G4c_s$SEED/fold0}"
export FAMILY_WEIGHTS="0.5 0.5" ARCH=linear FAMILY=mixture
export GILES_HYPER="random_partitions=1 threshold=0.03:0.08 p_te=0.9:1 p_re=0:0.05 label_noise=0:0"
export PRIOR_HYPER="two_arms=1 effect_scale=0.2:1 d_source=network s_source=genetics_1 s_b_source=genetics_2 effect_headroom=1 generic_weight=0.5 binary_outcome=1 generic_binary=1"
export LR_SCHEDULE=constant CONTEXT_POOL=susceptible CONTEXT_MAX=3000 OMP_NUM_THREADS=4
[ -s "$CACHE" ] || { echo "no cache $CACHE" >&2; exit 1; }
[ -d data/atlases ] || { echo "no atlas at data/atlases" >&2; exit 1; }
mkdir -p "$OUT"
# the launch must be G4c's: the shared values of G5 / TM and no 28 Oct switch
python scripts/prior_gate.py --batches 5 --contexts 512 --require share_zero=0:1 share_unit=0:1 share_small=0:1 \
    pos_share=0:1 ctx_binary=0:1 ctx_treated=0:1 \
    --expect prior.family=mixture prior.family_weights=0.5:0.5 prior.hyper.generic_weight=0.5 \
             prior.hyper_giles.random_partitions=1 pfn.context_max=3000 pfn.context_min=256 \
             pfn.iters=20000 pfn.arch=linear pfn.n_query=64
bash scripts/train_pfn_myriad.qsub.sh
