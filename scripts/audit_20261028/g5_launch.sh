#!/bin/bash
# G5 (28 Oct): C1 with randomised treatment allocation in the prior, as in the
# scored trial (BIAS = 0). Everything else is G4c's launch (C1 on this code):
# same pool, seed, iterations, architecture, context and prior, except
#   GILES_HYPER  + bias=0:0 bias_weights=1:0:0   (virtual trials: P(T=1) = 0.5)
#   PRIOR_HYPER  + gamma=0:0                      (theory family: no confounding)
# Run from the repository root of a clean checkout, with the atlas linked at
# data/atlases. CACHE and OUT can be overridden; the defaults are siena's.
#
#   bash scripts/audit_20261028/g5_launch.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
dirty=$(git status --porcelain --untracked-files=no | wc -l)
echo "code_commit=$(git rev-parse HEAD) cambios_sin_commit=$dirty"
[ "$dirty" -eq 0 ] || { echo "uncommitted changes: refusing to train" >&2; exit 1; }
export PYTHONPATH="$PWD/src"
# the training script reads these from the environment: anything left over
# from an earlier run (e.g. CONTEXT_MIN=64 from G4) must not reach G5
unset CACHE_ANCHOR_WEIGHT ITERS VARIANT BATCH CONTEXT_MIN AUGMENT CONTEXT_REPLACE STANDARDIZE D_MODEL N_LAYERS
# the same Python environment as the training script, for the check below
command -v module >/dev/null 2>&1 && module load python3/3.11
source "${VENV:-$HOME/venvs/neuro}/bin/activate"
export JOBNAME=G5-f0
export CACHE="${CACHE:-$HOME/p2_bcb/pools/kch_s0/fold0.npz}"
export OUT="${OUT:-$HOME/p2_bcb/pfn_screen/G5/fold0}"
export FAMILY_WEIGHTS="0.5 0.5" SEED=0 ARCH=linear FAMILY=mixture
export GILES_HYPER="random_partitions=1 threshold=0.03:0.08 p_te=0.9:1 p_re=0:0.05 label_noise=0:0 bias=0:0 bias_weights=1:0:0"
export PRIOR_HYPER="two_arms=1 effect_scale=0.2:1 d_source=network s_source=genetics_1 s_b_source=genetics_2 effect_headroom=1 generic_weight=0.5 binary_outcome=1 generic_binary=1 gamma=0:0"
export LR_SCHEDULE=constant CONTEXT_POOL=susceptible CONTEXT_MAX=3000 OMP_NUM_THREADS=4
[ -s "$CACHE" ] || { echo "no cache $CACHE" >&2; exit 1; }
[ -d data/atlases ] || { echo "no atlas at data/atlases" >&2; exit 1; }
python scripts/audit_20261028/g5_check.py --sample 200
mkdir -p "$OUT"
bash scripts/train_pfn_myriad.qsub.sh
