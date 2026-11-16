#!/bin/bash
# Job D (16 Nov): TabPFN continued on the final prior. This is
# final1m_launch.sh's launch with another trainer: the same g4c_launch.sh call
# (OBJECTIVE=new FAMILY_W="0 1", the E13a KCH pool, a clean tree, the same md5
# checks of the pool and the atlas, the same prior gate) plus RUN_TRAINER=tabpfn,
# so the prior is built by the lines that built the final model's and
# scripts/finetune_tabpfn_prior.py --prior-from-env trains on it.
#
# One model per seed (SEEDS, default 0 1 2), each meant for all folds: the pool
# is one file, the KCH pool of the final model, not a pool per fold. Training
# reads the pool and the atlas and nothing else, and this script scores
# nothing: no fold of any cohort is opened here.
#
# Steps, learning rate, simulated trials per draw of the prior, context length
# and ensemble size are variables; the defaults are those of the route-A
# wrapper (tabpfn_ft_perfold.sh: 2,000 steps, context 1,500, 1 estimator) and
# of the script (learning rate 1e-5, 4 trials per draw). The update itself is
# one simulated trial per step; FT_BATCH is how many trials one draw of the
# prior holds (the final model's batch is 4). MODEL_PATH gives the starting
# checkpoint by path (default: the package's own); its md5 is recorded.
# The seeds run one after another (PARALLEL=1: at once, if the card holds them).
#
# Writes <O>/s<seed>/{tabpfn_ft.pth,history.json,provenance.json} and
# <O>/logs/train_s<seed>.log; a seed whose provenance.json says finished is
# skipped, so a rerun only does what is missing.
#
#   VENV=<environment with tabpfn> POOL=... ATLAS=... O=... [SEEDS="0 1 2"] [STEPS=2000] [LR=1e-5] \
#       [FT_BATCH=4] [N_CONTEXT=1500] [N_EST=1] [MODEL_PATH=...] [PARALLEL=1] \
#       bash scripts/audit_20261028/tabpfn_final_launch.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:?set VENV to an environment with tabpfn (the panel environment)}"
POOL="${POOL:?set POOL to the E13a KCH pool}"
ATLAS="${ATLAS:-$HOME/export_bcb/atlases}"
O="${O:?set O to the output folder}"
POOL_MD5="${POOL_MD5:-341e229d88f4}"
ATLAS_MD5="${ATLAS_MD5:-fd1b531e7cf9}"
SEEDS="${SEEDS:-0 1 2}"
STEPS="${STEPS:-2000}"
LR="${LR:-1e-5}"
FT_BATCH="${FT_BATCH:-4}"
N_CONTEXT="${N_CONTEXT:-1500}"
N_EST="${N_EST:-1}"
MODEL_PATH="${MODEL_PATH:-}"
PARALLEL="${PARALLEL:-0}"
for v in STEPS FT_BATCH N_CONTEXT N_EST; do
    [[ "${!v}" =~ ^[1-9][0-9]*$ ]] || { echo "$v must be a positive integer" >&2; exit 1; }
done
[[ "$LR" =~ ^[0-9]+(\.[0-9]+)?([eE]-?[0-9]+)?$ ]] || { echo "LR must be a number, e.g. 1e-5" >&2; exit 1; }
[[ "$SEEDS" =~ ^[0-9]+(\ [0-9]+)*$ ]] || { echo "SEEDS must be integers separated by spaces" >&2; exit 1; }
case "$PARALLEL" in 0|1) ;; *) echo "PARALLEL must be 0 or 1" >&2; exit 1 ;; esac
if [ -n "$MODEL_PATH" ]; then
    [[ "$MODEL_PATH" != *" "* ]] || { echo "MODEL_PATH must not contain spaces" >&2; exit 1; }
    [ -s "$MODEL_PATH" ] || { echo "no checkpoint $MODEL_PATH" >&2; exit 1; }
fi
mkdir -p "$O/logs"
[ "$(git -C "$HERE" status --porcelain --untracked-files=no | wc -l)" -eq 0 ] || { echo "$HERE has local changes" >&2; exit 1; }
[ -e "$HERE/data/atlases" ] || { mkdir -p "$HERE/data" && ln -s "$ATLAS" "$HERE/data/atlases"; }
# the digests of final1m_launch.sh, computed the same way
pool_md5=$(md5sum "$POOL" | cut -c1-12)
atlas_md5=$(cd -P "$ATLAS" && find functional_parcellation_2mm.nii.gz 2mm_parcellations -type f | LC_ALL=C sort \
    | xargs -d '\n' md5sum | md5sum | cut -c1-12)
echo "== $(date +%H:%M) code $(git -C "$HERE" rev-parse --short HEAD), pool md5 $pool_md5, atlas md5 $atlas_md5"
[ "$pool_md5" = "$POOL_MD5" ] || { echo "pool md5 $pool_md5, expected $POOL_MD5" >&2; exit 1; }
[ "$atlas_md5" = "$ATLAS_MD5" ] || { echo "atlas md5 $atlas_md5, expected $ATLAS_MD5" >&2; exit 1; }
echo "   steps $STEPS, lr $LR, trials per draw $FT_BATCH, context $N_CONTEXT, estimators $N_EST," \
     "starting weights ${MODEL_PATH:-the package default}, seeds $SEEDS"

finished() { [ -s "$O/s$1/tabpfn_ft.pth" ] && grep -q '"finished": true' "$O/s$1/provenance.json" 2>/dev/null; }

run() {  # seed
    finished "$1" && return 0
    (cd "$HERE" && env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VENV="$VENV" \
        OBJECTIVE=new FAMILY_W="0 1" ATLAS_SRC="$ATLAS" SEED="$1" \
        JOBNAME="tabpfn-final-s$1" CACHE="$POOL" OUT="$O/s$1" RUN_TRAINER=tabpfn \
        TABPFN_FT_ARGS="--steps $STEPS --lr $LR --processes-per-draw $FT_BATCH --n-context $N_CONTEXT --n-estimators $N_EST --pool-md5 $POOL_MD5 --atlas-md5 $atlas_md5${MODEL_PATH:+ --model-path $MODEL_PATH}" \
        bash scripts/audit_20261028/g4c_launch.sh) > "$O/logs/train_s$1.log" 2>&1
}
for s in $SEEDS; do
    if [ "$PARALLEL" = 1 ]; then run "$s" & else run "$s" || true; fi
done
wait || true
for s in $SEEDS; do
    if finished "$s"; then echo "   s$s: checkpoint $(md5sum "$O/s$s/tabpfn_ft.pth" | cut -c1-12), $(grep -c -iE "error|traceback|nan" "$O/logs/train_s$s.log" || true) error lines"
    else echo "   s$s: no finished checkpoint (see $O/logs/train_s$s.log)"; fi
done
echo "== $(date +%H:%M) done"
