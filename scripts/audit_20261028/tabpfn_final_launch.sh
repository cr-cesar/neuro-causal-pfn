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
# Steps, learning rate, context length and ensemble size are variables; the
# defaults are those of the route-A wrapper (tabpfn_ft_perfold.sh: 2,000 steps,
# context 1,500, 1 estimator) and of the script (learning rate 1e-5).
# TRIALS_PER_DRAW (default 4, the script's) is how many simulated trials one
# draw of the prior holds. It is not a batch: each trial is one update, so
# STEPS updates read STEPS trials whatever this value is (the final model's
# batch of 4 trials per update has no counterpart here).
#
# Starting weights: TABPFN_VERSION (default v3.5) is exported to the package as
# TABPFN_MODEL_VERSION and checked by the script, which takes the package's
# file for that version by path and records its md5; MODEL_PATH gives another
# file by path (its version is then not checked), MODEL_MD5 its expected md5.
#
# The seeds run one after another and the launch stops at the first that fails
# (PARALLEL=1: all at once, if the card holds them). Any seed without a
# finished model makes the launch exit with an error.
#
# Writes <O>/s<seed>/{tabpfn_ft.pth,history.json,provenance.json} and
# <O>/logs/train_s<seed>.log. A seed already finished with this very
# configuration (commit, pool, atlas, starting weights, options, prior) is left
# as it is; one finished with another configuration stops the launch: the
# script compares provenance.json with what is asked.
#
#   VENV=<environment with tabpfn> POOL=... ATLAS=... O=... [SEEDS="0 1 2"] [STEPS=2000] [LR=1e-5] \
#       [TRIALS_PER_DRAW=4] [N_CONTEXT=1500] [N_EST=1] [TABPFN_VERSION=v3.5] [MODEL_PATH=... [MODEL_MD5=...]] \
#       [PARALLEL=1] bash scripts/audit_20261028/tabpfn_final_launch.sh
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
[ -z "${FT_BATCH:-}" ] || { echo "FT_BATCH is now TRIALS_PER_DRAW: trials per draw of the prior, not per update" >&2; exit 1; }
TRIALS_PER_DRAW="${TRIALS_PER_DRAW:-4}"
N_CONTEXT="${N_CONTEXT:-1500}"
N_EST="${N_EST:-1}"
TABPFN_VERSION="${TABPFN_VERSION:-v3.5}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_MD5="${MODEL_MD5:-}"
PARALLEL="${PARALLEL:-0}"
for v in STEPS TRIALS_PER_DRAW N_CONTEXT N_EST; do
    [[ "${!v}" =~ ^[1-9][0-9]*$ ]] || { echo "$v must be a positive integer" >&2; exit 1; }
done
lr_mantissa="${LR%%[eE]*}"
[[ "$LR" =~ ^[0-9]+(\.[0-9]+)?([eE]-?[0-9]+)?$ && "$lr_mantissa" =~ [1-9] ]] || { echo "LR must be a positive number, e.g. 1e-5" >&2; exit 1; }
[[ "$SEEDS" =~ ^[0-9]+(\ [0-9]+)*$ ]] || { echo "SEEDS must be integers separated by spaces" >&2; exit 1; }
[[ "$TABPFN_VERSION" =~ ^v[0-9]+(\.[0-9]+)?$ ]] || { echo "TABPFN_VERSION must look like v3.5" >&2; exit 1; }
case "$PARALLEL" in 0|1) ;; *) echo "PARALLEL must be 0 or 1" >&2; exit 1 ;; esac
if [ -n "$MODEL_PATH" ]; then
    [[ "$MODEL_PATH" != *" "* ]] || { echo "MODEL_PATH must not contain spaces" >&2; exit 1; }
    [ -s "$MODEL_PATH" ] || { echo "no checkpoint $MODEL_PATH" >&2; exit 1; }
fi
[ -z "$MODEL_MD5" ] || [ -n "$MODEL_PATH" ] || { echo "MODEL_MD5 needs MODEL_PATH" >&2; exit 1; }
[ -z "$MODEL_MD5" ] || [[ "$MODEL_MD5" =~ ^[0-9a-f]{8,32}$ ]] || { echo "MODEL_MD5 must be 8 to 32 hex characters" >&2; exit 1; }
[ "$(git -C "$HERE" status --porcelain --untracked-files=no | wc -l)" -eq 0 ] || { echo "$HERE has local changes" >&2; exit 1; }
[ -e "$HERE/data/atlases" ] || { mkdir -p "$HERE/data" && ln -s "$ATLAS" "$HERE/data/atlases"; }
# the digests of final1m_launch.sh, computed the same way
pool_md5=$(md5sum "$POOL" | cut -c1-12)
atlas_md5=$(cd -P "$ATLAS" && find functional_parcellation_2mm.nii.gz 2mm_parcellations -type f | LC_ALL=C sort \
    | xargs -d '\n' md5sum | md5sum | cut -c1-12)
echo "== $(date +%H:%M) code $(git -C "$HERE" rev-parse --short HEAD), pool md5 $pool_md5, atlas md5 $atlas_md5"
[ "$pool_md5" = "$POOL_MD5" ] || { echo "pool md5 $pool_md5, expected $POOL_MD5" >&2; exit 1; }
[ "$atlas_md5" = "$ATLAS_MD5" ] || { echo "atlas md5 $atlas_md5, expected $ATLAS_MD5" >&2; exit 1; }
echo "   steps $STEPS, lr $LR, trials per draw $TRIALS_PER_DRAW, context $N_CONTEXT, estimators $N_EST," \
     "TabPFN $TABPFN_VERSION from ${MODEL_PATH:-the package file}, seeds $SEEDS"
mkdir -p "$O/logs"

finished() { [ -s "$O/s$1/tabpfn_ft.pth" ] && grep -q '"finished": true' "$O/s$1/provenance.json" 2>/dev/null; }

run() {  # seed; g4c_launch.sh checks the pool and the atlas again and gives their digests to the script
    echo "== $(date '+%F %H:%M') launch, code $(git -C "$HERE" rev-parse --short HEAD)" >> "$O/logs/train_s$1.log"
    (cd "$HERE" && env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VENV="$VENV" \
        OBJECTIVE=new FAMILY_W="0 1" ATLAS_SRC="$ATLAS" SEED="$1" \
        JOBNAME="tabpfn-final-s$1" CACHE="$POOL" OUT="$O/s$1" RUN_TRAINER=tabpfn \
        POOL_MD5="$POOL_MD5" ATLAS_MD5="$ATLAS_MD5" TABPFN_MODEL_VERSION="$TABPFN_VERSION" \
        TABPFN_FT_ARGS="--steps $STEPS --lr $LR --processes-per-draw $TRIALS_PER_DRAW --n-context $N_CONTEXT --n-estimators $N_EST --model-version $TABPFN_VERSION${MODEL_PATH:+ --model-path $MODEL_PATH}${MODEL_MD5:+ --model-md5 $MODEL_MD5}" \
        bash scripts/audit_20261028/g4c_launch.sh) >> "$O/logs/train_s$1.log" 2>&1
}
failed=" "
if [ "$PARALLEL" = 1 ]; then
    pids=()
    for s in $SEEDS; do run "$s" & pids+=("$!:$s"); done
    for p in "${pids[@]}"; do wait "${p%%:*}" || failed="$failed${p##*:} "; done
else
    for s in $SEEDS; do
        run "$s" || { failed="$failed$s "; echo "   s$s failed: the seeds after it are not started" >&2; break; }
    done
fi
status=0
for s in $SEEDS; do
    if [[ "$failed" == *" $s "* ]]; then
        status=1; echo "   s$s: FAILED: $(tail -n 1 "$O/logs/train_s$s.log")"
    elif finished "$s"; then
        echo "   s$s: checkpoint $(md5sum "$O/s$s/tabpfn_ft.pth" | cut -c1-12)"
    else
        status=1; echo "   s$s: no finished checkpoint (see $O/logs/train_s$s.log)"
    fi
done
echo "== $(date +%H:%M) $([ "$status" = 0 ] && echo done || echo "FAILED: not every seed has a finished model")"
exit "$status"
