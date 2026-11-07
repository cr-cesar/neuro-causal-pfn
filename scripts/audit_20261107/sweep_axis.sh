#!/bin/bash
# Stage 1 sweep (Brief 8 audit, 7 Nov): one level of one axis against R0, the
# final model's launch (final1m_launch.sh: family weights 0 1, OBJECTIVE=new,
# the E13a-encoded KCH pool, 250,000 steps of 4 trials = 1M), three PFN seeds,
# encoder seed 0. The level is given as g4c_launch.sh variables after the name;
# everything else is R0. R0's own seeds are final1M s0-s2 (commit 69250c3,
# beachcomber); train one same-machine R0 seed (NAME=r0, no variables) on every
# new machine to measure machine drift.
#
#   VENV=... POOL=... ATLAS=... O=... bash scripts/audit_20261107/sweep_axis.sh gw025 GENERIC_W=0.25
#   ... sweep_axis.sh wide RUN_GILES_RANGES=wide
#   ... sweep_axis.sh cw0 CATE_W=0
#   ... sweep_axis.sh cos RUN_LR_SCHEDULE=cosine
#   ... sweep_axis.sh dist DISTINCT=1
#   ... sweep_axis.sh noagn RUN_BIAS_WEIGHTS=0.667:0.333:0
#   SEEDS="0" ... sweep_axis.sh r0
# Then score on validation (CPU), e.g. as final1m_val.sh does:
#   CUDA_VISIBLE_DEVICES="" REPS=$W/e13/E13a/E13a/seed0/folds \
#       bash scripts/audit_20261028/score_val_clean.sh sweep_<name>_s<seed> --estimators $O/<name>/s<seed>/pfn.pt --only-estimators
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 NAME [VAR=value ...]" >&2; exit 1; }
NAME="$1"; shift
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:?set VENV to the training environment}"
POOL="${POOL:?set POOL to the E13a KCH pool}"
ATLAS="${ATLAS:-$HOME/export_bcb/atlases}"
O="${O:?set O to the output folder}"
POOL_MD5="${POOL_MD5:-341e229d88f4}"
ATLAS_MD5="${ATLAS_MD5:-fd1b531e7cf9}"
STEPS="${STEPS:-250000}"
SEEDS="${SEEDS:-0 1 2}"
for kv in "$@"; do [[ "$kv" =~ ^[A-Z_]+=.*$ ]] || { echo "bad variable $kv" >&2; exit 1; }; done
mkdir -p "$O/$NAME/logs"
[ "$(git -C "$HERE" status --porcelain --untracked-files=no | wc -l)" -eq 0 ] || { echo "$HERE has local changes" >&2; exit 1; }
[ -e "$HERE/data/atlases" ] || { mkdir -p "$HERE/data" && ln -s "$ATLAS" "$HERE/data/atlases"; }
pool_md5=$(md5sum "$POOL" | cut -c1-12)
atlas_md5=$(cd -P "$ATLAS" && find functional_parcellation_2mm.nii.gz 2mm_parcellations -type f | LC_ALL=C sort \
    | xargs -d '\n' md5sum | md5sum | cut -c1-12)
echo "== $(date +%H:%M) $NAME: code $(git -C "$HERE" rev-parse --short HEAD), pool md5 $pool_md5, atlas md5 $atlas_md5, level: ${*:-R0}"
[ "$pool_md5" = "$POOL_MD5" ] || { echo "pool md5 $pool_md5, expected $POOL_MD5" >&2; exit 1; }
[ "$atlas_md5" = "$ATLAS_MD5" ] || { echo "atlas md5 $atlas_md5, expected $ATLAS_MD5" >&2; exit 1; }

for s in $SEEDS; do
    [ -s "$O/$NAME/s$s/pfn.pt" ] && continue
    (cd "$HERE" && env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VENV="$VENV" \
        OBJECTIVE=new FAMILY_W="0 1" ATLAS_SRC="$ATLAS" SEED="$s" RUN_ITERS="$STEPS" \
        JOBNAME="sweep-$NAME-s$s" CACHE="$POOL" OUT="$O/$NAME/s$s" "$@" \
        bash scripts/audit_20261028/g4c_launch.sh) > "$O/$NAME/logs/train_s$s.log" 2>&1 &
done
wait || true
for s in $SEEDS; do
    if [ -s "$O/$NAME/s$s/pfn.pt" ]; then
        echo "   s$s: checkpoint $(md5sum "$O/$NAME/s$s/pfn.pt" | cut -c1-12), $(grep -c -iE "error|traceback|nan" "$O/$NAME/logs/train_s$s.log" || true) error lines"
    else echo "   s$s: no checkpoint (see $O/$NAME/logs/train_s$s.log)"; fi
done
echo "== $(date +%H:%M) done"
