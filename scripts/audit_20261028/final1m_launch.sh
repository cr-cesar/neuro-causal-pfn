#!/bin/bash
# The final model at 1M simulated trials (2 Nov): e13pool's launch (family
# weights 0 1, OBJECTIVE=new, the E13a-encoded KCH pool) with 250,000 steps
# of 4 trials, three seeds, plus a same-code, same-machine control at e13pool's
# budget (125,000 steps, seed 0), so 1M against 500k changes the budget only.
# All four train at once from this checkout; a rerun skips the checkpoints
# that exist. Before training: the pool's md5 and one md5 over the atlas files
# must match the expected digests (POOL_MD5, ATLAS_MD5, those of siena).
#
#   VENV=/scratch0/ccornejo/venvs/ncp POOL=... ATLAS=... O=... bash scripts/audit_20261028/final1m_launch.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:?set VENV to the training environment}"
POOL="${POOL:?set POOL to the E13a KCH pool}"
ATLAS="${ATLAS:-$HOME/export_bcb/atlases}"
O="${O:?set O to the output folder}"
POOL_MD5="${POOL_MD5:-341e229d88f4}"
ATLAS_MD5="${ATLAS_MD5:-fd1b531e7cf9}"
mkdir -p "$O/logs"
[ "$(git -C "$HERE" status --porcelain --untracked-files=no | wc -l)" -eq 0 ] || { echo "$HERE has local changes" >&2; exit 1; }
[ -e "$HERE/data/atlases" ] || { mkdir -p "$HERE/data" && ln -s "$ATLAS" "$HERE/data/atlases"; }
pool_md5=$(md5sum "$POOL" | cut -c1-12)
atlas_md5=$(cd -P "$ATLAS" && find functional_parcellation_2mm.nii.gz 2mm_parcellations -type f | LC_ALL=C sort \
    | xargs -d '\n' md5sum | md5sum | cut -c1-12)
echo "== $(date +%H:%M) code $(git -C "$HERE" rev-parse --short HEAD), pool md5 $pool_md5, atlas md5 $atlas_md5"
[ "$pool_md5" = "$POOL_MD5" ] || { echo "pool md5 $pool_md5, expected $POOL_MD5" >&2; exit 1; }
[ "$atlas_md5" = "$ATLAS_MD5" ] || { echo "atlas md5 $atlas_md5, expected $ATLAS_MD5" >&2; exit 1; }

run() {  # name, seed, steps
    [ -s "$O/$1/pfn.pt" ] && return 0
    (cd "$HERE" && env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VENV="$VENV" \
        OBJECTIVE=new FAMILY_W="0 1" ATLAS_SRC="$ATLAS" SEED="$2" RUN_ITERS="$3" \
        JOBNAME="final1m-$1" CACHE="$POOL" OUT="$O/$1" \
        bash scripts/audit_20261028/g4c_launch.sh) > "$O/logs/train_$1.log" 2>&1
}
run s0 0 250000 & run s1 1 250000 & run s2 2 250000 & run ctl500k_s0 0 125000 &
wait || true
for n in s0 s1 s2 ctl500k_s0; do
    if [ -s "$O/$n/pfn.pt" ]; then echo "   $n: checkpoint $(md5sum "$O/$n/pfn.pt" | cut -c1-12), $(grep -c -iE "error|traceback|nan" "$O/logs/train_$n.log" || true) error lines"
    else echo "   $n: no checkpoint (see $O/logs/train_$n.log)"; fi
done
echo "== $(date +%H:%M) done"
