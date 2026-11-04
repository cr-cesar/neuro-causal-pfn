#!/bin/bash
# Depth (4 Nov): the final model's launch (final1m_launch.sh: family weights
# 0 1, OBJECTIVE=new, the E13a-encoded KCH pool, 250,000 steps of 4 trials =
# 1M) with 12 layers of width 256 instead of 6 (9.6M parameters against 4.8M),
# three seeds. The context ramp is fixed at the final model's 125,000 steps and
# there is a checkpoint every 62,500 steps (250k trials), as scale15m_launch.sh,
# so only the depth differs from the 1M point of that run, on the same machine
# and code. A rerun skips the seeds whose final checkpoint exists. Before
# training: the pool's md5 and one md5 over the atlas files must match.
#
#   VENV=... POOL=... ATLAS=... O=... bash scripts/audit_20261028/deep10m_launch.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:?set VENV to the training environment}"
POOL="${POOL:?set POOL to the E13a KCH pool}"
ATLAS="${ATLAS:-$HOME/export_bcb/atlases}"
O="${O:?set O to the output folder}"
POOL_MD5="${POOL_MD5:-341e229d88f4}"
ATLAS_MD5="${ATLAS_MD5:-fd1b531e7cf9}"
STEPS="${STEPS:-250000}"
LAYERS="${LAYERS:-12}"
mkdir -p "$O/logs"
[ "$(git -C "$HERE" status --porcelain --untracked-files=no | wc -l)" -eq 0 ] || { echo "$HERE has local changes" >&2; exit 1; }
[ -e "$HERE/data/atlases" ] || { mkdir -p "$HERE/data" && ln -s "$ATLAS" "$HERE/data/atlases"; }
pool_md5=$(md5sum "$POOL" | cut -c1-12)
atlas_md5=$(cd -P "$ATLAS" && find functional_parcellation_2mm.nii.gz 2mm_parcellations -type f | LC_ALL=C sort \
    | xargs -d '\n' md5sum | md5sum | cut -c1-12)
echo "== $(date +%H:%M) code $(git -C "$HERE" rev-parse --short HEAD), pool md5 $pool_md5, atlas md5 $atlas_md5, steps $STEPS, layers $LAYERS"
[ "$pool_md5" = "$POOL_MD5" ] || { echo "pool md5 $pool_md5, expected $POOL_MD5" >&2; exit 1; }
[ "$atlas_md5" = "$ATLAS_MD5" ] || { echo "atlas md5 $atlas_md5, expected $ATLAS_MD5" >&2; exit 1; }

run() {  # name, seed
    [ -s "$O/$1/pfn.pt" ] && return 0
    (cd "$HERE" && env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VENV="$VENV" \
        OBJECTIVE=new FAMILY_W="0 1" ATLAS_SRC="$ATLAS" SEED="$2" RUN_ITERS="$STEPS" \
        RUN_RAMP_ITERS=125000 RUN_CKPT_EVERY=62500 RUN_N_LAYERS="$LAYERS" \
        JOBNAME="deep10m-$1" CACHE="$POOL" OUT="$O/$1" \
        bash scripts/audit_20261028/g4c_launch.sh) > "$O/logs/train_$1.log" 2>&1
}
run s0 0 & run s1 1 & run s2 2 &
wait || true
for n in s0 s1 s2; do
    if [ -s "$O/$n/pfn.pt" ]; then
        echo "   $n: $(ls "$O/$n"/pfn_it*.pt 2>/dev/null | wc -l) intermediate checkpoints," \
             "final $(md5sum "$O/$n/pfn.pt" | cut -c1-12), $(grep -c -iE "error|traceback|nan" "$O/logs/train_$n.log" || true) error lines"
    else echo "   $n: no checkpoint (see $O/logs/train_$n.log)"; fi
done
echo "== $(date +%H:%M) done"
