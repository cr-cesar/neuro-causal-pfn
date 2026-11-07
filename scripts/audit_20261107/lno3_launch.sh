#!/bin/bash
# Wave 0, item 2 (Brief 8 audit, 7 Nov): leave-networks-out at the final
# model's budget with three PFN seeds. The final model's launch
# (final1m_launch.sh: family weights 0 1, OBJECTIVE=new, the E13a-encoded KCH
# pool, 250,000 steps of 4 trials = 1M) with one group of four networks left
# out of the virtual-trial family's random partitions. atom_table drops every
# atom of a left-out network from every channel (the whole network and both
# halves of both splits), so no virtual network is built from them.
#   g1..g4  networks 1-4, 5-8, 9-12, 13-16 left out; seeds 0, 1, 2
#   g0      optional same-code, same-machine control (CONTROL=1); the primary
#           contrast (held vs seen, lno3_score.sh) does not need it, because
#           it compares the same four PFNs with each other
# Up to MAXPAR trainings at once (default 3: siena runs three 4.8M seeds at
# once in about 5 h per million trials). A rerun skips existing checkpoints.
#
#   VENV=... POOL=... ATLAS=... O=... bash scripts/audit_20261107/lno3_launch.sh
#   SEEDS="0" CONTROL=1 ... bash scripts/audit_20261107/lno3_launch.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:?set VENV to the training environment}"
POOL="${POOL:?set POOL to the E13a KCH pool}"
ATLAS="${ATLAS:-$HOME/export_bcb/atlases}"
O="${O:?set O to the output folder}"
POOL_MD5="${POOL_MD5:-341e229d88f4}"
ATLAS_MD5="${ATLAS_MD5:-fd1b531e7cf9}"
STEPS="${STEPS:-250000}"
SEEDS="${SEEDS:-0 1 2}"
MAXPAR="${MAXPAR:-3}"
CONTROL="${CONTROL:-0}"
EXCL=("" "1+2+3+4" "5+6+7+8" "9+10+11+12" "13+14+15+16")
mkdir -p "$O/logs"
[ "$(git -C "$HERE" status --porcelain --untracked-files=no | wc -l)" -eq 0 ] || { echo "$HERE has local changes" >&2; exit 1; }
[ -e "$HERE/data/atlases" ] || { mkdir -p "$HERE/data" && ln -s "$ATLAS" "$HERE/data/atlases"; }
pool_md5=$(md5sum "$POOL" | cut -c1-12)
atlas_md5=$(cd -P "$ATLAS" && find functional_parcellation_2mm.nii.gz 2mm_parcellations -type f | LC_ALL=C sort \
    | xargs -d '\n' md5sum | md5sum | cut -c1-12)
echo "== $(date +%H:%M) code $(git -C "$HERE" rev-parse --short HEAD), pool md5 $pool_md5, atlas md5 $atlas_md5, steps $STEPS, seeds $SEEDS"
[ "$pool_md5" = "$POOL_MD5" ] || { echo "pool md5 $pool_md5, expected $POOL_MD5" >&2; exit 1; }
[ "$atlas_md5" = "$ATLAS_MD5" ] || { echo "atlas md5 $atlas_md5, expected $ATLAS_MD5" >&2; exit 1; }

groups="1 2 3 4"; [ "$CONTROL" = 1 ] && groups="0 $groups"
running=0
for s in $SEEDS; do
    for g in $groups; do
        n="g${g}_s$s"
        [ -s "$O/$n/pfn.pt" ] && continue
        (cd "$HERE" && env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VENV="$VENV" \
            OBJECTIVE=new FAMILY_W="0 1" ATLAS_SRC="$ATLAS" SEED="$s" RUN_ITERS="$STEPS" \
            EXCLUDE_NETWORKS="${EXCL[$g]}" JOBNAME="lno3-$n" CACHE="$POOL" OUT="$O/$n" \
            bash scripts/audit_20261028/g4c_launch.sh) > "$O/logs/train_$n.log" 2>&1 &
        running=$((running + 1))
        if [ "$running" -ge "$MAXPAR" ]; then wait -n || true; running=$((running - 1)); fi
    done
done
wait || true
missing=0
for s in $SEEDS; do for g in $groups; do
    n="g${g}_s$s"
    if [ -s "$O/$n/pfn.pt" ]; then
        echo "   $n: checkpoint $(md5sum "$O/$n/pfn.pt" | cut -c1-12), $(grep -c -iE "error|traceback|nan" "$O/logs/train_$n.log" || true) error lines," \
             "$(grep -m1 -oE "leave-networks-out: [0-9]+ network\(s\) left out, [0-9]+ atoms remain" "$O/logs/train_$n.log" || echo 'no atom log line')"
    else echo "   $n: no checkpoint (see $O/logs/train_$n.log)"; missing=1; fi
done; done
echo "== $(date +%H:%M) done"
[ "$missing" = 0 ]
