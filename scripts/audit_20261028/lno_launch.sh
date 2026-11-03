#!/bin/bash
# Leave-networks-out (2 Nov) on siena: five trainings of e13pool's launch
# (fw01's recipe on the E13a pool: family weights 0 1, OBJECTIVE=new, 500k
# trials, seed 0), all from this checkout, so they share code and machine:
#   g0  the same-code control, no network left out;
#   g1..g4  networks 1-4, 5-8, 9-12, 13-16 left out of the virtual-trial
#       family's random partitions (EXCLUDE_NETWORKS).
# Then lno_score.sh scores them (CPU) and runs the paired tests. Up to MAXPAR
# trainings run at once (default 5); a training that fails leaves no
# checkpoint, and a rerun skips the ones that exist.
#
#   bash scripts/audit_20261028/lno_launch.sh
#   MAXPAR=3 bash scripts/audit_20261028/lno_launch.sh
set -euo pipefail
PY="${PY:-$HOME/env/bin/python}"
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
W="${W:-$HOME/p2_bcb}"
POOL="${POOL:-$W/pools/kch_e13a_s0/fold0.npz}"
ATLAS="${ATLAS:-$HOME/export_bcb/atlases}"
O="${O:-$W/pfn_lno}"
MAXPAR="${MAXPAR:-5}"
EXCL=("" "1+2+3+4" "5+6+7+8" "9+10+11+12" "13+14+15+16")
mkdir -p "$O/logs"
[ -s "$POOL" ] || { echo "no pool $POOL" >&2; exit 1; }
[ "$(git -C "$HERE" status --porcelain --untracked-files=no | wc -l)" -eq 0 ] || { echo "$HERE has local changes" >&2; exit 1; }
[ -e "$HERE/data/atlases" ] || { mkdir -p "$HERE/data" && ln -s "$ATLAS" "$HERE/data/atlases"; }
echo "== $(date +%H:%M) code $(git -C "$HERE" rev-parse --short HEAD), pool md5 $(md5sum "$POOL" | cut -c1-12)"

running=0
for g in 0 1 2 3 4; do
    [ -s "$O/g$g/pfn.pt" ] && continue
    (cd "$HERE" && env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VENV="$(dirname "$(dirname "$PY")")" \
        OBJECTIVE=new FAMILY_W="0 1" ATLAS_SRC="$ATLAS" SEED=0 RUN_ITERS=125000 EXCLUDE_NETWORKS="${EXCL[$g]}" \
        JOBNAME="lno-g$g-s0" CACHE="$POOL" OUT="$O/g$g" \
        bash scripts/audit_20261028/g4c_launch.sh) > "$O/logs/train_g$g.log" 2>&1 &
    running=$((running + 1))
    if [ "$running" -ge "$MAXPAR" ]; then wait -n || true; running=$((running - 1)); fi
done
wait || true
missing=0
for g in 0 1 2 3 4; do
    if [ -s "$O/g$g/pfn.pt" ]; then echo "   g$g: $(grep -c -iE "error|traceback|nan" "$O/logs/train_g$g.log" || true) error lines, checkpoint ok"
    else echo "   g$g: no checkpoint (see $O/logs/train_g$g.log)"; missing=1; fi
done
[ "$missing" = 0 ] || exit 1
CK_DIR="$O" bash "$HERE/scripts/audit_20261028/lno_score.sh"
