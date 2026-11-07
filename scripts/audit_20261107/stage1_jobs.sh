#!/bin/bash
# Brief 9 task 6: the ready stage 1 jobs (27 runs), three seeds each, one job
# (three seeds at once) after another, then validation scoring (CPU). Every job
# is sweep_axis.sh NAME VARIABLES; R0 (F1) is the dissertation's final1M.
# 2^(4-1) factorial, generator D = ABC: A generic share 0.5 / 0.75, B virtual-
# trial ranges clean / wide, C effect term CATE_W 0.5 / 0, D learning rate
# constant / cosine. Plus distinct contexts and generic share 0.25.
#
# GATE (Brief 9, rules): the protocol must be committed before the first sweep
# run. PROTOCOL names a file tracked in this repository, with no local changes;
# its commit and md5 are written to the log. The script refuses otherwise.
#
#   PROTOCOL=docs/protocol_stage1.md VENV=... POOL=... O=$W/pfn_stage1 \
#       bash scripts/audit_20261107/stage1_jobs.sh            # train + score
#   JOBS="F2 F3" ... stage1_jobs.sh                            # a subset
#   SCORE_ONLY=1 ... stage1_jobs.sh                            # scoring only
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
W="${W:-$HOME/p2_bcb}"
O="${O:?set O to the output folder}"
REPS="${REPS:-$W/e13/E13a/E13a/seed0/folds}"
SEEDS="${SEEDS:-0 1 2}"
PROTOCOL="${PROTOCOL:?set PROTOCOL to the committed protocol file (Brief 9: commit it before the first sweep run)}"
git -C "$HERE" ls-files --error-unmatch "$PROTOCOL" >/dev/null 2>&1 || { echo "$PROTOCOL is not tracked in $HERE" >&2; exit 1; }
[ -z "$(git -C "$HERE" status --porcelain -- "$PROTOCOL")" ] || { echo "$PROTOCOL has local changes" >&2; exit 1; }
echo "== protocol $PROTOCOL: commit $(git -C "$HERE" log -1 --format=%h -- "$PROTOCOL"), md5 $(md5sum "$HERE/$PROTOCOL" | cut -c1-12)"

declare -A V=(
  [F2]="GENERIC_W=0.75 RUN_LR_SCHEDULE=cosine"
  [F3]="RUN_GILES_RANGES=wide RUN_LR_SCHEDULE=cosine"
  [F4]="GENERIC_W=0.75 RUN_GILES_RANGES=wide"
  [F5]="CATE_W=0 RUN_LR_SCHEDULE=cosine"
  [F6]="GENERIC_W=0.75 CATE_W=0"
  [F7]="RUN_GILES_RANGES=wide CATE_W=0"
  [F8]="GENERIC_W=0.75 RUN_GILES_RANGES=wide CATE_W=0 RUN_LR_SCHEDULE=cosine"
  [dist]="DISTINCT=1"
  [gw025]="GENERIC_W=0.25"
)
JOBS="${JOBS:-F2 F3 F4 F5 F6 F7 F8 dist gw025}"
for j in $JOBS; do [ -n "${V[$j]:-}" ] || { echo "unknown job $j" >&2; exit 1; }; done

if [ -z "${SCORE_ONLY:-}" ]; then
    for j in $JOBS; do
        # shellcheck disable=SC2086
        SEEDS="$SEEDS" O="$O" bash "$HERE/scripts/audit_20261107/sweep_axis.sh" "$j" ${V[$j]}
    done
fi

echo "== $(date +%H:%M) validation scoring (CPU)"
rr() { echo "$HOME/neuro-causal-pfn-perfold/outputs_val_clean/$1/$2/replica/E13a/E13a/seed0/disconnectome-extrasclean/replica_results.csv"; }
for j in $JOBS; do
    for s in $SEEDS; do
        L="sweep_${j}_s$s"
        [ -s "$O/$j/s$s/pfn.pt" ] || { echo "   $L: no checkpoint, skipped"; continue; }
        [ -s "$(rr "$L" receptor)" ] && [ -s "$(rr "$L" genetics)" ] && continue
        CUDA_VISIBLE_DEVICES="" REPS="$REPS" bash "$HERE/scripts/audit_20261028/score_val_clean.sh" "$L" \
            --estimators "$O/$j/s$s/pfn.pt" --only-estimators &
    done
    wait
done
echo "== $(date +%H:%M) screening table"
"${PY:-$HOME/env/bin/python}" "$HERE/scripts/audit_20261107/stage1_screen.py" --seeds $SEEDS \
    --levels $(for j in $JOBS; do echo "$j=sweep_${j}_s{s}"; done) --out "$O/stage1_screen.csv"
