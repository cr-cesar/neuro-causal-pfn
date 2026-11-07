#!/bin/bash
# Wave 0, item 2, scoring (Brief 8 audit, 7 Nov), on siena, CPU, validation
# pseudo-folds only (E13a seed 0's validation latents, the clean context).
# Contrasts, paired over the 16 deficits, each split and their mean:
#   held vs seen (PRIMARY): each deficit scored by the PFNs that never saw its
#       network, against the PFNs that did; both sides come from the same four
#       leave-out trainings, so a difference that leaving atoms out makes
#       everywhere (fewer atoms, another prior composition) cancels. The
#       dissertation's single-seed 500k run gave +0.0028 [-0.0012, 0.0067].
#   held vs final1M, seen vs final1M (secondary): against the dissertation's
#       model, which differs from these runs in machine and commit as well.
# Every seed's rows enter one composite, so paired_deficit_test.py averages a
# deficit over seeds (and, for "seen", over the three PFNs that saw it).
# Aggregates only in the logs.
#
#   CK_DIR=~/p2_bcb/pfn_lno3 bash scripts/audit_20261107/lno3_score.sh
set -euo pipefail
PY="${PY:-$HOME/env/bin/python}"
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
W="${W:-$HOME/p2_bcb}"
CK_DIR="${CK_DIR:-$W/pfn_lno3}"
REPS="${REPS:-$W/e13/E13a/E13a/seed0/folds}"
V="${V:-$HOME/neuro-causal-pfn-perfold/outputs_val_clean}"
ATLAS="${ATLAS:-$HOME/export_bcb/atlases}"
SEEDS="${SEEDS:-0 1 2}"
METRICS="${METRICS:-pehe_paper prescriptive_balacc}"
GROUPS_LNO=("1+2+3+4" "5+6+7+8" "9+10+11+12" "13+14+15+16")
REP=E13a/E13a/seed0
O="$CK_DIR/paired"; mkdir -p "$O"
rr() { echo "$V/$1/$2/replica/$REP/disconnectome-extrasclean/replica_results.csv"; }

for s in $SEEDS; do for g in 1 2 3 4; do
    [ -s "$CK_DIR/g${g}_s$s/pfn.pt" ] || { echo "no checkpoint $CK_DIR/g${g}_s$s/pfn.pt" >&2; exit 1; }
done; done
for s in $SEEDS; do [ -s "$(rr "final1M_s$s" receptor)" ] || { echo "no final1M seed $s results" >&2; exit 1; }; done

echo "== $(date +%H:%M) scoring the leave-out PFNs on $REP (CPU)"
for s in $SEEDS; do
    for g in 1 2 3 4; do
        L="lno3_g${g}_s$s"
        [ -s "$(rr "$L" receptor)" ] && [ -s "$(rr "$L" genetics)" ] && continue
        CUDA_VISIBLE_DEVICES="" REPS="$REPS" bash "$HERE/scripts/audit_20261028/score_val_clean.sh" "$L" \
            --estimators "$CK_DIR/g${g}_s$s/pfn.pt" --only-estimators &
    done
    wait
done

# one composite per seed (lno_compose.py refuses a network left out twice);
# the paired test then reads the three seeds' composites together
for keep in held seen; do
    for s in $SEEDS; do
        for m in receptor genetics; do
            args=()
            for g in 1 2 3 4; do args+=(--group "${GROUPS_LNO[$((g - 1))]}=$(rr "lno3_g${g}_s$s" $m)"); done
            "$PY" "$HERE/scripts/audit_20261028/lno_compose.py" --atlas-dir "$ATLAS" --keep $keep "${args[@]}" \
                --out "$CK_DIR/composite/$keep/s$s/$m/replica_results.csv"
        done
    done
done
held() { for s in $SEEDS; do echo "$CK_DIR/composite/held/s$s/$1/replica_results.csv"; done; }
seen() { for s in $SEEDS; do echo "$CK_DIR/composite/seen/s$s/$1/replica_results.csv"; done; }
final() { for s in $SEEDS; do rr "final1M_s$s" "$1"; done; }

pair() {  # tag, function giving A's files for a split, function giving B's files
    local tag="$1" afn="$2" bfn="$3" metric="$4"
    for m in receptor genetics; do
        # shellcheck disable=SC2046
        "$PY" "$HERE/scripts/paired_deficit_test.py" --a $($afn $m) --b $($bfn $m) --metric "$metric" \
            --label-a "${tag%%_vs_*}" --label-b "${tag#*_vs_}" --out "$O/${tag}_${m}_${metric}.csv" | sed -n "1,3p"
    done
    "$PY" "$HERE/scripts/paired_split_mean.py" --receptor "$O/${tag}_receptor_${metric}_per_deficit.csv" \
        --genetics "$O/${tag}_genetics_${metric}_per_deficit.csv" --label-a "${tag%%_vs_*}" --label-b "${tag#*_vs_}" \
        --out "$O/${tag}_both_${metric}.csv" | sed -n "1,3p"
}
for metric in $METRICS; do
    echo "== $metric: held vs seen (primary)"; pair lno3held_vs_lno3seen held seen "$metric"
    echo "== $metric: held vs final1M";        pair lno3held_vs_final1M held final "$metric"
    echo "== $metric: seen vs final1M";        pair lno3seen_vs_final1M seen final "$metric"
done
echo "== md5 of the *_both files"
for f in "$O"/*_both_*.csv; do echo "   $(basename "$f") $(md5sum "$f" | cut -c1-12)"; done
