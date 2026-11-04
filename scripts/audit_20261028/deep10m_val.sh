#!/bin/bash
# Validation of the depth run (4 Nov), on siena: the four checkpoints of each
# seed of deep10m_launch.sh (12 layers; 250k, 500k, 750k and 1M trials),
# scored on E13a seed 0's validation latents (CPU), as scale15m_val.sh. Paired
# tests over the 16 deficits, PEHE paper and prescriptive balanced accuracy,
# each split and their mean:
#   each budget vs the same budget of scale15m (6 layers, same machine and
#   code): only the depth differs; 1M is the primary comparison;
#   12 layers at 1M vs final1M: against the model of the dissertation.
# Each scoring is skipped if its results exist. Aggregates only in the logs.
#
#   bash scripts/audit_20261028/deep10m_val.sh
set -euo pipefail
PY="${PY:-$HOME/env/bin/python}"
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
W="${W:-$HOME/p2_bcb}"
CK="${CK:-$W/pfn_deep10m}"
REPS="${REPS:-$W/e13/E13a/E13a/seed0/folds}"
V="${V:-$HOME/neuro-causal-pfn-perfold/outputs_val_clean}"
REP=E13a/E13a/seed0
O="$CK/paired"; mkdir -p "$O"
rr() { echo "$V/$1/$2/replica/$REP/disconnectome-extrasclean/replica_results.csv"; }
BUDGETS="250k 500k 750k 1000k"
ckpt() {  # seed folder, budget
    case "$2" in
        250k) echo "$CK/$1/pfn_it62500.pt" ;;   500k) echo "$CK/$1/pfn_it125000.pt" ;;
        750k) echo "$CK/$1/pfn_it187500.pt" ;;  1000k) echo "$CK/$1/pfn.pt" ;;
    esac
}

for n in s0 s1 s2; do for b in $BUDGETS; do
    [ -s "$(ckpt $n $b)" ] || { echo "no checkpoint $(ckpt $n $b)" >&2; exit 1; }
done; done
for s in 0 1 2; do [ -s "$(rr "final1M_s$s" receptor)" ] || { echo "no final1M seed $s results" >&2; exit 1; }; done
for b in $BUDGETS; do for s in 0 1 2; do
    [ -s "$(rr "scale15m_${b}_s$s" receptor)" ] || { echo "no scale15m $b seed $s results" >&2; exit 1; }
done; done
for b in $BUDGETS; do
    echo "== $(date +%H:%M) scoring $b, seeds 0-2, on $REP (CPU)"
    for n in s0 s1 s2; do
        L="deep10m_${b}_$n"
        [ -s "$(rr "$L" receptor)" ] && [ -s "$(rr "$L" genetics)" ] && continue
        CUDA_VISIBLE_DEVICES="" REPS="$REPS" bash "$HERE/scripts/audit_20261028/score_val_clean.sh" "$L" \
            --estimators "$(ckpt $n $b)" --only-estimators &
    done
    wait
    for n in s0 s1 s2; do
        for m in receptor genetics; do [ -s "$(rr "deep10m_${b}_$n" $m)" ] || { echo "no results for $b $n $m" >&2; exit 1; }; done
    done
done

pair() {  # tag, A label glob, B label glob
    local tag="$1" a="$2" b="$3"
    for metric in pehe_paper prescriptive_balacc; do
        for m in receptor genetics; do
            "$PY" "$HERE/scripts/paired_deficit_test.py" --a "$(rr "$a" $m)" --b "$(rr "$b" $m)" --metric $metric \
                --label-a "${tag%%_vs_*}" --label-b "${tag#*_vs_}" --out "$O/${tag}_${m}_${metric}.csv" | sed -n "1,3p"
        done
        "$PY" "$HERE/scripts/paired_split_mean.py" --receptor "$O/${tag}_receptor_${metric}_per_deficit.csv" \
            --genetics "$O/${tag}_genetics_${metric}_per_deficit.csv" --label-a "${tag%%_vs_*}" --label-b "${tag#*_vs_}" \
            --out "$O/${tag}_both_${metric}.csv" | sed -n "1,3p"
    done
}
for b in $BUDGETS; do
    echo "== 12 vs 6 layers at $b"; pair "deep${b}_vs_s15m${b}" "deep10m_${b}_s[012]" "scale15m_${b}_s[012]"
done
echo "== 12 layers at 1M vs final1M"; pair deep1000k_vs_final1M "deep10m_1000k_s[012]" "final1M_s[012]"
echo "== md5 of the *_both files"
for f in "$O"/*_both*.csv; do echo "   $(basename "$f") $(md5sum "$f" | cut -c1-12)"; done
echo "== $(date +%H:%M) done"
