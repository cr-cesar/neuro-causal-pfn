#!/bin/bash
# Validation of the final model at 1M trials (3 Nov), on siena: the three 1M
# seeds and the same-code, same-machine control at 500k (ctl500k_s0), all from
# final1m_launch.sh, scored on E13a seed 0's validation latents (CPU), as
# e13pool. Paired tests over the 16 deficits, PEHE paper and prescriptive
# balanced accuracy, each split and their mean:
#   1M vs ctl500k: only the budget differs (the control has seed 0 only);
#   1M s0 vs ctl500k: the same, at the same seed;
#   1M vs e13pool: the budget, and the code commit and machine of the training;
#   ctl500k vs e13pool s0: the code commit and machine alone (noise floor).
# Each scoring is skipped if its results exist. Aggregates only in the logs.
#
#   bash scripts/audit_20261028/final1m_val.sh
set -euo pipefail
PY="${PY:-$HOME/env/bin/python}"
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
W="${W:-$HOME/p2_bcb}"
CK="${CK:-$W/pfn_final1M}"
REPS="${REPS:-$W/e13/E13a/E13a/seed0/folds}"
V="${V:-$HOME/neuro-causal-pfn-perfold/outputs_val_clean}"
E13POOL="${E13POOL:-finalKe13pool_lin4p8M}"
REP=E13a/E13a/seed0
O="$CK/paired"; mkdir -p "$O"
rr() { echo "$V/$1/$2/replica/$REP/disconnectome-extrasclean/replica_results.csv"; }

for n in s0 s1 s2 ctl500k_s0; do [ -s "$CK/$n/pfn.pt" ] || { echo "no checkpoint $CK/$n/pfn.pt" >&2; exit 1; }; done
for s in 0 1 2; do [ -s "$(rr "${E13POOL}_s$s" receptor)" ] || { echo "no e13pool seed $s results" >&2; exit 1; }; done
echo "== $(date +%H:%M) scoring 1M s0..s2 and ctl500k_s0 on $REP (CPU)"
for n in s0 s1 s2 ctl500k_s0; do
    L="final1M_$n"
    [ -s "$(rr "$L" receptor)" ] && [ -s "$(rr "$L" genetics)" ] && continue
    CUDA_VISIBLE_DEVICES="" REPS="$REPS" bash "$HERE/scripts/audit_20261028/score_val_clean.sh" "$L" \
        --estimators "$CK/$n/pfn.pt" --only-estimators &
done
wait
for n in s0 s1 s2 ctl500k_s0; do
    for m in receptor genetics; do [ -s "$(rr "final1M_$n" $m)" ] || { echo "no results for $n $m" >&2; exit 1; }; done
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
echo "== 1M vs ctl500k";            pair final1M_vs_ctl500k "final1M_s[012]" final1M_ctl500k_s0
echo "== 1M s0 vs ctl500k";         pair final1Ms0_vs_ctl500k final1M_s0 final1M_ctl500k_s0
echo "== 1M vs e13pool";            pair final1M_vs_e13pool "final1M_s[012]" "${E13POOL}_s*"
echo "== ctl500k vs e13pool s0";    pair ctl500k_vs_e13pools0 final1M_ctl500k_s0 "${E13POOL}_s0"
echo "== md5 of the *_both files"
for f in "$O"/*_both_*.csv; do echo "   $(basename "$f") $(md5sum "$f" | cut -c1-12)"; done
