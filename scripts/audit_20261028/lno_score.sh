#!/bin/bash
# Leave-networks-out (2 Nov), scoring on siena: the control g0 (no network
# left out) and the four PFNs g1..g4 (one group of four networks left out of
# the virtual-trial family's partitions each), all e13pool's launch at seed 0
# from one checkout and machine (lno_launch.sh), scored on E13a seed 0's
# validation latents (CPU). Paired tests over the 16 deficits:
#   held vs control: each deficit scored by the PFN that never saw its
#       network, against the control that saw every network (the test);
#   seen vs control: each deficit scored by the three PFNs that did see it
#       (does leaving networks out cost anything where nothing is hidden?);
#   control vs e13pool seed 0: the same launch on another code commit (and,
#       when trained elsewhere, machine), i.e. the training noise floor.
# Aggregates only in the logs.
#
#   CK_DIR=~/p2_bcb/pfn_lno bash scripts/audit_20261028/lno_score.sh
set -euo pipefail
PY="${PY:-$HOME/env/bin/python}"
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
W="${W:-$HOME/p2_bcb}"
CK_DIR="${CK_DIR:-$W/pfn_lno}"
REPS="${REPS:-$W/e13/E13a/E13a/seed0/folds}"
V="${V:-$HOME/neuro-causal-pfn-perfold/outputs_val_clean}"
ATLAS="${ATLAS:-$HOME/export_bcb/atlases}"
E13POOL="${E13POOL:-finalKe13pool_lin4p8M_s0}"
GROUPS_LNO=("1+2+3+4" "5+6+7+8" "9+10+11+12" "13+14+15+16")
REP=E13a/E13a/seed0
O="$CK_DIR/paired"; mkdir -p "$O"
rr() { echo "$V/$1/$2/replica/$REP/disconnectome-extrasclean/replica_results.csv"; }

for g in 0 1 2 3 4; do [ -s "$CK_DIR/g$g/pfn.pt" ] || { echo "no checkpoint $CK_DIR/g$g/pfn.pt" >&2; exit 1; }; done
echo "== $(date +%H:%M) scoring g0..g4 on $REP (CPU)"
for g in 0 1 2 3 4; do
    [ -s "$(rr "lno_g${g}_s0" receptor)" ] && [ -s "$(rr "lno_g${g}_s0" genetics)" ] && continue
    CUDA_VISIBLE_DEVICES="" REPS="$REPS" bash "$HERE/scripts/audit_20261028/score_val_clean.sh" "lno_g${g}_s0" \
        --estimators "$CK_DIR/g$g/pfn.pt" --only-estimators &
done
wait

pair() {  # tag, A per modality (a function of m), B label
    local tag="$1" afn="$2" b="$3"
    for m in receptor genetics; do
        "$PY" "$HERE/scripts/paired_deficit_test.py" --a "$($afn $m)" --b "$(rr "$b" $m)" \
            --label-a "${tag%%_vs_*}" --label-b "${tag#*_vs_}" --out "$O/${tag}_${m}_pehe_paper.csv" | sed -n "1,3p"
    done
    "$PY" "$HERE/scripts/paired_split_mean.py" --receptor "$O/${tag}_receptor_pehe_paper_per_deficit.csv" \
        --genetics "$O/${tag}_genetics_pehe_paper_per_deficit.csv" --label-a "${tag%%_vs_*}" --label-b "${tag#*_vs_}" \
        --out "$O/${tag}_both_pehe_paper.csv" | sed -n "1,3p"
}
for keep in held seen; do
    for m in receptor genetics; do
        args=()
        for g in 1 2 3 4; do args+=(--group "${GROUPS_LNO[$((g - 1))]}=$(rr "lno_g${g}_s0" $m)"); done
        "$PY" "$HERE/scripts/audit_20261028/lno_compose.py" --atlas-dir "$ATLAS" --keep $keep "${args[@]}" \
            --out "$CK_DIR/composite/$keep/$m/replica_results.csv"
    done
done
held() { echo "$CK_DIR/composite/held/$1/replica_results.csv"; }
seen() { echo "$CK_DIR/composite/seen/$1/replica_results.csv"; }
ctl() { rr lno_g0_s0 "$1"; }
echo "== held vs control";        pair lnoheld_vs_control held lno_g0_s0
echo "== seen vs control";        pair lnoseen_vs_control seen lno_g0_s0
echo "== control vs e13pool s0";  pair control_vs_e13pools0 ctl "$E13POOL"
echo "== md5 of the *_both files"
for f in "$O"/*_both_pehe_paper.csv; do echo "   $(basename "$f") $(md5sum "$f" | cut -c1-12)"; done
