#!/bin/bash
# Leave-networks-out (2 Nov), scoring on siena: the four PFNs trained on
# Myriad (e13pool's launch, seed 0, each with one group of four networks left
# out of the virtual-trial family's partitions) scored on E13a seed 0's
# validation latents (CPU), then paired against e13pool seed 0, which saw
# every network:
#   held: each deficit scored by the PFN that never saw its network;
#   seen: each deficit scored by the three PFNs that did (does leaving
#         networks out cost anything where nothing is hidden?).
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
REF="${REF:-finalKe13pool_lin4p8M_s0}"
GROUPS_LNO=("1+2+3+4" "5+6+7+8" "9+10+11+12" "13+14+15+16")
O="$CK_DIR/paired"; mkdir -p "$O"
rr() { echo "$V/$1/$2/replica/$3/disconnectome-extrasclean/replica_results.csv"; }

for g in 1 2 3 4; do [ -s "$CK_DIR/g$g/pfn.pt" ] || { echo "no checkpoint $CK_DIR/g$g/pfn.pt" >&2; exit 1; }; done
for g in 1 2 3 4; do
    [ -s "$(rr "lno_g${g}_s0" receptor E13a/E13a/seed0)" ] && continue
    CUDA_VISIBLE_DEVICES="" REPS="$REPS" bash "$HERE/scripts/audit_20261028/score_val_clean.sh" "lno_g${g}_s0" \
        --estimators "$CK_DIR/g$g/pfn.pt" --only-estimators &
done
wait

for keep in held seen; do
    for m in receptor genetics; do
        args=()
        for g in 1 2 3 4; do args+=(--group "${GROUPS_LNO[$((g - 1))]}=$(rr "lno_g${g}_s0" $m E13a/E13a/seed0)"); done
        "$PY" "$HERE/scripts/audit_20261028/lno_compose.py" --atlas-dir "$ATLAS" --keep $keep "${args[@]}" \
            --out "$CK_DIR/composite/$keep/$m/replica_results.csv"
        "$PY" "$HERE/scripts/paired_deficit_test.py" --a "$CK_DIR/composite/$keep/$m/replica_results.csv" \
            --b "$(rr "$REF" $m E13a/E13a/seed0)" --label-a "lno_$keep" --label-b e13pool_s0 \
            --out "$O/lno_${keep}_vs_e13pool_s0_${m}_pehe_paper.csv" | sed -n "1,3p"
    done
    "$PY" "$HERE/scripts/paired_split_mean.py" --receptor "$O/lno_${keep}_vs_e13pool_s0_receptor_pehe_paper_per_deficit.csv" \
        --genetics "$O/lno_${keep}_vs_e13pool_s0_genetics_pehe_paper_per_deficit.csv" --label-a "lno_$keep" --label-b e13pool_s0 \
        --out "$O/lno_${keep}_vs_e13pool_s0_both_pehe_paper.csv" | sed -n "1,3p"
done
echo "== md5"
md5sum "$O"/*.csv | cut -c1-12 | tr '\n' ' '; echo
