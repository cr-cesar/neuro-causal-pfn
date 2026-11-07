#!/bin/bash
# Wave 0, item 1 (Brief 8 audit, 7 Nov): the dissertation's final model (three
# seeds) and the generic panel on the validation pseudo-folds in the cells the
# final model was never scored in: noisy (TE + RE = 1), location bias, and
# allocation by responsiveness. CPU for the PFN, LR / ET and CausalPFN;
# TabPFN runs where the panel's venv and GPU are (PANEL=1). Validation only:
# nothing here touches the UCLH test folds.
#
# Needs the patched giles_replica.py (pehe_implied, responsiveness cells) in
# the source tree the scorer imports (SRC of score_val_clean.sh). Every row of
# replica_results.csv then carries pehe_paper (against tau in {-1, 0, 1}, the
# paper's convention) and pehe_implied (against TE (1 - RE) tau, the CATE the
# cell implies); the two coincide in the ideal, location-bias and
# responsiveness cells.
#
#   FINAL=~/p2_bcb/pfn_final1m bash scripts/audit_20261107/wave0_cells.sh
#   CELLS="noisy_te050" PANEL=1 bash scripts/audit_20261107/wave0_cells.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
W="${W:-$HOME/p2_bcb}"
FINAL="${FINAL:-$W/pfn_final1m}"
REPS="${REPS:-$W/e13/E13a/E13a/seed0/folds}"      # the final model reads E13a latents, not E1
SEEDS="${SEEDS:-0 1 2}"
PANEL="${PANEL:-0}"
CELLS="${CELLS:-noisy_te075 noisy_te050 noisy_te025 location_bias_050 location_bias_095 location_bias responsiveness_060 responsiveness_075 responsiveness_090}"
grep -q "def implied_tau" "${SRC:-$HOME/ncp_g8/src}/neurocausalpfn/prior/giles_replica.py" \
    || { echo "the scorer's source tree lacks the 7 Nov giles_replica patch" >&2; exit 1; }
for s in $SEEDS; do [ -s "$FINAL/s$s/pfn.pt" ] || { echo "no checkpoint $FINAL/s$s/pfn.pt" >&2; exit 1; }; done

for c in $CELLS; do
    echo "== $(date +%H:%M) cell $c"
    for s in $SEEDS; do
        CUDA_VISIBLE_DEVICES="" REPS="$REPS" bash "$HERE/scripts/audit_20261028/score_val_clean.sh" \
            "w0_${c}_final1M_s$s" --scenario "$c" --estimators "$FINAL/s$s/pfn.pt" --only-estimators &
    done
    # logistic regression and extra trees (built-ins) and the off-the-shelf CausalPFN
    CUDA_VISIBLE_DEVICES="" REPS="$REPS" bash "$HERE/scripts/audit_20261028/score_val_clean.sh" \
        "w0_${c}_lret_causalpfn" --scenario "$c" --estimators causalpfn &
    wait
    if [ "$PANEL" = 1 ]; then
        REPS="$REPS" bash "$HERE/scripts/audit_20261028/score_val_clean.sh" \
            "w0_${c}_tabpfn" --scenario "$c" --classifiers tabpfn
    fi
done
echo "== $(date +%H:%M) done; compare with scripts/paired_deficit_test.py --metric pehe_implied (and pehe_paper)"
