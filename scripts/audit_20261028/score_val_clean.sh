#!/bin/bash
# Validation scoring with the clean context (1 Nov): ncpfold.score --test-extras
# at the per-fold commit whose extras_folds removes every other image of the
# queried patients from the context (79e8b3c, branch extras-clean-context).
# Both splits run in parallel and the script waits for them; results land next
# to the earlier leaky ones under .../disconnectome-extrasclean, so nothing is
# overwritten. Aggregates only in the logs (PEHE, balanced accuracy, counts).
#
#   bash scripts/audit_20261028/score_val_clean.sh LABEL [ncpfold.score args]
#   bash scripts/audit_20261028/score_val_clean.sh finalK_lin4p8M_s0 --estimators ~/p2_bcb/pfn_finalK/lin4p8M/s0/pfn.pt --only-estimators
#   bash scripts/audit_20261028/score_val_clean.sh lret                       # logistic regression and extra trees
#
# Defaults are siena's; CUDA_VISIBLE_DEVICES is inherited ("" for CPU).
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 LABEL [ncpfold.score args]" >&2; exit 1; }
L="$1"; shift
PF="${PF:-$HOME/ncp_pf_clean}"
PF_COMMIT="${PF_COMMIT:-79e8b3c}"
SRC="${SRC:-$HOME/ncp_g8/src}"
PY="${PY:-$HOME/env/bin/python}"
V="${V:-$HOME/neuro-causal-pfn-perfold/outputs_val_clean}"
REPS="${REPS:-$HOME/neuro-causal-pfn-perfold/outputs_perfold_ws/E1/E1/seed0/folds}"
BCB="${BCB:-$HOME/export_bcb}"
LOGS="${LOGS:-$HOME/p2_bcb/clean_val_logs}"
mkdir -p "$LOGS"
have=$(git -C "$PF" rev-parse --short=7 HEAD)
[ "$have" = "$PF_COMMIT" ] || { echo "$PF is at $have, not $PF_COMMIT" >&2; exit 1; }
[ "$(git -C "$PF" status --porcelain --untracked-files=no | wc -l)" -eq 0 ] || { echo "$PF has uncommitted changes" >&2; exit 1; }
for m in receptor genetics; do
    mkdir -p "$V/$L/$m"
    [ -d "$V/$L/$m/labels" ] || cp -r "$V/C1/$m/labels" "$V/$L/$m/"
    (cd "$PF" && env PYTHONPATH="$PF:$SRC" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PFN_CONTEXT_CAP=3000 nice -n 10 \
        "$PY" -m ncpfold.score --reps "$REPS" --data-dir "$BCB/Full data" --atlas-dir "$BCB/atlases" \
        --groups "$BCB/fold_table.csv" --test-extras --extras-context clean --modality "$m" \
        --out-root "$V/$L/$m" "$@") > "$LOGS/$L-$m.log" 2>&1 &
done
wait
for m in receptor genetics; do
    echo "== $L $m"
    grep -hE "PEHE paper|had another image" "$LOGS/$L-$m.log" || echo "no headline: see $LOGS/$L-$m.log"
done
