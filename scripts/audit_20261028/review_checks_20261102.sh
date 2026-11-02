#!/bin/bash
# CPU checks of the external review of 2 Nov, on siena, from a clean worktree
# at this commit (no GPU, no training). Aggregates only, in $OUT:
#
# 1. the prior of the final model (g4c_launch.sh with PRIOR_CHECK_ONLY=1, so
#    the cfg is the launch's): cache metadata and atom count, class shares by
#    family against the gate's default requirements, realised family shares,
#    context scale and distinct rows per context (prior_checks.py);
# 2. where the final model loses to TabPFN-3.5 (S-learner) on validation:
#    paired tests over the 16 deficits of pehe_paper, mse_between,
#    mse_within and the mean effect per true class (effect_decomposition);
# 3. the two family-weight runs (theory : virtual-trial 0.25 : 0.75 and
#    0.75 : 0.25) against the final model, paired as for the other ablations.
#
#   bash scripts/audit_20261028/review_checks_20261102.sh            # all three
#   STEPS="2 3" bash scripts/audit_20261028/review_checks_20261102.sh
set -euo pipefail
STEPS="${STEPS:-1 2 3}"
PY="${PY:-$HOME/env/bin/python}"
V="${V:-$HOME/neuro-causal-pfn-perfold/outputs_val_clean}"
P="${P:-$HOME/p2_bcb/pfn_finalK_gw/paired}"
OUT="${OUT:-$HOME/p2_bcb/review_checks_20261102}"
REF="${REF:-finalK_lin4p8M}"
TABPFN="${TABPFN:-tabpfn_e1}"
mkdir -p "$OUT" "$P"
rr() { echo "$V/$1/$2/replica/E1/E1/seed0/disconnectome-extrasclean/replica_results.csv"; }

if [[ " $STEPS " == *" 1 "* ]]; then
    echo "== 1 prior checks (final model's launch, CPU)"
    env VENV="$(dirname "$(dirname "$PY")")" OBJECTIVE=new ATLAS_SRC="$HOME/export_bcb/atlases" SEED=0 \
        RUN_ITERS=125000 CACHE="$HOME/p2_bcb/pools/kch_s0/fold0.npz" OUT="$OUT/launch_out" \
        PRIOR_CHECK_ONLY=1 PRIOR_CHECK_OUT="$OUT" PRIOR_CHECK_ARGS="--batches 25 --contexts 512 3000" \
        bash scripts/audit_20261028/g4c_launch.sh 2>&1 | grep -vE "^\s*$" | tee "$OUT/prior_checks.log"
fi

if [[ " $STEPS " == *" 2 "* ]]; then
    echo "== 2 final model vs TabPFN-3.5 (S-learner), validation, by component"
    for m in receptor genetics; do
        ls $(rr "${REF}_s*" $m) >/dev/null
        ls "$(rr "$TABPFN" $m)" >/dev/null
    done
    for metric in pehe_paper mse_between mse_within tau_hat_pos tau_hat_zero tau_hat_neg; do
        for m in receptor genetics; do
            "$PY" scripts/paired_deficit_test.py --a "$(rr "${REF}_s*" $m)" --b "$(rr "$TABPFN" $m)" \
                --b-filter classifier=tabpfn learner=one --metric "$metric" \
                --label-a finalK --label-b tabpfnS --out "$OUT/finalK_vs_tabpfnS_${m}_${metric}.csv" | sed -n "1,2p"
        done
        "$PY" scripts/paired_split_mean.py --receptor "$OUT/finalK_vs_tabpfnS_receptor_${metric}_per_deficit.csv" \
            --genetics "$OUT/finalK_vs_tabpfnS_genetics_${metric}_per_deficit.csv" --label-a finalK --label-b tabpfnS \
            --out "$OUT/finalK_vs_tabpfnS_both_${metric}.csv" | sed -n "1,2p"
    done
fi

if [[ " $STEPS " == *" 3 "* ]]; then
    echo "== 3 family weights vs the final model, validation"
    for t in fw025075 fw075025; do
        for m in receptor genetics; do
            "$PY" scripts/paired_deficit_test.py --a "$(rr "finalK${t}*_s*" $m)" --b "$(rr "${REF}_s*" $m)" \
                --label-a "$t" --label-b finalK --out "$P/${t}_vs_finalK_${m}_pehe_paper.csv" | sed -n "1,3p"
        done
        "$PY" scripts/paired_split_mean.py --receptor "$P/${t}_vs_finalK_receptor_pehe_paper_per_deficit.csv" \
            --genetics "$P/${t}_vs_finalK_genetics_pehe_paper_per_deficit.csv" --label-a "$t" --label-b finalK \
            --out "$P/${t}_vs_finalK_both_pehe_paper.csv" | sed -n "1,3p"
    done
fi
echo "== md5"
md5sum "$OUT"/*.csv "$P"/fw0*_vs_finalK_* 2>/dev/null | cut -c1-12 | tr '\n' ' '; echo
