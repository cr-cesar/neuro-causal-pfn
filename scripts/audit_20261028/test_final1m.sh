#!/bin/bash
# The single test of the final model (3 Nov), on siena: the 1M Neuro-CausalPFN
# (final1m_launch.sh, three seeds) on the 2,828 test patients, with the
# protocol of the earlier final test (Table 4.10): ncpfold.score --test-singles
# with the fold table, both splits, 16 deficits, encoder seed = PFN seed (here
# E13a's per-fold encoders), one checkpoint per seed for all ten folds, no
# rotations. ncpfold is pinned at the validation commit (79e8b3c), whose
# changes over the earlier test's (f905819) touch only --test-extras and
# --only-folds, so the test path is the same.
#
# Comparators, fixed here before the test is run:
#   on E13a latents (same three encoder seeds): TabPFN-3.5 S-learner,
#       CausalPFN, logistic regression T-learner, ExtraTrees T-learner;
#   on E1 latents (already scored): TabPFN-3.5 S-learner, the earlier final
#       model (4.8M linear, 500k).
# Paired tests over the 16 trials of PEHE (pehe_paper) and balanced accuracy
# (prescriptive_balacc), final model - comparator; Holm over the 6 comparators
# x 2 metrics per split (family 12) and over the 6 comparators on the mean of
# the splits, per metric (family 6).
#
# Steps (STEPS="0 1 2 3 4" by default):
#   0 check the inputs: the three checkpoints and their md5 against CK_MD5
#     (the digests printed on beachcomber), the E13a folds of each seed, the
#     E1 files; counts only; stops if anything is missing;
#   1 the final model on the test (CPU); refuses if its output exists, so the
#     test is run once;
#   2 comparators on E13a latents: LR/ET and CausalPFN (CPU); TabPFN (GPU,
#     the panel venv; needs the TabPFN token exported in this shell); each
#     skipped if it exists;
#   3 paired tests and Holm;
#   4 summary table (aggregates).
#
#   CK_MD5="s0=<12> s1=<12> s2=<12>" STEPS=0 bash scripts/audit_20261028/test_final1m.sh
#   CK_MD5="s0=<12> s1=<12> s2=<12>" bash scripts/audit_20261028/test_final1m.sh
set -euo pipefail
STEPS="${STEPS:-0 1 2 3 4}"
PY="${PY:-$HOME/env/bin/python}"
PANEL_PY="${PANEL_PY:-$HOME/env_panel/bin/python}"   # TabPFN
CPFN_PY="${CPFN_PY:-$HOME/env_cpfn/bin/python}"      # CausalPFN (the causalpfn package)
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
PF="${PF:-$HOME/ncp_pf_clean}"
PF_COMMIT="${PF_COMMIT:-79e8b3c}"
SRC="${SRC:-$HOME/ncp_g8/src}"
W="${W:-$HOME/p2_bcb}"
CK="${CK:-$W/pfn_final1M}"
CK_MD5="${CK_MD5:?set CK_MD5 to \"s0=<md5 prefix> s1=... s2=...\" as printed on beachcomber}"
E13A="${E13A:-$W/e13/E13a/E13a}"
BCB="${BCB:-$HOME/export_bcb}"
PFD="${PFD:-$HOME/neuro-causal-pfn-perfold}"
T="${T:-$PFD/outputs_test_final1m}"
# E1 files, {m} for the split (set outside ${:-}, whose first } would end the default)
[ -n "${E1_TABPFN:-}" ] || E1_TABPFN="$PFD/outputs_scores_panel_{m}/replica/E1/E1/seed*/disconnectome-singles/replica_results.csv"
[ -n "${E1_FINAL:-}" ] || E1_FINAL="$PFD/outputs_test_final/finalK_lin4p8M_test_s*/{m}/replica/E1/E1/seed*/disconnectome-singles/replica_results.csv"
LOGS="$T/logs"; P="$T/paired"
SEEDS=(0 1 2)
mkdir -p "$LOGS" "$P"
# a results file of label L, split m, encoder seed s (glob "*" for all seeds)
rr() { echo "$T/$1/$2/replica/E13a/E13a/seed$3/disconnectome-singles/replica_results.csv"; }
e1() { local p="$1"; echo "${p//\{m\}/$2}"; }
n_files() { local n=0 f; for f in $1; do [ -s "$f" ] && n=$((n + 1)); done; echo $n; }

score() {  # label, seed, python, ncpfold.score args...
    local L="$1" s="$2" py="$3"; shift 3
    for m in receptor genetics; do
        mkdir -p "$T/$L/$m"
        (cd "$PF" && env -u PFN_ROTATIONS -u PFN_ARM_FLIP -u PFN_ROTATIONS_SEED PYTHONPATH="$PF:$SRC" \
            OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PFN_CONTEXT_CAP=3000 nice -n 10 \
            "$py" -m ncpfold.score --reps "$E13A/seed$s/folds" --data-dir "$BCB/Full data" \
            --atlas-dir "$BCB/atlases" --groups "$BCB/fold_table.csv" --test-singles --modality "$m" \
            --out-root "$T/$L/$m" "$@") > "$LOGS/$L-s$s-$m.log" 2>&1
    done
}

if [[ " $STEPS " == *" 0 "* ]]; then
    echo "== $(date +%H:%M) 0 inputs"
    have=$(git -C "$PF" rev-parse --short=7 HEAD)
    [ "$have" = "$PF_COMMIT" ] || { echo "$PF is at $have, not $PF_COMMIT" >&2; exit 1; }
    [ "$(git -C "$PF" status --porcelain --untracked-files=no | wc -l)" -eq 0 ] || { echo "$PF has uncommitted changes" >&2; exit 1; }
    bad=0
    "$CPFN_PY" -c "import causalpfn" 2>/dev/null || { echo "   $CPFN_PY has no causalpfn package"; bad=1; }
    for s in "${SEEDS[@]}"; do
        f="$CK/s$s/pfn.pt"; want=$(echo " $CK_MD5 " | sed -n "s/.* s$s=\([0-9a-f]*\) .*/\1/p")
        if [ ! -s "$f" ]; then echo "   s$s: no checkpoint"; bad=1; continue; fi
        got=$(md5sum "$f" | cut -c1-12)
        [ -n "$want" ] && [ "$got" = "$want" ] && ok=matches || { ok="DIFFERS from ${want:-<none>}"; bad=1; }
        n=$(n_files "$E13A/seed$s/folds/fold*.npz")
        [ "$n" -eq 10 ] || bad=1
        echo "   s$s: checkpoint md5 $got $ok; E13a seed$s folds $n/10"
    done
    for m in receptor genetics; do
        a=$(n_files "$(e1 "$E1_TABPFN" $m)"); b=$(n_files "$(e1 "$E1_FINAL" $m)")
        echo "   E1 $m: TabPFN files $a, earlier final model files $b (3 each expected)"
        [ "$a" -eq 3 ] && [ "$b" -eq 3 ] || bad=1
    done
    [ -d "$T/final1M_test_s0" ] && echo "   the final model's test output already exists: step 1 will refuse"
    [ "$bad" = 0 ] || { echo "inputs incomplete: nothing scored" >&2; exit 1; }
fi

if [[ " $STEPS " == *" 1 "* ]]; then
    echo "== $(date +%H:%M) 1 final model on the test (CPU)"
    for s in "${SEEDS[@]}"; do
        [ -e "$T/final1M_test_s$s" ] && { echo "$T/final1M_test_s$s exists: the test has been run" >&2; exit 1; }
    done
    for s in "${SEEDS[@]}"; do
        CUDA_VISIBLE_DEVICES="" score "final1M_test_s$s" "$s" "$PY" --estimators "$CK/s$s/pfn.pt" --only-estimators &
    done
    wait
    for s in "${SEEDS[@]}"; do for m in receptor genetics; do
        [ -s "$(rr "final1M_test_s$s" $m $s)" ] || { echo "no results for s$s $m: see $LOGS" >&2; exit 1; }
    done; done
fi

if [[ " $STEPS " == *" 2 "* ]]; then
    echo "== $(date +%H:%M) 2 comparators on E13a latents"
    for s in "${SEEDS[@]}"; do
        [ -s "$(rr lret_e13a genetics $s)" ] || CUDA_VISIBLE_DEVICES="" score lret_e13a "$s" "$PY" &
        [ -s "$(rr causalpfn_e13a genetics $s)" ] || CUDA_VISIBLE_DEVICES="" score causalpfn_e13a "$s" "$CPFN_PY" \
            --estimators causalpfn --only-estimators &
    done
    wait
    for s in "${SEEDS[@]}"; do   # GPU, one seed at a time
        [ -s "$(rr tabpfn_e13a genetics $s)" ] || score tabpfn_e13a "$s" "$PANEL_PY" --classifiers tabpfn
    done
    for L in lret_e13a causalpfn_e13a tabpfn_e13a; do for m in receptor genetics; do
        [ "$(n_files "$(rr $L $m '*')")" -eq 3 ] || { echo "$L $m: not all seeds scored, see $LOGS" >&2; exit 1; }
    done; done
fi

if [[ " $STEPS " == *" 3 "* ]]; then
    echo "== $(date +%H:%M) 3 paired tests"
    # comparator: tag, files of split m (a function of m), filters
    b_tabpfn()   { rr tabpfn_e13a "$1" '*'; }
    b_causal()   { rr causalpfn_e13a "$1" '*'; }
    b_lret()     { rr lret_e13a "$1" '*'; }
    b_tabpfnE1() { e1 "$E1_TABPFN" "$1"; }
    b_finalE1()  { e1 "$E1_FINAL" "$1"; }
    COMP=("tabpfnS_e13a b_tabpfn classifier=tabpfn learner=one"
          "causalpfn_e13a b_causal"
          "lrT_e13a b_lret classifier=logistic_regression learner=two"
          "etT_e13a b_lret classifier=extra_trees learner=two"
          "tabpfnS_e1 b_tabpfnE1 classifier=tabpfn learner=one"
          "final500k_e1 b_finalE1")
    for c in "${COMP[@]}"; do
        read -r tag fn flt <<< "$c"
        for metric in pehe_paper prescriptive_balacc; do
            for m in receptor genetics; do
                "$PY" "$HERE/scripts/paired_deficit_test.py" --a "$(rr 'final1M_test_s*' $m '*')" --b "$($fn $m)" \
                    ${flt:+--b-filter $flt} --metric "$metric" --label-a final1M --label-b "$tag" \
                    --out "$P/final1M_vs_${tag}_${m}_${metric}.csv" | sed -n "1,2p"
            done
            "$PY" "$HERE/scripts/paired_split_mean.py" --receptor "$P/final1M_vs_${tag}_receptor_${metric}_per_deficit.csv" \
                --genetics "$P/final1M_vs_${tag}_genetics_${metric}_per_deficit.csv" --label-a final1M --label-b "$tag" \
                --out "$P/final1M_vs_${tag}_both_${metric}.csv" | sed -n "1,2p"
        done
    done
    for m in receptor genetics; do
        "$PY" "$HERE/scripts/holm_adjust.py" --inputs "$P/final1M_vs_*_${m}_pehe_paper.csv" "$P/final1M_vs_*_${m}_prescriptive_balacc.csv" \
            --out "$P/holm_${m}.csv"
    done
    for metric in pehe_paper prescriptive_balacc; do
        "$PY" "$HERE/scripts/holm_adjust.py" --inputs "$P/final1M_vs_*_both_${metric}.csv" --out "$P/holm_both_${metric}.csv"
    done
fi

if [[ " $STEPS " == *" 4 "* ]]; then
    echo "== $(date +%H:%M) 4 summary"
    runs=()
    for s in "${SEEDS[@]}"; do runs+=(--run "final1M s$s=$(rr "final1M_test_s$s" '{m}' $s)"); done
    runs+=(--run "final1M mean3=$(rr 'final1M_test_s*' '{m}' '*')"
           --run "TabPFN-S E13a=$(rr tabpfn_e13a '{m}' '*'):classifier=tabpfn,learner=one"
           --run "CausalPFN E13a=$(rr causalpfn_e13a '{m}' '*')"
           --run "LR-T E13a=$(rr lret_e13a '{m}' '*'):classifier=logistic_regression,learner=two"
           --run "ET-T E13a=$(rr lret_e13a '{m}' '*'):classifier=extra_trees,learner=two"
           --run "TabPFN-S E1=$E1_TABPFN:classifier=tabpfn,learner=one"
           --run "final 500k E1=$E1_FINAL")
    "$PY" "$HERE/scripts/audit_20261028/val_table.py" "${runs[@]}" --out "$T/summary_test2828.csv"
    echo "== Holm, per split (family 12) and on the mean (family 6 per metric)"
    for f in "$P"/holm_*.csv; do
        echo "-- $(basename "$f")"
        "$PY" - "$f" <<'EOF'
import sys
import pandas as pd
d = pd.read_csv(sys.argv[1])
d["source"] = d["source"].str.replace("final1M_vs_", "").str.replace(".csv", "", regex=False)
cols = [c for c in ("split", "source", "diff_mean", "n_a_lower", "n_b_lower", "p_holm_t", "holm_family_size") if c in d]
print(d[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
EOF
    done
    echo "== md5"
    md5sum "$T/summary_test2828.csv" "$P"/holm_*.csv | cut -c1-12 | tr '\n' ' '; echo
fi
