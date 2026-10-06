#!/bin/bash
# One entry point for the leak-free re-runs on Myriad. Run from
# ~/Scratch/neuro-causal-pfn (the main clone) on a login node; every block
# submits its jobs with their dependencies and returns at once.
#
#   bash scripts/myriad_phase1b.sh setup      # causalpfn + faiss, pretrained weights, pull both repos
#   bash scripts/myriad_phase1b.sh t4         # Tier 4 of the design: off-the-shelf CausalPFN on the per-fold latents of E1/E5 (CPU)
#   bash scripts/myriad_phase1b.sh perfold E3 E11b [E1]   # published-budget per-fold training, independent jobs, scoring chained
#   bash scripts/myriad_phase1b.sh clinical <cohort.csv> <id-col> <images-dir> "<latents glob>" "<outcome specs>" "<covariates>" <vol-col> [regress cols]
#   bash scripts/myriad_phase1b.sh status     # queue by state and block, latest leaderboards
#
# Environment knobs: PERFOLD (default ~/Scratch/neuro-causal-pfn-perfold),
# GROUP_TABLE (default outputs/groups_public.csv here, outputs_perfold/groups_public.csv there),
# SEEDS (default 3), OUT_ROOT (default outputs_perfold_grp), T4_OUT (default outputs_perfold_t4).
set -euo pipefail

MAIN="$(cd "$(dirname "$0")/.." && pwd)"
PERFOLD="${PERFOLD:-$HOME/Scratch/neuro-causal-pfn-perfold}"
SEEDS="${SEEDS:-3}"
OUT_ROOT="${OUT_ROOT:-outputs_perfold_grp}"
T4_OUT="${T4_OUT:-outputs_perfold_t4}"
PF_GROUPS="${PF_GROUPS:-outputs_perfold/groups_public.csv}"
cmd="${1:-status}"; shift || true

activate() { module load python3/3.11 2>/dev/null || true; source ~/venvs/neuro/bin/activate; }

case "$cmd" in
  setup)
    activate
    (cd "$MAIN" && git pull -q origin main && pip install -q -e . >/dev/null)
    (cd "$PERFOLD" && git pull -q origin main)
    pip install -q "causalpfn==0.1.4" faiss-cpu
    python - <<'PY'
from causalpfn import CATEEstimator
CATEEstimator("cpu").load_model()
print("CausalPFN weights cached in ~/.cache/causalpfn")
PY
    ;;

  t4)
    # the design's Tier 4: fixed off-the-shelf estimator over each frozen
    # per-fold representation, same folds and trials as the LR/ET scoring
    cd "$PERFOLD"
    for eid in E1 E5; do
      for s in $(seq 0 $((SEEDS - 1))); do
        reps="$OUT_ROOT/$eid/$eid/seed$s/folds"
        [ -d "$reps" ] || { echo "skip $reps (no folds)"; continue; }
        jid=$(qsub -terse -N t4-${eid,,}s$s -l h_rt=6:0:0 -l mem=8G \
              -v REPS="$reps",GROUP_TABLE=$PF_GROUPS,TEST_SINGLES=1,ESTIMATORS=causalpfn,ONLY_ESTIMATORS=1,OUT=$T4_OUT \
              qsub/perfold_score.qsub.sh)
        echo "t4 $eid seed$s -> job $jid"
      done
    done
    echo "leaderboard: $PERFOLD/$T4_OUT/leaderboard.csv (classifier column tells LR/ET vs causalpfn)"
    ;;

  perfold)
    # published budget (32-epoch cap, ~0.4-0.7 h per fold): one independent job
    # per fold (arrays are throttled harder), then one scoring job per
    # (eid, seed) held on its ten folds
    activate
    cd "$PERFOLD"
    eids="${*:-E3 E11b}"
    python -m ncpfold.plan --eids $eids --seeds "$SEEDS" --budget published
    python -m ncpfold.plan --eids $eids --seeds "$SEEDS" --budget published --emit-qsub --independent \
        --h-rt 1:30:0 --groups "$PF_GROUPS" --out-root "$OUT_ROOT" > /tmp/perfold_submit_$$.sh
    declare -A HOLD
    while read -r line; do
      [[ "$line" =~ ^qsub ]] || continue
      key=$(echo "$line" | sed -E 's/.*-v EID=([^,]+),RUN=([0-9]+),SEED=([0-9]+).*/\1 \2 \3/')
      jid=$(eval "$line" | cut -d. -f1)
      HOLD["$key"]="${HOLD[$key]:+${HOLD[$key]},}$jid"
    done < /tmp/perfold_submit_$$.sh
    for key in "${!HOLD[@]}"; do
      set -- $key; eid=$1; run=$2; seed=$3
      label=$(python -c "from ncpfold.catalogue import find_run; print(find_run('$eid', $run).label)")
      reps="$OUT_ROOT/$eid/$label/seed$seed/folds"
      sj=$(qsub -terse -N sc-${eid,,}r${run}s$seed -hold_jid "${HOLD[$key]}" -l h_rt=3:0:0 \
           -v REPS="$reps",GROUP_TABLE=$PF_GROUPS,TEST_SINGLES=1,OUT=$OUT_ROOT qsub/perfold_score.qsub.sh)
      echo "$eid run$run seed$seed: folds ${HOLD[$key]} -> scoring $sj"
    done
    rm -f /tmp/perfold_submit_$$.sh
    ;;

  clinical)
    # clinical cohort_csv id_col images_dir "latents glob" "outcome specs" "covariates" vol_col [regress cols...]
    cd "$MAIN"
    cohort=$1; idcol=$2; images=$3; latents=$4; outcomes=$5; covars=$6; volcol=$7; shift 7
    regress=""; for c in "$@"; do regress="$regress --regress \"$c\""; done
    outs=""; for o in $outcomes; do outs="$outs --outcome \"$o\""; done
    name=$(basename "$cohort" .csv)
    jid=$(qsub -terse -N clin-$name -l h_rt=4:0:0 -l mem=8G -pe smp 4 -cwd -j y -b y \
          "module load python3/3.11; source ~/venvs/neuro/bin/activate; \
           python scripts/clinical_validation.py --cohort '$cohort' --id-col '$idcol' --images-dir '$images' \
           --latents $latents $outs --covariates $covars --volume-col '$volcol' --with-covariates $regress \
           --out outputs/clinical_$name.csv")
    echo "clinical $name -> job $jid (outputs/clinical_$name.csv)"
    ;;

  status)
    echo "== queue (state x block)"
    qstat 2>/dev/null | awk 'NR>2 {split($3,a,"-"); split(a[1],b,"r"); print $5, substr($3,1,4)}' | sort | uniq -c | sort -k2,2 -k3,3
    echo "== running now"; qstat 2>/dev/null | awk 'NR>2 && $5=="r" {print "  "$3, $6, $7}'
    for lb in "$PERFOLD/$OUT_ROOT/leaderboard.csv" "$PERFOLD/$T4_OUT/leaderboard.csv"; do
      [ -f "$lb" ] && { echo "== $lb"; column -s, -t < "$lb" | cut -c1-140; }
    done
    ;;

  *) echo "unknown command $cmd"; exit 2 ;;
esac
