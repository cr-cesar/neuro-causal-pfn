#!/bin/bash
# One entry point for the leak-free re-runs on Myriad. Run from
# ~/Scratch/neuro-causal-pfn (the main clone) on a login node; every block
# submits its jobs with their dependencies and returns at once.
#
#   bash scripts/myriad_phase1b.sh setup      # causalpfn + faiss, pretrained weights, pull both repos
#   bash scripts/myriad_phase1b.sh t4 [eids]  # Tier 4 of the design: off-the-shelf CausalPFN on per-fold latents (CPU); ROOT=outputs_perfold_pub scores the 32-epoch variants into outputs_perfold_pub_t4
#   bash scripts/myriad_phase1b.sh ensemble E1 E5   # seed ensemble (3 seeds concatenated fold by fold) of finished per-fold reps, LR/ET + CausalPFN
#   bash scripts/myriad_phase1b.sh repair            # resubmit published-budget folds that died (time limit after a slow start) + their scoring
#   bash scripts/myriad_phase1b.sh perfold E1 E3 E11b    # published-budget (32-epoch) per-fold training into PUB_ROOT, independent jobs, scoring chained
#   POOL_LESIONS=<dir> POOL_LATENTS=<npz> bash scripts/myriad_phase1b.sh phase2   # cache -> reduced PFN (GPU) -> scored on E1 per-fold (published budget)
#   bash scripts/myriad_phase1b.sh e12            # curriculum ablation: 3 variants x SEEDS on the pilot's cache; "e12 summary" collects the rows
#   bash scripts/myriad_phase1b.sh clinical <cohort.csv> <id-col> <images-dir> "<latents glob>" "<outcome specs>" "<covariates>" <vol-col> [regress cols]
#   bash scripts/myriad_phase1b.sh status     # queue by state and block, latest leaderboards
#
# Counting with ls | wc -l always carries "|| true": under set -o pipefail an
# ls with no match would abort the script inside $(...).
# REPS is always handed to qsub through the environment (-v REPS, no value):
# variant labels carry commas (E11b[backbone=cnn,w_dice=1.0]) that -v NAME=value would split.
# Environment knobs: PERFOLD (default ~/Scratch/neuro-causal-pfn-perfold),
# GROUP_TABLE (default outputs/groups_public.csv here, outputs_perfold/groups_public.csv there),
# SEEDS (default 3), OUT_ROOT (default outputs_perfold_grp), T4_OUT (default outputs_perfold_t4).
set -euo pipefail

MAIN="$(cd "$(dirname "$0")/.." && pwd)"
PERFOLD="${PERFOLD:-$HOME/Scratch/neuro-causal-pfn-perfold}"
SEEDS="${SEEDS:-3}"
OUT_ROOT="${OUT_ROOT:-outputs_perfold_grp}"
T4_OUT="${T4_OUT:-outputs_perfold_t4}"
# published-budget (32-epoch) per-fold runs go to their own root: the layout is
# <root>/<eid>/<label>/seed<S>/folds with no budget in the path, so reusing
# outputs_perfold_grp would overwrite the 200-epoch E1/E5 folds
PUB_ROOT="${PUB_ROOT:-outputs_perfold_pub}"
PF_GROUPS="${PF_GROUPS:-outputs_perfold/groups_public.csv}"
cmd="${1:-status}"; shift || true

# works from a non-interactive ssh too: the module command is a shell function
# that only login shells define
activate() {
  type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh 2>/dev/null || true
  module load python3/3.11 2>/dev/null || true
  source ~/venvs/neuro/bin/activate
}

case "$cmd" in
  setup)
    activate
    (cd "$MAIN" && git pull -q origin main && pip install -q -e . >/dev/null)
    (cd "$PERFOLD" && git pull -q origin main)
    # faiss must come as a wheel: the source build needs a C++17 compiler and
    # fails with Myriad's icc 18. Newer wheels need a newer glibc than the
    # login nodes have, so try from newest to oldest.
    pip install -q --only-binary=:all: "faiss-cpu==1.9.0.post1" \
      || pip install -q --only-binary=:all: "faiss-cpu==1.8.0.post1" \
      || pip install -q --only-binary=:all: "faiss-cpu==1.7.4"
    pip install -q --no-deps "causalpfn==0.1.4"
    pip install -q --only-binary=:all: huggingface_hub tqdm
    python -c "import faiss, causalpfn; print('faiss', faiss.__version__)"
    python - <<'PY'
from causalpfn import CATEEstimator
CATEEstimator("cpu").load_model()
print("CausalPFN weights cached in ~/.cache/causalpfn")
PY
    ;;

  t4)
    # the design's Tier 4: fixed off-the-shelf estimator over each frozen
    # per-fold representation, same folds and trials as the LR/ET scoring.
    # Every label folder of the given eids under ROOT (default the 200-epoch
    # root; ROOT=outputs_perfold_pub for the published-budget variants) gets
    # one CPU job; rows land in the T4 root of that training root (T4_OUT for
    # the 200-epoch root, <ROOT>_t4 otherwise).
    cd "$PERFOLD"
    root="${ROOT:-$OUT_ROOT}"
    # one T4 root per training root: the replica layout has no budget in its
    # path, so 32-epoch and 200-epoch scorings of the same label would
    # overwrite each other inside a shared root (outputs_perfold_pub -> outputs_perfold_pub_t4)
    t4out="$T4_OUT"; [ "$root" = "$OUT_ROOT" ] || t4out="${root}_t4"
    for eid in "${@:-E1 E5}"; do
      for reps in "$root/$eid"/*/seed*/folds; do
        [ -d "$reps" ] || { echo "skip $eid: no folds under $root/$eid"; continue; }
        n=$(ls "$reps"/fold*.npz 2>/dev/null | wc -l || true)
        [ "$n" -eq 10 ] || { echo "skip $reps ($n folds)"; continue; }
        s=${reps%/folds}; s=${s##*/seed}
        lbl=$(basename "$(dirname "$(dirname "$reps")")")
        # already scored under the fixed estimator: skip, so a later run only
        # picks up the sets that have completed since
        n_done=$(ls "$t4out/replica/$eid/${lbl//\//_}/seed$s"/*-singles/replica_headline.csv 2>/dev/null | wc -l || true)
        [ "$n_done" -eq 0 ] || { echo "done  $lbl seed$s"; continue; }
        jid=$(REPS="$reps" qsub -terse -N t4-${eid,,}s$s -l h_rt=6:0:0 -l mem=8G \
              -v REPS,GROUP_TABLE=$PF_GROUPS,TEST_SINGLES=1,ESTIMATORS=causalpfn,ONLY_ESTIMATORS=1,OUT=$t4out \
              qsub/perfold_score.qsub.sh)
        echo "t4 $lbl seed$s -> job $jid"
      done
    done
    echo "leaderboard: $PERFOLD/$t4out/leaderboard.csv"
    ;;

  ensemble)
    # seed ensemble of finished per-fold representations: the three seeds'
    # fold latents concatenated column-wise (label <eid>+x3), scored on the
    # same folds and singles twice, with LR/ET (into ROOT) and with the fixed
    # CausalPFN (into T4_OUT). ROOT defaults to the 200-epoch root; set
    # ROOT=outputs_perfold_pub for the published-budget folds.
    cd "$PERFOLD"
    root="${ROOT:-$OUT_ROOT}"
    for eid in "${@:-E1 E5}"; do
      reps="$root/$eid/$eid/seed?/folds"
      n=$(ls -d $reps 2>/dev/null | wc -l || true)
      [ "$n" -ge 2 ] || { echo "skip $eid: $n seed folders under $root"; continue; }
      j1=$(REPS="$reps" qsub -terse -N ens-${eid,,} -l h_rt=4:0:0 \
           -v REPS,GROUP_TABLE=$PF_GROUPS,TEST_SINGLES=1,ENSEMBLE=1,ENSEMBLE_ONLY=1,OUT=$root qsub/perfold_score.qsub.sh)
      j2=$(REPS="$reps" qsub -terse -N ens-${eid,,}-t4 -l h_rt=6:0:0 -l mem=8G \
           -v REPS,GROUP_TABLE=$PF_GROUPS,TEST_SINGLES=1,ENSEMBLE=1,ENSEMBLE_ONLY=1,ESTIMATORS=causalpfn,ONLY_ESTIMATORS=1,OUT=$T4_OUT qsub/perfold_score.qsub.sh)
      echo "ensemble $eid ($n seeds, $root): LR/ET job $j1 -> $root/leaderboard.csv; CausalPFN job $j2 -> $T4_OUT/leaderboard.csv"
    done
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
        --h-rt 1:30:0 --groups "$PF_GROUPS" --out-root "$PUB_ROOT" > /tmp/perfold_submit_$$.sh
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
      reps="$PUB_ROOT/$eid/$label/seed$seed/folds"
      sj=$(REPS="$reps" qsub -terse -N sc-${eid,,}r${run}s$seed -hold_jid "${HOLD[$key]}" -l h_rt=3:0:0 \
           -v REPS,GROUP_TABLE=$PF_GROUPS,TEST_SINGLES=1,WITH_VOLUME=1,OUT=$PUB_ROOT qsub/perfold_score.qsub.sh)
      echo "$eid run$run seed$seed: folds ${HOLD[$key]} -> scoring $sj"
    done
    rm -f /tmp/perfold_submit_$$.sh
    ;;

  repair)
    # resubmit the published-budget folds that died (a job killed at its time
    # limit after a slow start leaves no fold<k>.npz) with a 3 h limit, and
    # hold a fresh scoring job on them. Only tasks that are neither queued nor
    # running are resubmitted, so it is safe to run at any time. The queued
    # jobs cannot be altered (Myriad's JSV rejects qalter -l), hence the repair.
    activate
    cd "$PERFOLD"
    plan=$(python -m ncpfold.plan --eids ${*:-E1 E3 E11b} --seeds "$SEEDS" --budget published --emit-qsub --independent \
           --h-rt 3:0:0 --groups "$PF_GROUPS" --out-root "$PUB_ROOT")
    # full job names: the plain qstat table truncates them to 10 characters, so
    # a fold-10 task (e11br2s2f10) would never match and be resubmitted
    queued=$(qstat -xml 2>/dev/null | sed -n 's/.*<JB_name>\([^<]*\)<\/JB_name>.*/\1/p')
    for d in "$PUB_ROOT"/E*/*/seed*/folds; do
      [ -d "$d" ] || continue
      rel=${d#$PUB_ROOT/}; eid=${rel%%/*}
      label=$(basename "$(dirname "$(dirname "$d")")"); seed=${d%/folds}; seed=${seed##*/seed}
      hold=""
      for k in $(seq 1 10); do
        [ -f "$d/fold$((k - 1)).npz" ] && continue
        line=$(printf '%s\n' "$plan" | grep -F -- "EID=$eid," | grep -F -- "SEED=$seed," | grep -F -- " -t $k " | grep -F -- "# $label")
        [ -n "$line" ] || { echo "no plan line for $eid $label seed$seed task $k"; continue; }
        name=$(echo "$line" | sed -E 's/.*-N ([^ ]+).*/\1/')
        if printf '%s\n' "$queued" | grep -qx "$name"; then continue; fi    # still queued or running
        jid=$(eval "$line" | cut -d. -f1)
        echo "$eid $label seed$seed fold$((k - 1)): resubmitted as $jid (3 h limit)"
        hold="${hold:+$hold,}$jid"
      done
      if [ -n "$hold" ]; then
        sj=$(REPS="$d" qsub -terse -N sc-${eid,,}s$seed-r -hold_jid "$hold" -l h_rt=3:0:0 \
             -v REPS,GROUP_TABLE=$PF_GROUPS,TEST_SINGLES=1,WITH_VOLUME=1,OUT=$PUB_ROOT qsub/perfold_score.qsub.sh)
        echo "  scoring $sj held on $hold"
        continue
      fi
      # all ten folds present but no headline and no scoring queued: the
      # chained scoring failed (or was never submitted) -> score now
      # the replica folder is named after the channel (disconnectome-singles, ...)
      n_head=$(ls "$PUB_ROOT/replica/$eid/${label//\//_}/seed$seed"/*-singles/replica_headline.csv 2>/dev/null | wc -l || true)
      n_npz=$(ls "$d"/fold*.npz 2>/dev/null | wc -l || true)
      if [ "$n_npz" -eq 10 ] && [ "$n_head" -eq 0 ]; then
        run_name=$(printf '%s\n' "$plan" | grep -F -- "EID=$eid," | grep -F -- "SEED=$seed," | grep -F -- " -t 1 " | grep -F -- "# $label" | sed -E 's/.*-N ([^ ]+)f1 .*/\1/')
        if printf '%s\n' "$queued" | grep -qx "sc-$run_name" || printf '%s\n' "$queued" | grep -qx "sc-${eid,,}s$seed-r"; then continue; fi
        sj=$(REPS="$d" qsub -terse -N sc-${eid,,}s$seed-r -l h_rt=3:0:0 \
             -v REPS,GROUP_TABLE=$PF_GROUPS,TEST_SINGLES=1,WITH_VOLUME=1,OUT=$PUB_ROOT qsub/perfold_score.qsub.sh)
        echo "$eid $label seed$seed: 10 folds, no headline -> scoring $sj"
      fi
    done
    ;;

  phase2)
    # Phase 2 pilot as one chain: anatomy cache of the covariate pool (CPU)
    # -> reduced Neuro-Causal-PFN, one sequential GPU job -> the checkpoint
    # scored as a fixed in-context estimator on the E1 per-fold folds (CPU),
    # next to the CausalPFN rows of T4_OUT. The pool is given on the command
    # line: POOL_LESIONS (lesion dir) and POOL_LATENTS (matching latents npz).
    cd "$MAIN"
    lesions="${POOL_LESIONS:?set POOL_LESIONS to the lesion directory of the pool}"
    latents="${POOL_LATENTS:?set POOL_LATENTS to the latents npz of the pool}"
    tag="${TAG:-pilot}"; cache="outputs/prior_cache/${tag}.npz"; out="outputs/pfn_reduced_${tag}"
    cj=$(qsub -terse -N prior-$tag -v LESIONS="$lesions",LATENTS="$latents",OUT="$cache" scripts/build_prior_cache_myriad.qsub.sh)
    tj=$(qsub -terse -N pfn-$tag -hold_jid "$cj" \
         -v CACHE="$cache",OUT="$out",ITERS="${ITERS:-20000}",SEED="${SEED:-0}",ARCH="${ARCH:-tabicl}" scripts/train_pfn_myriad.qsub.sh)
    cd "$PERFOLD"
    # the checkpoint is scored on the headline encoder's per-fold latents: E1
    # with the published budget (SCORE_REPS overrides; compare with the
    # CausalPFN row of the same reps, outputs_perfold_pub_t4)
    score_reps="${SCORE_REPS:-$PUB_ROOT/E1/E1/seed0/folds}"
    sj=$(REPS="$score_reps" qsub -terse -N sc-pfn-$tag -hold_jid "$tj" -l h_rt=6:0:0 -l mem=8G \
         -v REPS,GROUP_TABLE=$PF_GROUPS,TEST_SINGLES=1,ESTIMATORS="$MAIN/$out/pfn.pt",ONLY_ESTIMATORS=1,OUT=outputs_perfold_pfn/$tag \
         qsub/perfold_score.qsub.sh)
    echo "phase2 $tag: cache $cj -> pfn $tj ($out/pfn.pt, ~5 h) -> scoring on $score_reps $sj -> $PERFOLD/outputs_perfold_pfn/$tag/leaderboard.csv"
    ;;

  e12)
    # the curriculum ablation of the design: three variants (nocurr, ctx,
    # ctx+stages) x SEEDS, each one sequential GPU job of ~5 h on the same
    # prior cache as the pilot (TAG), each checkpoint scored as a fixed
    # estimator on the headline encoder's per-fold latents. Rows are collected
    # by "e12 summary" (one leaderboard per checkpoint under outputs_perfold_pfn).
    tag="${TAG:-kch_e1}"; cache="outputs/prior_cache/${tag}.npz"
    if [ "${1:-}" = "summary" ]; then
      activate
      cd "$PERFOLD"
      python - <<'PY'
import glob, os, pandas as pd
rows = []
# the pilot (phase2 before tags) wrote at the root of outputs_perfold_pfn; tagged runs under their own folder
files = [(f, "pilot") for f in glob.glob("outputs_perfold_pfn/replica/**/replica_headline.csv", recursive=True)]
files += [(f, f.split("/")[1]) for f in glob.glob("outputs_perfold_pfn/*/replica/**/replica_headline.csv", recursive=True)]
for f, run in sorted(files):
    r = pd.read_csv(f).iloc[0]
    rows.append({"run": run, "variant": run.rsplit("_s", 1)[0], "pehe_paper": r["pehe_paper_mean"], "balacc": r["balacc_mean"]})
if not rows:
    print("no PFN scorings yet"); raise SystemExit
d = pd.DataFrame(rows); print(d.round(4).to_string(index=False)); print()
print(d.groupby("variant").agg(n=("pehe_paper", "size"), pehe_mean=("pehe_paper", "mean"), pehe_std=("pehe_paper", lambda v: v.std(ddof=0)),
                               balacc_mean=("balacc", "mean")).round(4).to_string())
PY
      exit 0
    fi
    activate
    cd "$MAIN"
    [ -f "$cache" ] || { echo "no prior cache $cache: run phase2 first (it builds it)"; exit 2; }
    score_reps="${SCORE_REPS:-$PUB_ROOT/E1/E1/seed0/folds}"
    for v in ${VARIANTS:-nocurr ctx ctx+stages}; do
      for s in $(seq 0 $((SEEDS - 1))); do
        run="${v//+/-}_s$s"; out="outputs/pfn_e12_${run}"
        cd "$MAIN"
        tj=$(qsub -terse -N pfn-$run -v CACHE="$cache",OUT="$out",ITERS="${ITERS:-20000}",SEED="$s",ARCH="${ARCH:-tabicl}",VARIANT="$v" scripts/train_pfn_myriad.qsub.sh)
        cd "$PERFOLD"
        sj=$(REPS="$score_reps" qsub -terse -N sc-$run -hold_jid "$tj" -l h_rt=6:0:0 -l mem=8G \
             -v REPS,GROUP_TABLE=$PF_GROUPS,TEST_SINGLES=1,ESTIMATORS="$MAIN/$out/pfn.pt",ONLY_ESTIMATORS=1,OUT=outputs_perfold_pfn/$run \
             qsub/perfold_score.qsub.sh)
        echo "e12 $v seed$s: pfn $tj -> scoring $sj (outputs_perfold_pfn/$run)"
      done
    done
    echo "when done: bash scripts/myriad_phase1b.sh e12 summary"
    ;;

  clinical)
    # clinical cohort_csv id_col images_dir "latents glob" "outcome specs" "covariates" vol_col [regress cols...]
    cd "$MAIN"
    cohort=$1; idcol=$2; images=$3; latents=$4; outcomes=$5; covars=$6; volcol=$7; shift 7
    name=$(basename "$cohort" .csv)
    jid=$(qsub -terse -N clin-$name -v COHORT="$cohort",IDCOL="$idcol",IMAGES="$images",LATENTS="$latents",OUTCOMES="$outcomes",COVARS="$covars",VOLCOL="$volcol",REGRESS="$*",OUT=outputs/clinical_$name.csv \
          scripts/clinical_validation_myriad.qsub.sh)
    echo "clinical $name -> job $jid (outputs/clinical_$name.csv)"
    ;;

  status)
    echo "== queue (state x block)"
    qstat 2>/dev/null | awk 'NR>2 {split($3,a,"-"); split(a[1],b,"r"); print $5, substr($3,1,4)}' | sort | uniq -c | sort -k2,2 -k3,3
    echo "== running now"; qstat 2>/dev/null | awk 'NR>2 && $5=="r" {print "  "$3, $6, $7}'
    for lb in "$PERFOLD/$OUT_ROOT/leaderboard.csv" "$PERFOLD/$T4_OUT/leaderboard.csv" "$PERFOLD/$PUB_ROOT/leaderboard.csv" "$PERFOLD/${PUB_ROOT}_t4/leaderboard.csv"; do
      [ -f "$lb" ] && { echo "== $lb"; column -s, -t < "$lb" | cut -c1-140; }
    done
    ;;

  *) echo "unknown command $cmd"; exit 2 ;;
esac
