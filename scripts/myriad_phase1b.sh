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
#   REP=outputs_perfold_pub/E1/E1/seed0/folds TAG=e1 FOLDS="0 1" PRIOR_FAMILY=theory bash scripts/myriad_phase1b.sh phase2pf   # one PFN per fold on that fold's UCLH training side
#   CONTEXT_MAX=3000 (training context; default 1024 with susceptible sampling) and AUGMENT=0 (no latent rotation) apply to phase2, e12 and phase2pf
#   D_MODEL=512 N_LAYERS=12 ITERS=60000 (stage 3, scale) apply to phase2 like CONTEXT_MAX
#   TRAIN_HRT=8:0:0 shortens the GPU training job's wall-clock request (default 36 h; before a
#   cluster drain only jobs that fit before the outage start)
#   ONLY="E4[25+25] E6[fusion_mode=both] E7b" bash scripts/myriad_phase1b.sh perfold E4 E6 E7b   # only those variants
#   POOL_IMAGES=<disco dir> bash scripts/myriad_phase1b.sh pool   # pool latents with the published-budget E1 (fold-0 encoder), one GPU job
#   PRIOR_FAMILY=<theory|giles|mixture> TAG=<tag> [POOL_IMAGES=<dir> POOL_LATENTS=<npz> HOLD_JID=<job>] bash scripts/myriad_phase1b.sh phase2   # [cache ->] reduced PFN (GPU) -> scored on E1 per-fold (published budget)
#   bash scripts/myriad_phase1b.sh probe [pfn.pt ...]   # CPU probe: does a checkpoint read the treatment, and on which prior family is it accurate
#   PRIOR_HYPER="two_arms=1 effect_scale=0.2:1 binary_outcome=0.5" VARIANT_TAG=<name> ... phase2   # theory prior extensions (see intersynth_theory.py); the tag enters the run name
#   ITERS=100000 LR_SCHEDULE=cosine BATCH=16 ... phase2   # longer training with warmup + cosine decay and a larger batch
#   bash scripts/myriad_phase1b.sh pfnscore <run> ...   # score finished PFN checkpoints again (context capped at the checkpoint's context_max)
#   PRIOR_FAMILY=<family> TAG=<tag> [POOL_IMAGES=<dir> POOL_LATENTS=<npz> HOLD_JID=<job>] bash scripts/myriad_phase1b.sh e12   # curriculum ablation: 3 variants x SEEDS on cache TAG (built first if missing); "e12 summary" collects the rows
#
# POOL_IMAGES is one directory for the whole Phase 2 chain: the pool block
# encodes it into POOL_LATENTS and phase2 / e12 build the prior cache from it,
# so the prior's responsiveness and the covariates come from the same images
# (the disconnectomes, which the scoring labels as well). POOL_LESIONS is the
# former name of the cache's directory and is accepted with a warning.
#   bash scripts/myriad_phase1b.sh clinical <cohort.csv> <id-col> <images-dir> "<latents glob>" "<outcome specs>" "<covariates>" <vol-col> [regress cols]
#   bash scripts/myriad_phase1b.sh status     # queue by state and block, latest leaderboards
#   bash scripts/myriad_phase1b.sh snapshot   # aggregate snapshot for the audit log into status/<date>/ (then commit to results-snapshot)
#   SCENARIO=location_bias ROOT=outputs_perfold_pub bash scripts/myriad_phase1b.sh scenario E1   # confounded cell under LR/ET and CausalPFN (audit F01)
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

# Phase 2: the Neuro-Prior family is always chosen explicitly. The theory
# family alone (the pilot's) has non-negative, continuous, mostly small effects
# and scored at chance on the virtual trial (PEHE paper 0.608, balanced
# accuracy 0.50); giles covers the virtual trial; mixture draws from both.
need_family() {
  # context pool of the prior (where a process draws its context): susceptible
  # (default) or all (the pilot's sampling); "-all" marks the latter in run names
  ctx="${CONTEXT_POOL:-susceptible}"; ctxsuf=""
  case "$ctx" in susceptible) ;; all) ctxsuf="-all" ;; *) echo "CONTEXT_POOL must be susceptible or all"; exit 2 ;; esac
  fam="${PRIOR_FAMILY:-}"
  case "$fam" in
    theory|giles|mixture) ;;
    *) echo "set PRIOR_FAMILY=theory|giles|mixture (FAMILY_WEIGHTS=\"<theory> <giles>\" for the mixture, default \"0.5 0.5\");"
       echo "the theory family alone scored at chance on the virtual trial; check a checkpoint with: bash scripts/myriad_phase1b.sh probe <pfn.pt>"
       exit 2 ;;
  esac
}

# the prior cache of tag $1: sets cache and hold (the -hold_jid option for the
# jobs that read it). A missing cache is built from POOL_IMAGES (the directory
# the pool latents were exported from) and POOL_LATENTS by a CPU job, after
# HOLD_JID when given (e.g. the pool export). The cache job checks that the
# channel of the images (lesion masks or disconnectomes) matches the latents'
# and fails otherwise (ALLOW_CHANNEL_MISMATCH=1 overrides).
ensure_cache() {
  cache="outputs/prior_cache/$1.npz"; hold=()
  if [ -s "$cache" ]; then echo "using the existing prior cache $cache"; return 0; fi
  local images="${POOL_IMAGES:-}" latents="${POOL_LATENTS:-}" cj
  if [ -z "$images" ] && [ -n "${POOL_LESIONS:-}" ]; then
    echo "warning: POOL_LESIONS is the former name of POOL_IMAGES; the cache must be built from the directory the pool latents were exported from" >&2
    images="$POOL_LESIONS"
  fi
  # a cache job for this tag already queued or running (a second call before
  # it finishes): hold on it instead of submitting a second writer
  cj=$(qstat -xml 2>/dev/null | tr -d '\n' | sed 's/<job_list/\n<job_list/g' \
       | grep "<JB_name>prior-$1</JB_name>" | sed -n 's/.*<JB_job_number>\([0-9]*\)<.*/\1/p' | head -1 || true)
  if [ -n "$cj" ]; then hold=(-hold_jid "$cj"); echo "prior cache job $cj already queued -> $cache"; return 0; fi
  if [ -z "$images" ] || [ -z "$latents" ]; then
    echo "no prior cache $cache: set POOL_IMAGES (the directory the pool latents were exported from) and POOL_LATENTS to build it (HOLD_JID=<job> to wait for the pool export)"; exit 2
  fi
  [ -d "$images" ] || { echo "no image dir $images"; exit 2; }
  if [ -z "${HOLD_JID:-}" ] && [ ! -s "$latents" ]; then
    echo "no latents $latents: wait for the pool export or pass HOLD_JID=<its job id>"; exit 2
  fi
  cj=$(ALLOW_CHANNEL_MISMATCH="${ALLOW_CHANNEL_MISMATCH:-}" qsub -terse -N prior-$1 ${HOLD_JID:+-hold_jid "$HOLD_JID"} \
       -v IMAGES="$images",LATENTS="$latents",OUT="$cache",ALLOW_CHANNEL_MISMATCH scripts/build_prior_cache_myriad.qsub.sh)
  hold=(-hold_jid "$cj")
  echo "prior cache $cj${HOLD_JID:+ (after $HOLD_JID)} -> $cache from $images (log prior-$1.o$cj)"
}

# short job-name prefix: plain qstat shows 10 characters, so the variant and
# seed go first and the tag last (full names: qstat -xml)
short_variant() { case "$1" in nocurr) echo nc ;; ctx) echo ctx ;; ctx+stages) echo cs ;; *) echo "${1//+/-}" ;; esac; }

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
    # H_RT: wall-clock per fold job (default 1:30:0, enough for one-channel
    # folds at ~0.5 h; two-channel folds measure ~0.9 h, use 3:0:0).
    # ONLY="E4[25+25] E6[fusion_mode=both] E7b" keeps those variants (label,
    # eid:run or eid); read into an array so the brackets are never globbed
    only=(); [ -z "${ONLY:-}" ] || { read -ra only <<< "$ONLY"; only=(--only "${only[@]}"); }
    python -m ncpfold.plan --eids $eids --seeds "$SEEDS" --budget published "${only[@]}"
    python -m ncpfold.plan --eids $eids --seeds "$SEEDS" --budget published "${only[@]}" --emit-qsub --independent \
        --h-rt "${H_RT:-1:30:0}" --groups "$PF_GROUPS" --out-root "$PUB_ROOT" > /tmp/perfold_submit_$$.sh
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
    # next to the CausalPFN rows of T4_OUT. PRIOR_FAMILY is required; the
    # cache of TAG is reused, or built from POOL_IMAGES (the directory the
    # pool latents were exported from) and POOL_LATENTS (the latents npz),
    # after HOLD_JID when given.
    need_family
    cd "$MAIN"
    # the architecture enters the run name when it is not the default, so that
    # ARCH=linear and the tabicl run of the same TAG and family do not share a
    # checkpoint, a job name or a leaderboard (the prior cache is still shared)
    archsuf=""; [ "${ARCH:-tabicl}" = tabicl ] || archsuf="-${ARCH}"
    # VARIANT_TAG names a prior variant (e.g. 2arms) so that runs of the same
    # TAG, family and architecture do not share a checkpoint
    tag="${TAG:-pilot}"; run="${tag}-${fam}${ctxsuf}${archsuf}${VARIANT_TAG:+-$VARIANT_TAG}"; out="outputs/pfn_reduced_${run}"
    ensure_cache "$tag"
    tj=$(FAMILY_WEIGHTS="${FAMILY_WEIGHTS:-}" PRIOR_HYPER="${PRIOR_HYPER:-}" LR_SCHEDULE="${LR_SCHEDULE:-constant}" BATCH="${BATCH:-}" CONTEXT_MAX="${CONTEXT_MAX:-}" AUGMENT="${AUGMENT:-}" CONTEXT_REPLACE="${CONTEXT_REPLACE:-}" STANDARDIZE="${STANDARDIZE:-}" D_MODEL="${D_MODEL:-}" N_LAYERS="${N_LAYERS:-}" qsub -terse -N pfn-$run -l h_rt="${TRAIN_HRT:-36:0:0}" ${hold[@]+"${hold[@]}"} \
         -v CACHE="$cache",OUT="$out",ITERS="${ITERS:-20000}",SEED="${SEED:-0}",ARCH="${ARCH:-tabicl}",FAMILY="$fam",FAMILY_WEIGHTS,PRIOR_HYPER,LR_SCHEDULE,BATCH,CONTEXT_MAX,AUGMENT,CONTEXT_REPLACE,STANDARDIZE,D_MODEL,N_LAYERS,CONTEXT_POOL="$ctx" \
         scripts/train_pfn_myriad.qsub.sh)
    cd "$PERFOLD"
    # the checkpoint is scored on the headline encoder's per-fold latents: E1
    # with the published budget (SCORE_REPS overrides; compare with the
    # CausalPFN row of the same reps, outputs_perfold_pub_t4)
    score_reps="${SCORE_REPS:-$PUB_ROOT/E1/E1/seed0/folds}"
    sj=$(REPS="$score_reps" PFN_CONTEXT_CAP="${PFN_CONTEXT_CAP:-}" qsub -terse -N sc-pfn-$run -hold_jid "$tj" -l h_rt=6:0:0 -l mem=8G \
         -v REPS,PFN_CONTEXT_CAP,GROUP_TABLE=$PF_GROUPS,TEST_SINGLES=1,ESTIMATORS="$MAIN/$out/pfn.pt",ONLY_ESTIMATORS=1,OUT=outputs_perfold_pfn/$run \
         qsub/perfold_score.qsub.sh)
    echo "phase2 $run: pfn $tj ($out/pfn.pt, ~1-2 h at 20,000 iterations with context_max 1024) -> scoring on $score_reps $sj -> $PERFOLD/outputs_perfold_pfn/$run/leaderboard.csv"
    ;;

  pfnscore)
    # score finished PFN checkpoints again (a scoring job that died, or a new
    # context cap): bash scripts/myriad_phase1b.sh pfnscore <run> [<run> ...],
    # <run> = the suffix of outputs/pfn_reduced_<run>/pfn.pt, scored into
    # outputs_perfold_pfn/<run>. The context is capped at the checkpoint's
    # context_max (PFN_CONTEXT_CAP overrides; 0 = whole train slice).
    cd "$PERFOLD"
    [ $# -gt 0 ] || { echo "usage: pfnscore <run> [<run> ...]"; exit 2; }
    score_reps="${SCORE_REPS:-$PUB_ROOT/E1/E1/seed0/folds}"
    for run in "$@"; do
      ck="$MAIN/outputs/pfn_reduced_${run}/pfn.pt"
      [ -s "$ck" ] || { echo "no checkpoint $ck"; continue; }
      sj=$(REPS="$score_reps" PFN_CONTEXT_CAP="${PFN_CONTEXT_CAP:-}" qsub -terse -N sc-pfn-$run -l h_rt=6:0:0 -l mem=8G \
           -v REPS,PFN_CONTEXT_CAP,GROUP_TABLE=$PF_GROUPS,TEST_SINGLES=1,ESTIMATORS="$ck",ONLY_ESTIMATORS=1,OUT=outputs_perfold_pfn/$run \
           qsub/perfold_score.qsub.sh)
      echo "pfnscore $run: $sj (log sc-pfn-$run.o$sj) -> $PERFOLD/outputs_perfold_pfn/$run/leaderboard.csv"
    done
    ;;

  phase2pf)
    # Per-fold pools (Phase 2, tab "latents and UCLH", options A'/B'): PFN k
    # pre-trains on the training side of fold k of REP, encoded by fold k's
    # own encoder (UCLH only; EXTRA_CACHE + EXTRA_LATENTS append a second
    # cohort such as KCH), and is scored on fold k's test patients only. One
    # CPU job builds the pools, one GPU job per fold trains, one CPU job
    # scores every fold with its own checkpoint. FOLDS (default all ten)
    # selects the folds (B': "0 1"); SEED defaults to the representation's;
    # the prior defaults to the headroom variant (PRIOR_HYPER overrides).
    #   REP=outputs_perfold_pub/E1/E1/seed0/folds TAG=e1 FOLDS="0 1" bash scripts/myriad_phase1b.sh phase2pf
    need_family
    cd "$MAIN"
    rep="${REP:?set REP to a per-fold representation folder, relative to $PERFOLD}"
    repdir="$PERFOLD/$rep"; [ -f "$repdir/meta.json" ] || { echo "no per-fold representation at $repdir"; exit 2; }
    tag="${TAG:?set TAG (names the run)}"; folds="${FOLDS:-0 1 2 3 4 5 6 7 8 9}"
    seed="${SEED:-$(python -c "import json; print(json.load(open('$repdir/meta.json'))['seed'])")}"
    cohort_cache="${COHORT_CACHE:-outputs/prior_cache/uclh_disco_x.npz}"
    [ -s "$cohort_cache" ] || { echo "no cohort anatomy cache $cohort_cache (scripts/build_prior_cache_myriad.qsub.sh)"; exit 2; }
    hyper="${PRIOR_HYPER-two_arms=1 effect_scale=0.2:1 d_source=network s_source=genetics_1 s_b_source=genetics_2 effect_headroom=1}"
    archsuf=""; [ "${ARCH:-linear}" = linear ] || archsuf="-${ARCH}"
    # POOL=extra: the pool is the EXTRA cohort alone with fold k's latents
    # (the KCH benchmark pool decided on 19 Oct); run names carry -xonly
    xsuf=""; [ -z "${EXTRA_CACHE:-}" ] || { xsuf="-x"; [ "${POOL:-}" = extra ] && xsuf="-xonly"; }
    extra_only=""; [ "${POOL:-}" = extra ] && extra_only=1
    run="${tag}-${fam}${ctxsuf}${archsuf}${VARIANT_TAG:+-$VARIANT_TAG}${xsuf}_s$seed"
    pools="outputs/prior_cache/fold_pools/${run}"
    cj=$(FOLDS="$folds" EXTRA_CACHE="${EXTRA_CACHE:-}" EXTRA_LATENTS="${EXTRA_LATENTS:-}" EXTRA_ONLY="$extra_only" qsub -terse -N pools-$tag \
         -v CACHE="$cohort_cache",FOLD_DIR="$repdir",OUT_DIR="$pools",FOLDS,EXTRA_CACHE,EXTRA_LATENTS,EXTRA_ONLY scripts/fold_pools_myriad.qsub.sh)
    echo "phase2pf $run: pools $cj -> $pools (folds $folds; prior: $hyper)"
    tjs=""
    for k in $folds; do
      out="outputs/pfn_pf_${run}/fold$k"
      tj=$(FAMILY_WEIGHTS="${FAMILY_WEIGHTS:-}" PRIOR_HYPER="$hyper" LR_SCHEDULE="${LR_SCHEDULE:-constant}" BATCH="${BATCH:-}" CONTEXT_MAX="${CONTEXT_MAX:-}" AUGMENT="${AUGMENT:-}" CONTEXT_REPLACE="${CONTEXT_REPLACE:-}" STANDARDIZE="${STANDARDIZE:-}" \
           qsub -terse -N pf$k-$tag -l h_rt="${TRAIN_HRT:-36:0:0}" -hold_jid "$cj" \
           -v CACHE="$pools/fold$k.npz",OUT="$out",ITERS="${ITERS:-20000}",SEED="$seed",ARCH="${ARCH:-linear}",FAMILY="$fam",FAMILY_WEIGHTS,PRIOR_HYPER,LR_SCHEDULE,BATCH,CONTEXT_MAX,AUGMENT,CONTEXT_REPLACE,STANDARDIZE,CONTEXT_POOL="$ctx" \
           scripts/train_pfn_myriad.qsub.sh)
      tjs="${tjs:+$tjs,}$tj"
      echo "  fold $k: pfn $tj -> $out/pfn.pt"
    done
    cd "$PERFOLD"
    # SCORE_DATA_DIR / SCORE_MODALITY: the trials the folds are scored on
    # (default the per-fold repo's data; the BCB pipeline passes data_bcb)
    sj=$(REPS="$rep" PFN_CONTEXT_CAP="${PFN_CONTEXT_CAP:-0}" ONLY_FOLDS="$folds" DATA_DIR="${SCORE_DATA_DIR:-data/Full data}" MODALITY="${SCORE_MODALITY:-receptor}" \
         qsub -terse -N sc-pf-$tag -hold_jid "$tjs" -l h_rt=6:0:0 -l mem=16G \
         -v REPS,PFN_CONTEXT_CAP,ONLY_FOLDS,DATA_DIR,MODALITY,GROUP_TABLE=$PF_GROUPS,TEST_SINGLES=1,ESTIMATORS="$MAIN/outputs/pfn_pf_${run}/fold{fold}/pfn.pt",ONLY_ESTIMATORS=1,OUT="${SCORE_OUT:-outputs_perfold_pfn/pf_$run}" \
         qsub/perfold_score.qsub.sh)
    echo "  scoring $sj (each fold with its own checkpoint, full context) -> $PERFOLD/outputs_perfold_pfn/pf_$run/leaderboard.csv"
    ;;

  probe)
    # treatment sensitivity and accuracy per prior family of PFN checkpoints
    # (scripts/probe_pfn.py), one CPU job; default: the pilot's checkpoint.
    cd "$MAIN"
    ckpts="${*:-outputs/pfn_reduced_kch_e1/pfn.pt}"
    for c in $ckpts; do [ -f "$c" ] || { echo "no checkpoint $c"; exit 2; }; done
    out="${PROBE_OUT:-outputs/probe_pfn_$(date +%m%d_%H%M).csv}"
    jid=$(CKPTS="$ckpts" qsub -terse -N probe-pfn -v CKPTS,OUT="$out" scripts/probe_pfn_myriad.qsub.sh)
    echo "probe $jid -> $out (log probe-pfn.o$jid)"
    ;;

  pool)
    # latents of the external covariate pool for Phase 2, encoded with a
    # published-budget E1 fold encoder (the headline encoder family): one short
    # GPU job. POOL_IMAGES = the pool's disconnectome dir, the same directory
    # phase2 / e12 build the prior cache from (the export records it in the
    # npz and the cache job checks the channel against it); POOL_CKPT defaults
    # to seed 0 / fold 0 of the published-budget E1.
    cd "$MAIN"
    images="${POOL_IMAGES:?set POOL_IMAGES to the disconnectome dir of the pool}"
    [ -d "$images" ] || { echo "no image dir $images"; exit 2; }
    ckpt="${POOL_CKPT:-$PERFOLD/$PUB_ROOT/E1/E1/seed0/fold0/disco/vae_disconnectome.pt}"
    [ -f "$ckpt" ] || { echo "no checkpoint $ckpt"; exit 2; }
    out="${POOL_OUT:-outputs/latents_pool32}"
    jid=$(CKPTS="$ckpt" qsub -terse -N pool-export -v CKPTS,IMAGES="$images",OUT="$out" scripts/export_latents_myriad.qsub.sh)
    echo "pool export $jid -> $out/E1_seed0_disco.npz (log pool-export.o$jid); then, with the cache built from the same directory:"
    echo "  PRIOR_FAMILY=<family> TAG=<tag> POOL_IMAGES=\"$images\" POOL_LATENTS=$out/E1_seed0_disco.npz HOLD_JID=$jid bash scripts/myriad_phase1b.sh phase2"
    echo "  PRIOR_FAMILY=<family> TAG=<tag> POOL_IMAGES=\"$images\" POOL_LATENTS=$out/E1_seed0_disco.npz HOLD_JID=$jid bash scripts/myriad_phase1b.sh e12"
    ;;

  e12)
    # the curriculum ablation of the design: three variants (nocurr, ctx,
    # ctx+stages) x SEEDS, each one sequential GPU job (measured: ~12 h with
    # the context curriculum, ~18 h for nocurr, which trains at full context
    # throughout) on the prior cache TAG, each checkpoint scored as a fixed
    # estimator on the headline encoder's per-fold latents. Rows are collected
    # by "e12 summary" (one leaderboard per checkpoint under outputs_perfold_pfn).
    # A missing cache is built first (CPU job, the PFN jobs wait for it) from
    # POOL_IMAGES (the directory the pool latents were exported from) and
    # POOL_LATENTS; HOLD_JID makes that cache job wait for an
    # earlier job, e.g. the "pool" export that writes POOL_LATENTS. Building
    # the cache here instead of through phase2 avoids training phase2's PFN,
    # which is the same model as the ctx variant at seed 0.
    tag="${TAG:-kch_e1}"
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
    need_family
    archsuf=""; [ "${ARCH:-tabicl}" = tabicl ] || archsuf="-${ARCH}"
    activate
    cd "$MAIN"
    ensure_cache "$tag"
    score_reps="${SCORE_REPS:-$PUB_ROOT/E1/E1/seed0/folds}"
    for v in ${VARIANTS:-nocurr ctx ctx+stages}; do
      for s in $(seq 0 $((SEEDS - 1))); do
        # cache tag and prior family are part of the run name, so ablations
        # on two caches or two families never share a checkpoint
        run="${tag}-${fam}${ctxsuf}${archsuf}-${v//+/-}_s$s"; out="outputs/pfn_e12_${run}"
        jn="$(short_variant "$v")-s$s-${fam:0:3}-$tag"
        cd "$MAIN"
        tj=$(FAMILY_WEIGHTS="${FAMILY_WEIGHTS:-}" PRIOR_HYPER="${PRIOR_HYPER:-}" LR_SCHEDULE="${LR_SCHEDULE:-constant}" BATCH="${BATCH:-}" qsub -terse -N pfn-$jn ${hold[@]+"${hold[@]}"} \
             -v CACHE="$cache",OUT="$out",ITERS="${ITERS:-20000}",SEED="$s",ARCH="${ARCH:-tabicl}",VARIANT="$v",FAMILY="$fam",FAMILY_WEIGHTS,PRIOR_HYPER,LR_SCHEDULE,BATCH,CONTEXT_POOL="$ctx" \
             scripts/train_pfn_myriad.qsub.sh)
        cd "$PERFOLD"
        sj=$(REPS="$score_reps" PFN_CONTEXT_CAP="${PFN_CONTEXT_CAP:-}" qsub -terse -N sc-$jn -hold_jid "$tj" -l h_rt=6:0:0 -l mem=8G \
             -v REPS,PFN_CONTEXT_CAP,GROUP_TABLE=$PF_GROUPS,TEST_SINGLES=1,ESTIMATORS="$MAIN/$out/pfn.pt",ONLY_ESTIMATORS=1,OUT=outputs_perfold_pfn/$run \
             qsub/perfold_score.qsub.sh)
        echo "e12 $v seed$s: pfn $tj (log pfn-$jn.o$tj) -> scoring $sj (outputs_perfold_pfn/$run)"
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

  snapshot)
    # aggregate status snapshot for the audit log (no filenames, ids or
    # latents): queue, every leaderboard, the headline rows, per-deficit
    # slice sizes (n_train / n_test per deficit and fold) and the PFN timing
    # lines. Written to status/<date>/ in the main clone; commit and push that
    # folder to the results-snapshot branch so that it is readable from GitHub.
    activate
    day="${SNAPSHOT:-$(date +%Y%m%d_%H%M)}"; out="$MAIN/status/$day"; mkdir -p "$out"
    cd "$MAIN"
    bash scripts/myriad_phase1b.sh status > "$out/status.txt" 2>&1 || true
    { git -C "$MAIN" log --oneline -1; git -C "$PERFOLD" log --oneline -1; } > "$out/commits.txt"
    for lb in "$PERFOLD/$OUT_ROOT" "$PERFOLD/$T4_OUT" "$PERFOLD/$PUB_ROOT" "$PERFOLD/${PUB_ROOT}_"* "$PERFOLD/outputs_perfold_pfn" "$PERFOLD/outputs_perfold_pfn"/*; do
      [ -f "$lb/leaderboard.csv" ] || continue
      cp "$lb/leaderboard.csv" "$out/leaderboard_$(basename "$lb").csv"
      [ -f "$lb/leaderboard.md" ] && cp "$lb/leaderboard.md" "$out/leaderboard_$(basename "$lb").md"
    done
    PERFOLD="$PERFOLD" python scripts/snapshot_aggregates.py "$out"
    # PFN timing lines (iteration, context length, loss, wall-clock) of every PFN
    # log, with the prior options the run was trained with (anatomy sources,
    # two arms, effect headroom) from the config the job prints first
    for f in pfn-*.o*; do
      [ -f "$f" ] || continue
      echo "== $f"; grep -E '^node: |parameters|iter [0-9]+/|"(two_arms|d_source|s_source|s_b_source|effect_headroom)":' "$f" \
        | awk '/^node: / || /parameters/ || /"(two_arms|d_source|s_source|s_b_source|effect_headroom)":/ || /iter (1|100|1000|5000|10000|15000|20000)\//'
    done > "$out/pfn_timing.txt" 2>/dev/null || true
    ls "$out"
    # The clone stays on the local results-snapshot branch (main + status commits);
    # before launching anything, bring the code level with main: git fetch origin main && git merge origin/main
    echo "snapshot $out; publish it with: git add status && git commit -m 'status snapshot $day' && git push -u origin HEAD:results-snapshot"
    ;;

  scenario)
    # F01 of the audit log: score finished per-fold representations under a
    # confounded cell of the virtual trial (SCENARIO=location_bias, the paper's
    # 0.305 anchor) with LR/ET and with CausalPFN, into <root>_<scenario> and
    # <root>_<scenario>_t4, same folds and test set as the ideal cell.
    cd "$PERFOLD"
    sc="${SCENARIO:-location_bias}"; root="${ROOT:-$PUB_ROOT}"
    # MODALITY=genetics scores the same latents on the gene-expression split of
    # the atlas (the virtual trial is a different trial per split); a non-default
    # split gets its own output roots so it never overwrites the receptor cells
    mod="${MODALITY:-receptor}"; modsuf=""; [ "$mod" = receptor ] || modsuf="_${mod}"
    for eid in "${@:-E1}"; do
      for reps in "$root/$eid"/*/seed*/folds; do
        [ -d "$reps" ] || { echo "skip $eid: no folds under $root/$eid"; continue; }
        n=$(ls "$reps"/fold*.npz 2>/dev/null | wc -l || true); [ "$n" -eq 10 ] || { echo "skip $reps ($n folds)"; continue; }
        s=${reps%/folds}; s=${s##*/seed}
        j1=$(REPS="$reps" qsub -terse -N sc-${sc:0:3}${modsuf:1:3}-${eid,,}s$s -l h_rt=6:0:0 -l mem=8G \
             -v REPS,GROUP_TABLE=$PF_GROUPS,TEST_SINGLES=1,WITH_VOLUME=1,SCENARIO="$sc",MODALITY="$mod",OUT="${root}_${sc}${modsuf}" qsub/perfold_score.qsub.sh)
        j2=$(REPS="$reps" qsub -terse -N t4-${sc:0:3}${modsuf:1:3}-${eid,,}s$s -l h_rt=6:0:0 -l mem=8G \
             -v REPS,GROUP_TABLE=$PF_GROUPS,TEST_SINGLES=1,ESTIMATORS=causalpfn,ONLY_ESTIMATORS=1,SCENARIO="$sc",MODALITY="$mod",OUT="${root}_${sc}${modsuf}_t4" qsub/perfold_score.qsub.sh)
        echo "$sc $mod $eid seed$s: LR/ET $j1 -> ${root}_${sc}${modsuf}; CausalPFN $j2 -> ${root}_${sc}${modsuf}_t4"
      done
    done
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
