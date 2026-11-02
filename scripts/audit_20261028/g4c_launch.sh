#!/bin/bash
# G4c (28 Oct): C1's launch on the current code, the control of G5 and TM.
# This is g5_launch.sh without G5's two allocation changes (bias=0:0
# bias_weights=1:0:0 in GILES_HYPER, gamma=0:0 in PRIOR_HYPER), i.e. the line
# G4c was trained with. SEED (default 0, G4c's own) gives a second training of
# the same configuration: the variation between two trainings on one code,
# which C1 against G4c could not measure (they differ in the prior code too).
# Run from the root of a clean checkout with the atlas at data/atlases; CACHE
# and OUT can be overridden, the defaults are siena's.
#
#   SEED=1 bash scripts/audit_20261028/g4c_launch.sh
#   OBJECTIVE=new [CATE_W=1] bash scripts/audit_20261028/g4c_launch.sh
#   STD=context [OBJECTIVE=new] bash scripts/audit_20261028/g4c_launch.sh
#   GENERIC_W=0 [OBJECTIVE=new] bash scripts/audit_20261028/g4c_launch.sh
#   HEAD_RANGE=wide bash scripts/audit_20261028/g4c_launch.sh
#   S_SOURCE=receptor | FAMILY_W="0.25 0.75" [OBJECTIVE=new] bash scripts/audit_20261028/g4c_launch.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
dirty=$(git status --porcelain --untracked-files=no | wc -l)
echo "code_commit=$(git rev-parse HEAD) cambios_sin_commit=$dirty"
[ "$dirty" -eq 0 ] || { echo "uncommitted changes: refusing to train" >&2; exit 1; }
export PYTHONPATH="$PWD/src"
# anything left over from an earlier launch must not reach this one
unset CACHE_ANCHOR_WEIGHT ITERS VARIANT BATCH CONTEXT_MIN AUGMENT CONTEXT_REPLACE STANDARDIZE D_MODEL N_LAYERS \
    BOTH_ARMS CATE_WEIGHT HEAD_LO HEAD_HI N_QUERY CURRICULUM N_BINS ATTN
# RUN_ITERS (29 Oct): training steps, 4 simulated trials each (20000 = 80k
# trials, G4c's); any other value is tagged in the run name (..._i25000)
RUN_ITERS="${RUN_ITERS:-20000}"
[[ "$RUN_ITERS" =~ ^[1-9][0-9]*$ ]] || { echo "RUN_ITERS must be a positive integer" >&2; exit 1; }
[ "$RUN_ITERS" = 20000 ] || export ITERS="$RUN_ITERS"
# RUN_ARCH / RUN_D_MODEL / RUN_N_LAYERS (29 Oct): the model size of the design
# question (linear or TabICL, about 20M parameters); default G4c's linear
# 256 x 6 (4.8M). TabICL's width comes from embed_dim x n_cls, so only its
# depth is set. Any non-default value is tagged in the run name
RUN_ARCH="${RUN_ARCH:-linear}"
case "$RUN_ARCH" in linear|tabicl) ;; *) echo "RUN_ARCH must be linear or tabicl" >&2; exit 1 ;; esac
for v in RUN_D_MODEL RUN_N_LAYERS; do
    [[ -z "${!v:-}" || "${!v}" =~ ^[1-9][0-9]*$ ]] || { echo "$v must be a positive integer" >&2; exit 1; }
done
[ -z "${RUN_D_MODEL:-}" ] || export D_MODEL="$RUN_D_MODEL"
[ -z "${RUN_N_LAYERS:-}" ] || export N_LAYERS="$RUN_N_LAYERS"
# RUN_N_BINS / RUN_BATCH / RUN_CONTEXT_MAX / RUN_ATTN (2 Nov): the rest of the
# design's scale (1,024 bins, 8 simulated trials per update, longer contexts);
# defaults are the final model's (256 bins, batch 4, context 3,000, masked
# attention). RUN_ATTN=sdpa is the same attention with memory linear in the
# context, needed for wide models at long contexts. Non-defaults are tagged
for v in RUN_N_BINS RUN_BATCH RUN_CONTEXT_MAX; do
    [[ -z "${!v:-}" || "${!v}" =~ ^[1-9][0-9]*$ ]] || { echo "$v must be a positive integer" >&2; exit 1; }
done
RUN_CONTEXT_MAX="${RUN_CONTEXT_MAX:-3000}"
RUN_ATTN="${RUN_ATTN:-masked}"
case "$RUN_ATTN" in masked|sdpa) ;; *) echo "RUN_ATTN must be masked or sdpa" >&2; exit 1 ;; esac
[ "$RUN_ATTN" = masked ] || export ATTN="$RUN_ATTN"
[ -z "${RUN_N_BINS:-}" ] || export N_BINS="$RUN_N_BINS"
[ -z "${RUN_BATCH:-}" ] || export BATCH="$RUN_BATCH"
command -v module >/dev/null 2>&1 && module load python3/3.11
source "${VENV:-$HOME/venvs/neuro}/bin/activate"
# OBJECTIVE=new (ablation of 29 Oct): G4c's prior with TM's training target
# only (both arms, the effect term, head -0.05 to 1.05); queries and context
# schedule stay G4c's, so G4c-objective against G4c isolates the target
OBJECTIVE="${OBJECTIVE:-old}"
case "$OBJECTIVE" in new|old) ;; *) echo "OBJECTIVE must be new or old" >&2; exit 1 ;; esac
tag=$([ "$OBJECTIVE" = new ] && echo G4cobj || echo G4c)
# CATE_W (29 Oct, with OBJECTIVE=new): weight of the effect term; 0.5 is
# G4cobj's, any other value is tagged in the run name (G4cobjcw1, ...)
CATE_W="${CATE_W:-0.5}"
[[ "$CATE_W" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "CATE_W must be a number" >&2; exit 1; }
if [ "$CATE_W" != 0.5 ]; then
    [ "$OBJECTIVE" = new ] || { echo "CATE_W needs OBJECTIVE=new" >&2; exit 1; }
    tag="${tag}cw${CATE_W}"
fi
# STD=context (29 Oct): each training context standardised with its own
# statistics, as scoring standardises with the train slice's (default pool:
# the pool's statistics, every run so far)
STD="${STD:-pool}"
case "$STD" in pool|context) ;; *) echo "STD must be pool or context" >&2; exit 1 ;; esac
if [ "$STD" = context ]; then
    export STANDARDIZE=context; tag="${tag}std"; std_expect="prior.standardize=context"
else
    std_expect="prior.standardize=UNSET"          # the cohort's default, pool
fi
# ALLOC=g5 (29 Oct): G5's two allocation changes on top of this launch
# (virtual trials P(T=1) = 0.5, theory family without confounding), so that
# they can be combined with OBJECTIVE=new; default g4c, G4c's own
ALLOC="${ALLOC:-g4c}"
case "$ALLOC" in g4c|g5) ;; *) echo "ALLOC must be g4c or g5" >&2; exit 1 ;; esac
[ "$ALLOC" = g4c ] || tag="${tag}g5"
# GENERIC_W (1 Nov): share of the processes given a generic effect in latent
# space (generic_mix.py); 0.5 is G4c's and the final model's, 0 leaves the
# anatomy families alone, 1 keeps only their covariates. Any other value is
# tagged in the run name (..._gw0)
GENERIC_W="${GENERIC_W:-0.5}"
[[ "$GENERIC_W" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]] || { echo "GENERIC_W must be a number in [0, 1]" >&2; exit 1; }
[ "$GENERIC_W" = 0.5 ] || tag="${tag}_gw$GENERIC_W"
# HEAD_RANGE=wide (1 Nov, OBJECTIVE=old only): the single-arm objective with the
# outcome bins of OBJECTIVE=new (-0.05 to 1.05), so that the bin range is
# separated from the objective; default unit, G4c's 0 to 1
HEAD_RANGE="${HEAD_RANGE:-unit}"
case "$HEAD_RANGE" in unit|wide) ;; *) echo "HEAD_RANGE must be unit or wide" >&2; exit 1 ;; esac
if [ "$HEAD_RANGE" = wide ]; then
    [ "$OBJECTIVE" = old ] || { echo "HEAD_RANGE=wide needs OBJECTIVE=old (OBJECTIVE=new already uses it)" >&2; exit 1; }
    tag="${tag}hw"
fi
# S_SOURCE (1 Nov): the split whose two network halves give the anatomy
# family's susceptibilities S_A / S_B; genetics is every run so far (the halves
# of the scored genetics trials), receptor the control that swaps the split
S_SOURCE="${S_SOURCE:-genetics}"
case "$S_SOURCE" in genetics|receptor) ;; *) echo "S_SOURCE must be genetics or receptor" >&2; exit 1 ;; esac
[ "$S_SOURCE" = genetics ] || tag="${tag}_s$S_SOURCE"
# FAMILY_W (1 Nov): the (theory, virtual-trial) family weights, before the
# generic share is applied to all processes; "0.5 0.5" is every run so far
FAMILY_W="${FAMILY_W:-0.5 0.5}"
[[ "$FAMILY_W" =~ ^[0-9]+(\.[0-9]+)?\ [0-9]+(\.[0-9]+)?$ ]] || { echo "FAMILY_W must be two numbers, e.g. \"0.25 0.75\"" >&2; exit 1; }
read -r fw_theory fw_trial <<< "$FAMILY_W"
[ "$FAMILY_W" = "0.5 0.5" ] || tag="${tag}_fw${fw_theory}-${fw_trial}"
[ "$RUN_ITERS" = 20000 ] || tag="${tag}_i$RUN_ITERS"
[ "$RUN_ARCH" = linear ] || tag="${tag}_tabicl"
[ -z "${RUN_D_MODEL:-}" ] || tag="${tag}_d$RUN_D_MODEL"
[ -z "${RUN_N_LAYERS:-}" ] || tag="${tag}_L$RUN_N_LAYERS"
[ -z "${RUN_N_BINS:-}" ] || tag="${tag}_b$RUN_N_BINS"
[ -z "${RUN_BATCH:-}" ] || tag="${tag}_B$RUN_BATCH"
[ "$RUN_CONTEXT_MAX" = 3000 ] || tag="${tag}_c$RUN_CONTEXT_MAX"
[ "$RUN_ATTN" = masked ] || tag="${tag}_$RUN_ATTN"
# CACHE (2 Nov): one pool, or several separated by spaces (a multi-encoder
# prior, prior/multi_cache.py: one pool drawn per batch, uniformly), e.g. the
# KCH pool encoded by each of the ten fold encoders; several are tagged
CACHE="${CACHE:-$HOME/p2_bcb/pools/kch_s0/fold0.npz}"
read -r -a caches <<< "$CACHE"
[ "${#caches[@]}" -le 1 ] || tag="${tag}_pools${#caches[@]}"
export SEED="${SEED:-0}"
export JOBNAME="${JOBNAME:-$tag-f0-s$SEED}"
export CACHE
export OUT="${OUT:-$HOME/p2_bcb/pfn_screen/${tag}_s$SEED/fold0}"
if [ "$OBJECTIVE" = new ]; then
    export BOTH_ARMS=1 CATE_WEIGHT="$CATE_W" HEAD_LO=-0.05 HEAD_HI=1.05
    objective_expect="pfn.both_arms=True pfn.cate_weight=$CATE_W pfn.head_lo=-0.05 pfn.head_hi=1.05"
elif [ "$HEAD_RANGE" = wide ]; then
    export HEAD_LO=-0.05 HEAD_HI=1.05
    objective_expect="pfn.both_arms=UNSET pfn.head_lo=-0.05 pfn.head_hi=1.05"
else
    objective_expect="pfn.both_arms=UNSET pfn.head_lo=UNSET"
fi
export FAMILY_WEIGHTS="$FAMILY_W" ARCH="$RUN_ARCH" FAMILY=mixture
export GILES_HYPER="random_partitions=1 threshold=0.03:0.08 p_te=0.9:1 p_re=0:0.05 label_noise=0:0"
export PRIOR_HYPER="two_arms=1 effect_scale=0.2:1 d_source=network s_source=${S_SOURCE}_1 s_b_source=${S_SOURCE}_2 effect_headroom=1 generic_weight=$GENERIC_W binary_outcome=1 generic_binary=1"
if [ "$ALLOC" = g5 ]; then
    GILES_HYPER="$GILES_HYPER bias=0:0 bias_weights=1:0:0"; PRIOR_HYPER="$PRIOR_HYPER gamma=0:0"
fi
export LR_SCHEDULE=constant CONTEXT_POOL=susceptible CONTEXT_MAX="$RUN_CONTEXT_MAX" OMP_NUM_THREADS=4
for c in "${caches[@]}"; do [ -s "$c" ] || { echo "no cache $c" >&2; exit 1; }; done
[ -d data/atlases ] || { echo "no atlas at data/atlases" >&2; exit 1; }
# ATLAS_SRC (1 Nov): the atlas folder the run must use, e.g. the one scoring
# reads (~/export_bcb/atlases on siena); data/atlases must match it file by file
echo "atlas_dir=$(readlink -f data/atlases) caches=${#caches[@]}: $(for c in "${caches[@]}"; do readlink -f "$c"; done | tr '\n' ' ')"
if [ -n "${ATLAS_SRC:-}" ]; then
    diff -rq data/atlases "$ATLAS_SRC" >/dev/null || { echo "data/atlases differs from $ATLAS_SRC" >&2; exit 1; }
    echo "atlas_src=$(readlink -f "$ATLAS_SRC") identical"
fi
mkdir -p "$OUT"
attn_expect=$([ "$RUN_ATTN" = masked ] && echo UNSET || echo "$RUN_ATTN")
# the generic share replaces the effect of that share of ALL processes after the
# family draw, so the effective composition is theory w_t(1-g), trial w_v(1-g), generic g
head_bins=$([ "$OBJECTIVE" = new ] || [ "$HEAD_RANGE" = wide ] && echo "-0.05:1.05" || echo "0:1")
echo "launch tag=$tag objective=$OBJECTIVE cate_w=$CATE_W head=$head_bins generic_w=$GENERIC_W" \
     "s_source=$S_SOURCE family_w=$fw_theory:$fw_trial iters=$RUN_ITERS seed=$SEED" \
     "d_model=${RUN_D_MODEL:-256} n_layers=${RUN_N_LAYERS:-6} n_bins=${RUN_N_BINS:-256}" \
     "batch=${RUN_BATCH:-4} context_max=$RUN_CONTEXT_MAX attn=$RUN_ATTN"
awk -v t="$fw_theory" -v v="$fw_trial" -v g="$GENERIC_W" 'BEGIN { s = t + v;
    printf "effective composition: theory %.3f, virtual-trial %.3f, generic %.3f\n", t / s * (1 - g), v / s * (1 - g), g }'
# the launch must be G4c's: the shared values of G5 / TM and, unless
# OBJECTIVE=new, no 28 Oct switch (the gate's ranges are opened: G4c's prior
# is known to fail them, the run is its control)
# shellcheck disable=SC2086
python scripts/prior_gate.py --batches 5 --contexts 512 --require share_zero=0:1 share_unit=0:1 share_small=0:1 \
    pos_share=0:1 ctx_binary=0:1 ctx_treated=0:1 \
    --expect prior.family=mixture prior.family_weights=$fw_theory:$fw_trial prior.hyper.generic_weight=$GENERIC_W \
             prior.hyper.s_source=${S_SOURCE}_1 prior.hyper.s_b_source=${S_SOURCE}_2 \
             prior.hyper_giles.random_partitions=1 pfn.context_max=$RUN_CONTEXT_MAX pfn.context_min=256 \
             pfn.iters=$RUN_ITERS pfn.arch=$RUN_ARCH pfn.n_query=64 pfn.curriculum=UNSET $objective_expect $std_expect \
             pfn.n_bins=${RUN_N_BINS:-256} pfn.batch_size=${RUN_BATCH:-4} pfn.attn=$attn_expect
bash scripts/train_pfn_myriad.qsub.sh
