#!/bin/bash -l
# First real Neuro-Causal-PFN training (reduced configuration, one GPU).
#
#   VARIANT=nocurr|ctx|ctx+stages selects an E12 curriculum variant (scripts/myriad_phase1b.sh e12).
#   FAMILY=theory|giles|mixture selects the Neuro-Prior family (empty = theory, the pilot's);
#   FAMILY_WEIGHTS="0.5 0.5" sets the mixture's theory / giles weights (space-separated).
#   CONTEXT_POOL=susceptible|all: where a process draws its context (default susceptible; the pilot used all).
#   qsub -v CACHE=outputs/prior_cache/kch_E1_seed0.npz,OUT=outputs/pfn_reduced_kch_E1 \
#        scripts/train_pfn_myriad.qsub.sh
#   ITERS=20000 (default), SEED=0, ARCH=tabicl|linear
#
# Then score it on the Phase 1 folds (CPU):
#   qsub -v LATENTS="outputs/latents_E1/E1_seed0_disco.npz",GROUP_TABLE=outputs/groups_public.csv,\
#        PFN_CKPT=outputs/pfn_reduced_kch_E1/pfn.pt,OUT=outputs/giles_replica_pfn scripts/run_giles_replica_myriad.qsub.sh
#
#$ -N ncp-pfn
#$ -l h_rt=36:0:0
#$ -l gpu=1
#$ -pe smp 8
#$ -l mem=8G
#$ -l tmpfs=10G
#$ -cwd
#$ -j y
set -euo pipefail
# a failed step exits 100: SGE then keeps this job in error state and the jobs
# holding on it (-hold_jid) stay queued instead of starting on missing inputs
trap 'exit 100' ERR

# outside Myriad (scripts/run_jobs.sh on another machine) there is no module
# system; VENV names the environment to use
command -v module >/dev/null 2>&1 && module load python3/3.11
source "${VENV:-$HOME/venvs/neuro}/bin/activate"
# audit F25: the same config and seed ran 2.4x slower on another node, so
# every job states where it ran
echo "node: $(hostname -s) gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo none) job: ${JOB_ID:-?}"

CACHE="${CACHE:?set CACHE to the npz from scripts/build_prior_cache.py}"
# several caches, space-separated, are a multi-encoder pool (prior/multi_cache.py):
# one cohort per cache, drawn per batch; the first is the anchor and keeps the
# CACHE_ANCHOR_WEIGHT share of the batches (default: uniform)
for c in $CACHE; do [ -s "$c" ] || { echo "no prior cache $c (the cache job failed?)" >&2; exit 100; }; done
CACHE_ANCHOR_WEIGHT="${CACHE_ANCHOR_WEIGHT:-}"
OUT="${OUT:-outputs/pfn_reduced}"
ITERS="${ITERS:-20000}"
SEED="${SEED:-0}"
ARCH="${ARCH:-tabicl}"
VARIANT="${VARIANT:-}"      # E12: nocurr | ctx | ctx+stages (empty = the pilot's reduced config)
FAMILY="${FAMILY:-}"        # theory | giles | mixture (empty = theory)
FAMILY_WEIGHTS="${FAMILY_WEIGHTS:-}"
CONTEXT_POOL="${CONTEXT_POOL:-susceptible}"
PRIOR_HYPER="${PRIOR_HYPER:-}"
GILES_HYPER="${GILES_HYPER:-}"      # virtual-trial family ranges of the mixture, e.g. "threshold=0.01:0.2 p_te=0.5:1"
LR_SCHEDULE="${LR_SCHEDULE:-constant}"   # constant | cosine (warmup 2 % then cosine decay to 0.1 lr)
BATCH="${BATCH:-}"                        # batch size override (default: the config's)
CONTEXT_MAX="${CONTEXT_MAX:-}"            # training context override, e.g. 3000 (default: the family's cap, 1024 with susceptible sampling)
AUGMENT="${AUGMENT:-}"                    # 0 disables the random rotation of the pool latents (H5 of the Phase-2 audit)
CONTEXT_REPLACE="${CONTEXT_REPLACE:-}"    # 0 draws distinct lesions per context, capped at the slice (audit of 19 Oct, point 1)
STANDARDIZE="${STANDARDIZE:-}"
D_MODEL="${D_MODEL:-}"                    # model width override (reduced config: 256; stage 3 of the Phase-2 programme: 512)
N_LAYERS="${N_LAYERS:-}"                  # depth override (reduced config: 6; 512-wide runs use 12)            # context: standardise each training context with its own statistics, as scoring does (point 2)    # theory prior extensions, e.g. "two_arms=1 effect_scale=0.2:1 binary_outcome=0.5"

python - <<PY
import json
from neurocausalpfn.train.train_pfn import e12_config, parse_hyper_opts, reduced_config, run_pfn, with_family
fam = "$FAMILY" or "theory"
w = [float(x) for x in "$FAMILY_WEIGHTS".split()] or None
cp = "$CONTEXT_POOL"
cfg = e12_config("$VARIANT", fam, w, cp) if "$VARIANT" else with_family(reduced_config(), fam, w, cp)
cfg["out_dir"] = "$OUT"; cfg["seed"] = int("$SEED")
cfg["pfn"]["iters"] = int("$ITERS"); cfg["pfn"]["arch"] = "$ARCH"
cfg["pfn"]["lr_schedule"] = "$LR_SCHEDULE"
if "$BATCH": cfg["pfn"]["batch_size"] = int("$BATCH")
if "$CONTEXT_MAX": cfg["pfn"]["context_max"] = int("$CONTEXT_MAX")
if "$AUGMENT": cfg["prior"]["augment"] = bool(int("$AUGMENT"))
if "$CONTEXT_REPLACE": cfg["prior"]["context_replace"] = bool(int("$CONTEXT_REPLACE"))
if "$STANDARDIZE": cfg["prior"]["standardize"] = "$STANDARDIZE"
if "$D_MODEL": cfg["pfn"]["d_model"] = int("$D_MODEL")
if "$N_LAYERS": cfg["pfn"]["n_layers"] = int("$N_LAYERS")
caches = "$CACHE".split()
cfg["prior"]["cache"] = caches[0] if len(caches) == 1 else caches
if "$CACHE_ANCHOR_WEIGHT": cfg["prior"]["cache_anchor_weight"] = float("$CACHE_ANCHOR_WEIGHT")
cfg["prior"]["atlas_dir"] = "data/atlases"
cfg["prior"]["hyper"] = {**cfg["prior"].get("hyper", {}), **parse_hyper_opts("$PRIOR_HYPER")}
if "$GILES_HYPER": cfg["prior"]["hyper_giles"] = {**cfg["prior"].get("hyper_giles", {}), **parse_hyper_opts("$GILES_HYPER")}
print(json.dumps(cfg, indent=1, default=str))
model, history = run_pfn(cfg)
json.dump(history, open("$OUT/history.json", "w"))
PY
