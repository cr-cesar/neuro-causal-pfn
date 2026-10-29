#!/bin/bash
# Route A per fold (29 Oct): TabPFN fine-tuned on the Neuro-Prior of each
# fold's pool (UCLH + KCH without fold k's test patients), one checkpoint per
# fold, so checkpoint k scores fold k only (panel.PerFoldTabPFN, {fold} in
# TABPFN_MODEL_PATH). Prior and steps default to the 27 Oct route-A run
# (thr03 virtual trials mixed 0.5 / 0.5 with the theory family, generic 0.5,
# 2,000 steps, lr 1e-5, context 1,500, 1 estimator); check them against that
# run's history.json before launching.
#
#   CACHE='<pool dir>/fold{fold}.npz' OUT=<root> [FOLDS="0 1"] [STEPS=2000] [N_CONTEXT=1500] \
#       [N_EST=1] [SEED=0] [GILES_HYPER=...] [PRIOR_HYPER=...] bash scripts/audit_20261028/tabpfn_ft_perfold.sh
#
# writes <OUT>/fold<k>/{tabpfn_ft.pth,history.json,train.log,launch.txt}; a
# fold whose checkpoint exists is skipped, so a rerun only does what is missing.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
# (no literal braces inside the :? message: the first } would close the expansion)
CACHE="${CACHE:?set CACHE to the per-fold anatomy cache path, with the fold placeholder}"
OUT="${OUT:?set OUT to the output root}"
[[ "$CACHE" == *"{fold}"* ]] || { echo "CACHE needs {fold} in the path" >&2; exit 1; }
GILES_HYPER="${GILES_HYPER-random_partitions=1 threshold=0.03:0.08 p_te=0.5:1}"
PRIOR_HYPER="${PRIOR_HYPER-two_arms=1 effect_scale=0.2:1 d_source=network s_source=genetics_1 s_b_source=genetics_2 effect_headroom=1 generic_weight=0.5}"

for k in ${FOLDS:-0 1 2 3 4 5 6 7 8 9}; do
  c="${CACHE//\{fold\}/$k}"; o="$OUT/fold$k"
  if [ -s "$o/tabpfn_ft.pth" ] && [ -s "$o/launch.txt" ] && grep -q "^finished" "$o/launch.txt"; then
    echo "fold $k: checkpoint exists, skipped"; continue
  fi
  [ -s "$c" ] || { echo "fold $k: no cache $c" >&2; exit 1; }
  mkdir -p "$o"
  { echo "commit $(git -C "$REPO" rev-parse HEAD)"
    echo "dirty_files $(git -C "$REPO" status --porcelain | wc -l)"
    echo "cache $(basename "$(dirname "$c")")/$(basename "$c")"
    echo "giles_hyper $GILES_HYPER"; echo "prior_hyper $PRIOR_HYPER"
    echo "steps ${STEPS:-2000} n_context ${N_CONTEXT:-1500} n_estimators ${N_EST:-1} seed ${SEED:-0}"; } > "$o/launch.txt"
  python "$REPO/scripts/finetune_tabpfn_prior.py" --cache "$c" --family mixture --family-weights 0.5 0.5 \
      --giles-hyper "$GILES_HYPER" --prior-hyper "$PRIOR_HYPER" \
      --steps "${STEPS:-2000}" --n-context "${N_CONTEXT:-1500}" --n-estimators "${N_EST:-1}" \
      --seed "${SEED:-0}" --atlas-dir "${ATLAS_DIR:-data/atlases}" --out "$o" 2>&1 | tee "$o/train.log"
  echo "finished $(date -u +%FT%TZ)" >> "$o/launch.txt"
done
