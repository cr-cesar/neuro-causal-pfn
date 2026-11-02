#!/bin/bash
# The final model retrained on the KCH pool encoded by E13a (2 Nov), on siena.
# The pool is kch_s0/fold0.npz's recipe with E13a seed 0's fold-0 encoder in
# place of E1's; the launch is fw01's (final model without the theory family,
# same budget, 3 PFN seeds), so the only change against fw01 is the encoder of
# the pool. Validation is scored with E13a seed 0's latents, and fw01 is
# rescored on the same latents, so that:
#   (a) e13pool vs fw01, both on E13a latents: what aligning the prior with the
#       encoder adds;
#   (b) e13pool on E13a vs finalK on E1: the change of encoder and prior together.
# Steps (STEPS="1 2 3 4 5" by default, each skips what already exists):
#   1 copy the fold-0 checkpoint from beachcomber (one file, md5 checked; needs
#     Gladys's authorisation of the transfer);
#   2 KCH latents with that encoder (GPU, minutes);
#   3 the pool, checked against kch_s0/fold0.npz (same rows and keys, Z aside);
#   4 three trainings (GPU) and validation scoring (CPU, as finalK);
#   5 fw01 rescored on E13a latents (CPU) and the paired tests.
# Aggregates only in the logs: counts, md5 prefixes, PEHE lines.
#
#   bash scripts/audit_20261028/e13pool_launch.sh
#   STEPS="1 2 3" bash scripts/audit_20261028/e13pool_launch.sh
set -euo pipefail
STEPS="${STEPS:-1 2 3 4 5}"
PY="${PY:-$HOME/env/bin/python}"
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
CODE="${CODE:-$HOME/ncp_full}"   # fw01's checkout, so the training code is fw01's
W="${W:-$HOME/p2_bcb}"
E="${E:-$W/e13}"
KCH="${KCH:-$HOME/kch_bcbdisco}"
REPS="${REPS:-$E/E13a/E13a/seed0/folds}"
CK_DIR="$E/ckpt/E13a_s0_f0"
LAT="$W/latents/kch_e13a_s0/fold0"
POOL="$W/pools/kch_e13a_s0/fold0.npz"
REF_POOL="${REF_POOL:-$W/pools/kch_s0/fold0.npz}"
O="$W/pfn_e13pool"
V="${V:-$HOME/neuro-causal-pfn-perfold/outputs_val_clean}"
SCORE="$HERE/scripts/audit_20261028/score_val_clean.sh"
BC_JUMP="${BC_JUMP:-ccornejo@csrw.cs.ucl.ac.uk}"
BC_HOST="${BC_HOST:-ccornejo@beachcomber.cs.ucl.ac.uk}"
BC_CK="${BC_CK:-/scratch0/ccornejo/outputs_perfold_e13/E13a/E13a/seed0/fold0/disco/vae_disconnectome.pt}"
mkdir -p "$CK_DIR" "$O/logs" "$O/paired"
rr() { echo "$V/$1/$2/replica/$3/disconnectome-extrasclean/replica_results.csv"; }

if [[ " $STEPS " == *" 1 "* ]]; then
    echo "== $(date +%H:%M) 1 fold-0 encoder of E13a seed 0"
    if [ ! -s "$CK_DIR/vae_disconnectome.pt" ]; then
        # beachcomber's login shell is csh: the remote command goes on stdin
        echo "md5sum $BC_CK" | ssh -J "$BC_JUMP" "$BC_HOST" bash -s | cut -c1-32 > "$CK_DIR/remote.md5"
        echo "cat $BC_CK" | ssh -J "$BC_JUMP" "$BC_HOST" bash -s > "$CK_DIR/vae_disconnectome.pt.part"
        mv "$CK_DIR/vae_disconnectome.pt.part" "$CK_DIR/vae_disconnectome.pt"
    fi
    here_md5=$(md5sum "$CK_DIR/vae_disconnectome.pt" | cut -c1-32)
    [ "$here_md5" = "$(cat "$CK_DIR/remote.md5")" ] || { echo "md5 mismatch on the checkpoint" >&2; exit 1; }
    echo "   1 file, $(du -h "$CK_DIR/vae_disconnectome.pt" | cut -f1), md5 ${here_md5:0:12} matches"
fi

if [[ " $STEPS " == *" 2 "* ]]; then
    echo "== $(date +%H:%M) 2 KCH latents with that encoder"
    if ! ls "$LAT"/*.npz >/dev/null 2>&1; then
        (nice -n 5 "$PY" "$HERE/scripts/export_latents.py" --checkpoints "$CK_DIR/vae_disconnectome.pt" \
            --images-dir "$KCH" --out "$LAT" --device cuda) > "$O/logs/latents.log" 2>&1
    fi
    echo "   $(ls "$LAT"/*.npz | wc -l) latents file"
fi

if [[ " $STEPS " == *" 3 "* ]]; then
    echo "== $(date +%H:%M) 3 pool"
    [ -s "$POOL" ] || "$PY" "$HERE/scripts/build_fold_pool_cache.py" --cache "$W/caches/uclh_bcb.npz" --fold-dir "$REPS" \
        --folds 0 --out-dir "$(dirname "$POOL")" --extra-cache "$W/caches/kch_bcb.npz" \
        --extra-latents "$LAT/*.npz" --extra-only
    "$PY" - "$POOL" "$REF_POOL" <<'EOF'
import sys
import numpy as np
new, ref = (dict(np.load(p, allow_pickle=True)) for p in sys.argv[1:3])
skip = {"Z", "fold_file", "latents_source"}
keys = sorted(set(new) | set(ref))
bad = [k for k in keys if k not in skip and (k not in new or k not in ref
       or np.asarray(new[k]).shape != np.asarray(ref[k]).shape
       or (np.asarray(new[k]).dtype.kind in "fiub" and not np.array_equal(new[k], ref[k])))]
Z = np.asarray(new["Z"])
print(f"   rows {len(Z)} (reference {len(ref['Z'])}), latent dims {Z.shape[1]}, finite {bool(np.isfinite(Z).all())}, "
      f"keys differing from the reference besides Z: {len(bad)}")
sys.exit(1 if bad or Z.shape != np.asarray(ref["Z"]).shape or not np.isfinite(Z).all() else 0)
EOF
fi

if [[ " $STEPS " == *" 4 "* ]]; then
    echo "== $(date +%H:%M) 4 three trainings, fw01's launch on the E13a pool"
    for s in 0 1 2; do
        [ -s "$O/s$s/pfn.pt" ] && continue
        (cd "$CODE" && env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VENV="$(dirname "$(dirname "$PY")")" \
            OBJECTIVE=new FAMILY_W="0 1" ATLAS_SRC="$HOME/export_bcb/atlases" SEED=$s RUN_ITERS=125000 \
            JOBNAME=finalK-e13pool-lin4p8M-s$s CACHE="$POOL" OUT="$O/s$s" \
            bash scripts/audit_20261028/g4c_launch.sh) > "$O/logs/train_s$s.log" 2>&1 &
    done
    wait
    for s in 0 1 2; do
        [ -s "$(rr "finalKe13pool_lin4p8M_s$s" receptor E13a/E13a/seed0)" ] && continue
        CUDA_VISIBLE_DEVICES="" REPS="$REPS" bash "$SCORE" "finalKe13pool_lin4p8M_s$s" \
            --estimators "$O/s$s/pfn.pt" --only-estimators &
    done
    wait
fi

if [[ " $STEPS " == *" 5 "* ]]; then
    echo "== $(date +%H:%M) 5 fw01 on E13a latents, paired tests"
    for s in 0 1 2; do
        [ -s "$(rr "e13a_finalKfw01_s$s" receptor E13a/E13a/seed0)" ] && continue
        CUDA_VISIBLE_DEVICES="" REPS="$REPS" bash "$SCORE" "e13a_finalKfw01_s$s" \
            --estimators "$W/pfn_finalK_gw/fw01/s$s/pfn.pt" --only-estimators &
    done
    wait
    pair() {  # tag, A label glob, A replica, B label glob, B replica
        for m in receptor genetics; do
            "$PY" "$HERE/scripts/paired_deficit_test.py" --a "$(rr "$2" $m "$3")" --b "$(rr "$4" $m "$5")" \
                --label-a e13pool --label-b "${1#*_vs_}" --out "$O/paired/${1}_${m}_pehe_paper.csv" | sed -n "1,3p"
        done
        "$PY" "$HERE/scripts/paired_split_mean.py" --receptor "$O/paired/${1}_receptor_pehe_paper_per_deficit.csv" \
            --genetics "$O/paired/${1}_genetics_pehe_paper_per_deficit.csv" --label-a e13pool --label-b "${1#*_vs_}" \
            --out "$O/paired/${1}_both_pehe_paper.csv" | sed -n "1,3p"
    }
    pair e13pool_vs_fw01 "finalKe13pool_lin4p8M_s*" E13a/E13a/seed0 "e13a_finalKfw01_s*" E13a/E13a/seed0
    pair e13pool_vs_finalK "finalKe13pool_lin4p8M_s*" E13a/E13a/seed0 "finalK_lin4p8M_s*" E1/E1/seed0
    echo "== md5"
    md5sum "$O"/paired/*.csv | cut -c1-12 | tr '\n' ' '; echo
fi
