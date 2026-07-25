#!/usr/bin/env bash
# Portable runtime flags for laptops (macOS / CPU). Source this before running
# anything, so the flags also reach subprocesses and FAISS:
#
#     source scripts/portable_env.sh
#
# On the cluster you do NOT need this; utils/runtime.py handles the V100 path.
# Each variable is only exported if unset, so an explicit setting of yours wins.

# Apple and PyTorch/MKL each ship an OpenMP runtime; loading both aborts
# ("OMP: Error #15") unless this is set.
: "${KMP_DUPLICATE_LIB_OK:=TRUE}"; export KMP_DUPLICATE_LIB_OK

# One OpenMP thread avoids oversubscription/deadlocks with FAISS on macOS and
# keeps CPU runs deterministic. Raise on a large CPU box if you want throughput.
: "${OMP_NUM_THREADS:=1}"; export OMP_NUM_THREADS
: "${MKL_NUM_THREADS:=1}"; export MKL_NUM_THREADS

# Let unsupported 3D ops fall back to CPU if MPS is ever selected, so a run
# completes (slowly) instead of crashing.
: "${PYTORCH_ENABLE_MPS_FALLBACK:=1}"; export PYTORCH_ENABLE_MPS_FALLBACK

echo "portable_env: KMP_DUPLICATE_LIB_OK=$KMP_DUPLICATE_LIB_OK" \
     "OMP_NUM_THREADS=$OMP_NUM_THREADS" \
     "PYTORCH_ENABLE_MPS_FALLBACK=$PYTORCH_ENABLE_MPS_FALLBACK"
