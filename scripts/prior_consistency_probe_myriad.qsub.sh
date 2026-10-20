#!/bin/bash -l
# Audit of 19 Oct: score held-out processes of a checkpoint's own prior under
# the scoring path's transformations (scripts/prior_consistency_probe.py).
# CPU, minutes per checkpoint.
#   qsub -v CKPTS="outputs/pfn_reduced_<run>/pfn.pt ...",OUT_DIR=status/audit_consistency scripts/prior_consistency_probe_myriad.qsub.sh
#
#$ -N prior-consist
#$ -l h_rt=2:0:0
#$ -pe smp 4
#$ -l mem=8G
#$ -cwd
#$ -j y
set -euo pipefail
trap 'exit 100' ERR

module load python3/3.11
source ~/venvs/neuro/bin/activate

CKPTS="${CKPTS:?set CKPTS to one or more pfn.pt paths}"
OUT_DIR="${OUT_DIR:-outputs/prior_consistency}"
for ck in $CKPTS; do
    [ -s "$ck" ] || { echo "no checkpoint $ck"; exit 100; }
    run=$(basename "$(dirname "$ck")")
    python scripts/prior_consistency_probe.py --ckpt "$ck" --processes "${PROCESSES:-40}" \
        --n-context "${N_CONTEXT:-1024}" --jobs "${NSLOTS:-4}" --out "$OUT_DIR/${run}.csv"
done
