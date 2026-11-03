#!/bin/bash -l
# CPU check before a Myriad training of launch_myriad.qsub.sh (2 Nov): the
# same launch with PRIOR_CHECK_ONLY=1 (cfg, cache and prior built as the
# training would, prior_checks.py on a few batches, aggregates in
# PRIOR_CHECK_OUT), plus the digests a cross-machine run is matched on: one
# md5 over the atlas files the cache was built from, and the pool's md5.
#
#   qsub -N lno-chk -v LAUNCHER=g4c_launch.sh,CACHE=...,OUT=...,PRIOR_CHECK_OUT=...,EXCLUDE_NETWORKS=1+2+3+4 \
#        scripts/audit_20261028/check_myriad.qsub.sh
#
#$ -N ncp-check
#$ -l h_rt=2:0:0
#$ -pe smp 4
#$ -l mem=8G
#$ -cwd
#$ -j y
set -euo pipefail
trap 'exit 100' ERR
export PRIOR_CHECK_ONLY=1 PRIOR_CHECK_ARGS="${PRIOR_CHECK_ARGS:---batches 5 --contexts 512 3000}"
: "${PRIOR_CHECK_OUT:?set PRIOR_CHECK_OUT}"
if [ ! -e data/atlases ] && [ -n "${ATLAS_LINK:-}" ]; then
    mkdir -p data && ln -s "$ATLAS_LINK" data/atlases
fi
# the parcellation and both splits' halves, in a fixed order, one digest
echo "atlas_md5=$(cd -P data/atlases && find functional_parcellation_2mm.nii.gz 2mm_parcellations -type f | LC_ALL=C sort \
    | xargs -d '\n' md5sum | md5sum | cut -c1-12)"
echo "pool_md5=$(md5sum "${CACHE:?set CACHE}" | cut -c1-12)"
bash scripts/audit_20261028/launch_myriad.qsub.sh
