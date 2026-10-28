#!/bin/bash -l
# One of the 28 Oct launchers (tm_launch.sh, g4c_launch.sh, g5_launch.sh) as
# a Myriad GPU job. LAUNCHER names it; SEED, CACHE and OUT pass through.
# Submit from the root of a clean checkout of the commit to train:
#
#   qsub -N tm-s1 -v LAUNCHER=tm_launch.sh,SEED=1,CACHE=$HOME/Scratch/audit_tmp/g3/g3/cache_kch_s0/fold0.npz,OUT=$HOME/Scratch/audit_tmp/tm/TM_s1/fold0 \
#        scripts/audit_20261028/launch_myriad.qsub.sh
#
#$ -N ncp-launch
#$ -l h_rt=36:0:0
#$ -l gpu=1
#$ -pe smp 8
#$ -l mem=8G
#$ -l tmpfs=10G
#$ -cwd
#$ -j y
set -euo pipefail
trap 'exit 100' ERR
LAUNCHER="${LAUNCHER:?set LAUNCHER to tm_launch.sh, g4c_launch.sh or g5_launch.sh}"
case "$LAUNCHER" in tm_launch.sh|g4c_launch.sh|g5_launch.sh) ;; *) echo "unknown launcher $LAUNCHER" >&2; exit 100 ;; esac
export SEED="${SEED:-0}"
export CACHE="${CACHE:?set CACHE}" OUT="${OUT:?set OUT}"
bash "scripts/audit_20261028/$LAUNCHER"
