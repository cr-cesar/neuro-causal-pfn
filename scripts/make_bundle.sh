#!/bin/bash
# Build the bundle another machine needs to train Neuro-Causal-PFNs: the
# prior caches and per-fold pools with hashed file names, the KCH latents of
# the fold encoders (hashed too), the atlases, a jobs list and the
# instructions. No image, no name, no outcome leaves the cluster. Run on a
# Myriad login node (seconds of CPU).
#
#   bash scripts/make_bundle.sh ~/Scratch/bundle_$(date +%Y%m%d)
#
# Result: <dir>/neuro-causal-pfn-bundle.tgz and its size; copy it to the
# other machine and follow docs/RUN_ELSEWHERE.md there.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v module >/dev/null 2>&1 && module load python3/3.11
source "${VENV:-$HOME/venvs/neuro}/bin/activate"
DEST="${1:?usage: bash scripts/make_bundle.sh <destination dir>}"
JOBS="${JOBS:-jobs/mixed_prior_seeds12.txt}"
mkdir -p "$DEST/bundle/outputs" "$DEST/bundle/data" "$DEST/bundle/jobs"
export HASH_SALT_FILE="${HASH_SALT_FILE:-$HOME/.cache_salt}"
python scripts/hash_cache_names.py --dst "$DEST/bundle/outputs/prior_cache" \
    outputs/prior_cache/kch_e1_32d_x.npz outputs/prior_cache/uclh_disco_x.npz
for d in outputs/prior_cache/fold_pools/bcb-e1-s*; do
    [ -d "$d" ] && python scripts/hash_cache_names.py --src "$d" --dst "$DEST/bundle/outputs/prior_cache/fold_pools"
done
python scripts/hash_cache_names.py --src outputs/latents_pf_bcb --dst "$DEST/bundle/outputs"
cp -r data/atlases "$DEST/bundle/data/"
cp "$JOBS" "$DEST/bundle/jobs/"
cp docs/RUN_ELSEWHERE.md scripts/START.sh "$DEST/bundle/"
if [ -n "${FULL:-}" ]; then
    # FULL=1: a machine with no internet at run time also gets the code and
    # every wheel. PY (default 3.11) is the other machine's Python; the wheels
    # are manylinux2014 (glibc 2.17 or newer), CUDA 11.8 build of torch.
    # the code of the development branch (the clone's HEAD may be another branch)
    CODE_REF="${CODE_REF:-origin/claude/phase-1-vae-encoder-f02ugc}"
    git fetch -q origin "${CODE_REF#origin/}"
    git archive --format=tar.gz --prefix=neuro-causal-pfn/ -o "$DEST/bundle/code.tgz" "$CODE_REF"
    PY="${PY:-3.11}"; PYTAG="cp${PY/./}"
    grep -v -E '^-e|^\s*#|^\s*$' requirements.txt > "$DEST/requirements.download.txt"
    pip download -q --only-binary :all: --python-version "$PY" --implementation cp --abi "$PYTAG" \
        --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform linux_x86_64 --platform any \
        -d "$DEST/bundle/wheels" -r "$DEST/requirements.download.txt" pip "setuptools>=61" wheel
    echo "wheels for Python $PY: $(ls "$DEST/bundle/wheels" | wc -l) files, $(du -sh "$DEST/bundle/wheels" | cut -f1)"
fi
# nothing identifiable: no nifti, no csv with names, and a check on the npz keys
if find "$DEST/bundle" -iname '*.nii*' -path '*outputs*' | grep -q .; then echo "image files in the bundle: stop"; exit 2; fi
python - "$DEST/bundle" <<'EOF'
import glob, os, re, sys
import numpy as np
root = sys.argv[1]; bad = 0
for f in glob.glob(os.path.join(root, "outputs", "**", "*.npz"), recursive=True):
    with np.load(f, allow_pickle=True) as z:
        if "files" in z.files:
            names = np.asarray(z["files"]).ravel()
            if any(not re.fullmatch(r"[0-9a-f]{16}", str(n)) for n in names[:50]):
                print("unhashed names in", os.path.relpath(f, root)); bad += 1
print("bundle check:", "clean" if not bad else f"{bad} files with raw names")
sys.exit(1 if bad else 0)
EOF
tar -C "$DEST" -czf "$DEST/neuro-causal-pfn-bundle.tgz" bundle
du -sh "$DEST/neuro-causal-pfn-bundle.tgz"
