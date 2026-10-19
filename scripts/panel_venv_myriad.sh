#!/bin/bash
# A second venv on Myriad for the tabular foundation-model panel (TabPFN,
# TabICL): tabpfn needs torch >= 2.5 while the project venv pins 2.3.1.
# Login node, pip only (no compute). Binary wheels only: Myriad's glibc is
# 2.17, so anything without a manylinux2014 wheel (recent scipy, lightgbm)
# would try to compile and fail; the pins below all ship such wheels for
# Python 3.11. lightgbm is skipped: tabpfn imports it lazily, in a
# preprocessing option the panel does not use.
#
#   bash scripts/panel_venv_myriad.sh            # create or repair ~/venvs/panel
#   bash scripts/panel_venv_myriad.sh check      # import check only
set -euo pipefail
# "module" is a shell function of login shells; a plain "bash script" lacks it
if ! command -v module >/dev/null 2>&1; then
    for f in /etc/profile.d/modules.sh "${MODULESHOME:-/usr/share/Modules}/init/bash"; do
        [ -r "$f" ] && { set +u; source "$f"; set -u; break; }
    done
fi
module load python3/3.11
VENV="${VENV:-$HOME/venvs/panel}"
MAIN="${MAIN:-$HOME/Scratch/neuro-causal-pfn}"
PERFOLD="${PERFOLD:-$HOME/Scratch/neuro-causal-pfn-perfold}"

if [ "${1:-}" != "check" ]; then
    [ -d "$VENV" ] || python3 -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install -q -U pip
    pip install -q --only-binary :all: "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu118
    pip install -q --only-binary :all: "numpy==1.26.4" "scipy==1.11.4" "scikit-learn==1.4.2" \
        "pandas==2.2.2" "nibabel>=5.2" "einops>=0.8" safetensors "typing_extensions>=4.12" \
        huggingface-hub "pydantic>=2.8" "pydantic-settings>=2.10.1" joblib tqdm filelock psutil \
        "skrub<0.11" "matplotlib<3.10" pytest
    pip install -q --no-deps "tabpfn==9.2.0" "tabicl==2.2.0"
    # the project and the per-fold package, without their pinned dependencies
    pip install -q --no-deps -e "$MAIN" -e "$PERFOLD"
else
    source "$VENV/bin/activate"
fi

python - <<'EOF'
import numpy, scipy, sklearn, torch, tabpfn, tabicl, neurocausalpfn, ncpfold
print(f"torch {torch.__version__} cuda {torch.version.cuda} available {torch.cuda.is_available()}")
print(f"numpy {numpy.__version__} scipy {scipy.__version__} sklearn {sklearn.__version__} "
      f"tabpfn {tabpfn.__version__} tabicl {tabicl.__version__}")
from tabpfn import TabPFNClassifier
from tabicl import TabICLClassifier
from neurocausalpfn.pfn.panel import PANEL
print("imports ok:", sorted(PANEL))
EOF
