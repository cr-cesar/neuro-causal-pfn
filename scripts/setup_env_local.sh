#!/bin/bash
# Python environment for training the Neuro-Causal-PFN on a machine without a
# module system (a lab workstation with one NVIDIA GPU, driver for CUDA 11.8
# or newer). Creates .venv in the repository, installs torch 2.3.1 for CUDA
# 11.8 from the PyTorch index and the project in editable mode. Needs
# Python 3.10 or 3.11 on PATH as python3 (or set PYTHON).
#
#   bash scripts/setup_env_local.sh            # install from the internet
#   WHEELS=bundle/wheels bash scripts/setup_env_local.sh   # install offline from the bundle's wheels
#   bash scripts/setup_env_local.sh check      # import and GPU check only
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
VENV="${VENV:-$PWD/.venv}"
if [ "${1:-}" != "check" ]; then
    "$PYTHON" -c 'import sys; assert sys.version_info[:2] in ((3, 10), (3, 11)), sys.version' \
        || { echo "need Python 3.10 or 3.11 (set PYTHON=/path/to/python3.11)"; exit 2; }
    [ -d "$VENV" ] || "$PYTHON" -m venv "$VENV"
    source "$VENV/bin/activate"
    if [ -n "${WHEELS:-}" ]; then
        # offline: every wheel comes from the bundle (scripts/make_bundle.sh FULL=1)
        pip install -q --no-index --find-links "$WHEELS" -U pip setuptools wheel
        # --no-build-isolation: the editable install of the project builds with
        # the setuptools just installed instead of downloading one
        pip install -q --no-index --find-links "$WHEELS" --no-build-isolation -r requirements.txt
    else
        pip install -q -U pip
        pip install -q -r requirements.txt
    fi
else
    source "$VENV/bin/activate"
fi
python - <<'EOF'
import torch, numpy, scipy, sklearn, neurocausalpfn
print(f"torch {torch.__version__} cuda {torch.version.cuda} available {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"gpu {p.name}, {p.total_memory / 2**30:.0f} GB")
print(f"numpy {numpy.__version__} scipy {scipy.__version__} sklearn {sklearn.__version__}; neurocausalpfn ok")
EOF
