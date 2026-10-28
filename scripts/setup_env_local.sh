#!/bin/bash
# Python environment for training the Neuro-Causal-PFN on a machine without a
# module system (a lab workstation with one NVIDIA GPU, driver for CUDA 11.8
# or newer). Creates .venv in the repository, installs torch 2.3.1 for CUDA
# 11.8 from the PyTorch index and the project in editable mode. Needs
# Python 3.10 or 3.11 on PATH as python3 (or set PYTHON).
#
#   bash scripts/setup_env_local.sh            # install from the internet
#   WHEELS=bundle/wheels bash scripts/setup_env_local.sh   # install offline from the bundle's wheels
#   EXISTING_VENV=~/env bash scripts/setup_env_local.sh     # torch already there (e.g. RTX 5090: torch >= 2.7); adds the rest
#   bash scripts/setup_env_local.sh check      # import and GPU check only
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
VENV="${VENV:-$PWD/.venv}"
PIP() {   # pip of the environment, or uv when the environment was made by uv (no pip inside);
          # uv takes --python after the subcommand: uv pip install --python <py> ...
    if [ -x "$VENV/bin/pip" ]; then "$VENV/bin/pip" "$@"; else uv pip "$1" --python "$VENV/bin/python" "${@:2}"; fi
}
if [ "${1:-}" != "check" ]; then
    if [ -n "${EXISTING_VENV:-}" ]; then
        # an environment that already holds a working torch for this GPU (a
        # Blackwell card needs torch >= 2.7, newer than the pinned 2.3.1):
        # add the rest of the requirements and the project, torch untouched
        VENV="$EXISTING_VENV"
        [ -x "$VENV/bin/python" ] || { echo "no Python at $EXISTING_VENV/bin/python"; exit 2; }
        "$VENV/bin/python" -c 'import torch; assert torch.cuda.is_available(), "torch without CUDA"' \
            || { echo "torch with CUDA must already be installed in $EXISTING_VENV"; exit 2; }
        grep -v -E '^-e|^--extra|^torch|^\s*#|^\s*$' requirements.txt > /tmp/req_notorch_$$.txt
        OFF=(); [ -z "${WHEELS:-}" ] || OFF=(--no-index --find-links "$WHEELS")
        PIP install -q "${OFF[@]}" setuptools wheel
        PIP install -q "${OFF[@]}" -r /tmp/req_notorch_$$.txt
        PIP install -q "${OFF[@]}" --no-build-isolation --no-deps -e .
        rm -f /tmp/req_notorch_$$.txt
        echo "$VENV" > .venv_path
    else
        "$PYTHON" -c 'import sys; assert sys.version_info[:2] in ((3, 10), (3, 11)), sys.version' \
            || { echo "need Python 3.10 or 3.11 (set PYTHON=/path/to/python3.11), or EXISTING_VENV with a torch for this GPU"; exit 2; }
        [ -d "$VENV" ] || "$PYTHON" -m venv "$VENV"
        source "$VENV/bin/activate"
        if [ -n "${WHEELS:-}" ]; then
            # offline: every wheel comes from the bundle (scripts/make_bundle.sh FULL=1)
            pip install -q --no-index --find-links "$WHEELS" -U pip setuptools wheel
            pip install -q --no-index --find-links "$WHEELS" --no-build-isolation -r requirements.txt
        else
            pip install -q -U pip
            pip install -q -r requirements.txt
        fi
    fi
else
    [ -n "${EXISTING_VENV:-}" ] && VENV="$EXISTING_VENV"
    [ -f .venv_path ] && [ ! -x "$VENV/bin/python" ] && VENV="$(cat .venv_path)"
fi
export PATH="$VENV/bin:$PATH"
python - <<'EOF'
import torch, numpy, scipy, sklearn, neurocausalpfn
print(f"torch {torch.__version__} cuda {torch.version.cuda} available {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"gpu {p.name}, {p.total_memory / 2**30:.0f} GB")
print(f"numpy {numpy.__version__} scipy {scipy.__version__} sklearn {sklearn.__version__}; neurocausalpfn ok")
EOF
