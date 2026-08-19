#!/usr/bin/env bash
# Standalone, reproducible environment for the acsid-amd branch.
#
# Builds an isolated Python virtualenv (default: ${PROJECT_ROOT}/.venv-amd)
# with PyTorch for ROCm + the acsid_amd requirements, WITHOUT touching any
# CUDA/NVIDIA env on the same machine. Designed to be idempotent: re-run to
# upgrade-in-place or repair.
#
# Run on the cloud MI300X box (Ubuntu/Debian, ROCm driver present). Do NOT
# run on Windows/MINGW -- torch ROCm wheels are Linux-only.
#
# Usage (from anywhere):
#   bash acsid_amd/setup_env.sh                # default: .venv-amd, rocm6.2
#   VENV_DIR=/opt/acsid-amd ROCM_TAG=rocm6.3.4 \
#       bash acsid_amd/setup_env.sh           # override location / ROCm release
#
# After it finishes:
#   source .venv-amd/bin/activate
#   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "[setup_env] ERROR: this script must run on Linux with a ROCm driver." >&2
    echo "[setup_env]   You are on: $(uname -s)" >&2
    echo "[setup_env] torch ROCm wheels are Linux-only; not usable on Windows/macOS." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="${PROJECT_ROOT}"   # repo contains MiniOneRec/ at top level
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv-amd}"
ROCM_TAG="${ROCM_TAG:-rocm6.2}"
ROCM_INDEX="https://download.pytorch.org/whl/${ROCM_TAG}"
PY_VER="${PY_VER:-python3.10}"

echo "[setup_env] project root : ${PROJECT_ROOT}"
echo "[setup_env] venv dir     : ${VENV_DIR}"
echo "[setup_env] ROCm tag      : ${ROCM_TAG}"
echo "[setup_env] python base   : ${PY_VER}"

# 1) Ensure the venv exists (create lazily; don't blow away an existing one).
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    if ! command -v "${PY_VER}" >/dev/null 2>&1; then
        echo "[setup_env] '${PY_VER}' not found. Install it first, e.g.:" >&2
        echo "[setup_env]   sudo apt-get update && sudo apt-get install -y ${PY_VER}-venv" >&2
        exit 3
    fi
    echo "[setup_env] creating venv at ${VENV_DIR}"
    "${PY_VER}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "[setup_env] upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

# 2) PyTorch for ROCm FIRST, from the dedicated index so pip can't fall back
#    to a CUDA/CPU wheel. torch is deliberately NOT pinned in requirements.txt.
echo "[setup_env] installing torch from ${ROCM_INDEX}"
python -m pip install --no-cache-dir torch \
    --index-url "${ROCM_INDEX}"

# 3) Project requirements (no torch pin; no CUDA/nvidia-* wheels).
echo "[setup_env] installing acsid_amd requirements"
python -m pip install --no-cache-dir -r "${SCRIPT_DIR}/requirements.txt"

# 4) Sanity checks. ROCm exposes the GPU through torch's CUDA API (CUDA-on-HIP).
echo "[setup_env] verification:"
python - <<PY
import torch, sys
print("  python :", sys.version.split()[0])
print("  torch  :", torch.__version__)
print("  nsight?:", getattr(torch.version, "hip", None))
try:
    print("  cuda?  :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  device :", torch.cuda.get_device_name(0))
except Exception as e:
    print("  cuda probe failed:", e)
PY

# Optional: quick import of the modules that matter for this branch, to catch
# missing deps early (does not import bitsandbytes; AMD does not need it).
echo "[setup_env] module import smoke (MiniOneRec libs):"
cd "${PROJECT_ROOT}/MiniOneRec"
python - <<PY
import transformers, trl, datasets, pandas, numpy, sklearn, fire, wandb
print("  transformers:", transformers.__version__, "trl:", trl.__version__)
PY

echo
echo "[setup_env] DONE."
echo "[setup_env] Activate next time with: source ${VENV_DIR}/bin/activate"
echo "[setup_env] Then run: bash acsid_amd/run_experiments.sh   (set BASE_MODEL)"
