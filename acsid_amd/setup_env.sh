#!/usr/bin/env bash
# Standalone, reproducible environment for the acsid-amd branch.
#
# Default target image: Ubuntu + ROCm 7.2.3 + Python 3.12 + torch 2.11
#   (e.g. ModelScope/Aliyun DSW `rocm7.2.3-py312-torch2.11.0` series).
# The base image already ships a matching torch+ROCm in the SYSTEM python, so
# this script BUILDS A VENV THAT INHERITS SYSTEM PACKAGES
# (`--system-site-packages`): torch comes through from the host untouched,
# and only the acsid_amd requirements are added to the venv. This avoids
# reinstalling torch against a wrong ROCm tag/wheel.
#
# Do NOT run on Windows/macOS -- this script targets the cloud Linux+ROCm box.
#
# Usage (from anywhere):
#   bash acsid_amd/setup_env.sh                # default: .venv-amd, inherits system torch
#   VENV_DIR=/opt/acsid-amd PY_VER=python3.12 \
#       bash acsid_amd/setup_env.sh            # override location / base interpreter
#
# After it finishes:
#   source .venv-amd/bin/activate
#   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "[setup_env] ERROR: this script must run on Linux with a ROCm driver." >&2
    echo "[setup_env]   You are on: $(uname -s)" >&2
    echo "[setup_env] torch ROCm wheels are Linux-only; not usable on Windows/macOS." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv-amd}"
PY_VER="${PY_VER:-python3.12}"

echo "[setup_env] project root : ${PROJECT_ROOT}"
echo "[setup_env] venv dir     : ${VENV_DIR}"
echo "[setup_env] python base  : ${PY_VER}"
echo "[setup_env] inherits system torch (no torch reinstall)"

# 1) Make sure the base interpreter exists.
if ! command -v "${PY_VER}" >/dev/null 2>&1; then
    echo "[setup_env] '${PY_VER}' not found. Common fix on this image:" >&2
    echo "[setup_env]   sudo apt-get update && sudo apt-get install -y ${PY_VER} ${PY_VER}-venv" >&2
    echo "[setup_env] (override base interpreter with: PY_VER=python3.x bash setup_env.sh)" >&2
    exit 3
fi

# Sanity: the base image advertises a torch we can lean on. Fail loudly here
# rather than silently pulling a wrong wheel below. Run the python probe, then
# branch on its exit code (mixing a `<<'PY'` heredoc with `|| { ... }` makes
# bash's parser treat the heredoc body as part of the failing branch and
# explode; keep them separate).
require_system_torch() {
    "${PY_VER}" - <<'PY'
import torch
print("  -> system torch:", torch.__version__)
PY
}
if ! require_system_torch >/tmp/acsid_amd_torch_probe.log 2>&1; then
    cat /tmp/acsid_amd_torch_probe.log >&2
    echo "[setup_env] system torch not importable under ${PY_VER}." >&2
    echo "[setup_env] This script assumes the cloud image preinstalls torch+ROCm" >&2
    echo "[setup_env] (target: rocm7.2.3-py312-torch2.11.0). If so, fix PY_VER." >&2
    echo "[setup_env] If you instead need to install torch yourself, install it from" >&2
    echo "[setup_env] the PyTorch ROCm index matching your driver BEFORE running this" >&2
    echo "[setup_env] script; setup_env.sh picks it up via --system-site-packages." >&2
    exit 5
fi
cat /tmp/acsid_amd_torch_probe.log
rm -f /tmp/acsid_amd_torch_probe.log

# 2) Create the venv (inherit system packages so torch+ROCm pass through).
#    Debian/Ubuntu split the stdlib venv module into the `python3.X-venv` apt
#    package; the failure mode -- "ensurepip is not available" -- is detected
#    and auto-installed (when running as root) before retrying once.
create_venv_with_rescue() {
    local py="$1" target="$2"
    if "$py" -m venv --system-site-packages "$target"; then return 0; fi

    if "$py" -c "import ensurepip" >/dev/null 2>&1; then
        echo "[setup_env] venv creation failed but ensurepip imports ok -- unknown cause; see above." >&2
        return 1
    fi

    local py_short
    py_short="$("$py" -c 'import sys;print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
    local pkg="python${py_short}-venv"
    echo "[setup_env] ensurepip unavailable; need the ${pkg} apt package."

    if ! command -v apt-get >/dev/null 2>&1; then
        echo "[setup_env] apt-get not found (not Debian/Ubuntu)." >&2
        echo "[setup_env] Install ${pkg} through your distro's package manager, or use" >&2
        echo "[setup_env]   mamba env create -f acsid_amd/environment.yml  (conda alternative, see README)" >&2
        return 1
    fi

    if [ "$(id -u)" = "0" ]; then
        echo "[setup_env] running as root -> apt-get install -y ${pkg} (then retry)"
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y "${pkg}"
        if "$py" -m venv --system-site-packages "$target"; then return 0; fi
        echo "[setup_env] venv creation failed even after installing ${pkg}." >&2
        return 1
    else
        echo "[setup_env] not root. Please run (or have an admin run):" >&2
        echo "    sudo apt-get install -y ${pkg}" >&2
        echo "[setup_env] then re-run: bash acsid_amd/setup_env.sh" >&2
        echo "[setup_env] (conda fallback: mamba env create -f acsid_amd/environment.yml)" >&2
        return 1
    fi
}

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "[setup_env] creating venv at ${VENV_DIR} (with system site-packages)"
    create_venv_with_rescue "${PY_VER}" "${VENV_DIR}" \
        || { echo "[setup_env] could not create venv -- aborting." >&2; exit 4; }
else
    echo "[setup_env] venv already exists at ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# 3) Upgrade pip tooling in the venv. The torch inherited from the system is
#    untouched; deps below are installed INTO the venv.
echo "[setup_env] upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

# 4) Project requirements. torch is intentionally NOT in requirements.txt,
#    so pip here only adds the dependencies the project actually needs
#    (transformers, trl, gensim, ...). The system torch stays as-is.
echo "[setup_env] installing acsid_amd requirements into venv"
python -m pip install --no-cache-dir -r "${SCRIPT_DIR}/requirements.txt"

# 5) Sanity checks.
echo "[setup_env] verification:"
python - <<'PY'
import torch, sys, transformers, trl, datasets
print("  python        :", sys.version.split()[0])
print("  torch         :", torch.__version__)
print("  hip runtime   :", getattr(torch.version, "hip", None))
try:
    print("  cuda visible  :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  device        :", torch.cuda.get_device_name(0))
except Exception as e:
    print("  cuda probe failed:", e)
print("  transformers  :", transformers.__version__)
print("  trl           :", trl.__version__)
PY

# Optional: quick import of the modules that matter for this branch, to catch
# missing deps early. (Does not import bitsandbytes; AMD does not need it.)
echo "[setup_env] MiniOneRec lib import smoke:"
cd "${PROJECT_ROOT}/MiniOneRec"
python - <<'PY' || { echo "[setup_env] import smoke failed -- see above"; exit 6; }
import transformers, trl, datasets, pandas, numpy, sklearn, fire, wandb, gensim
print("  all SFT/RL + acsid deps import OK")
PY

echo
echo "[setup_env] DONE."
echo "[setup_env] Activate next time with: source ${VENV_DIR}/bin/activate"
echo "[setup_env] Then run: bash acsid_amd/run_experiments.sh   (set BASE_MODEL)"
