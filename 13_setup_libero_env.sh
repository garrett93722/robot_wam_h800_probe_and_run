#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
start_log "setup_libero_env"

require_dir "${LIBERO_REPO}" "LIBERO repo"
info "Repo: ${LIBERO_REPO}"
info "Conda env: ${LIBERO_ENV_NAME}"
info "Data dir: ${LIBERO_DATA_DIR}"

cat <<'PLAN'

LIBERO official baseline:
- Separate conda env, Python 3.8.13
- robosuite/LIBERO installed from LIBERO requirements and editable repo
- Official README pins torch==1.11.0+cu113, which is old.

Risk:
- Modern H20/H800 servers often have new drivers but not old CUDA 11.3 local toolkits.
- The cu113 wheel can still run on a new driver, but package conflicts with LingBot/DreamZero are likely.
- Do not mix this env with LingBot-VA or DreamZero.
PLAN

require_free_gb "$(dirname "${LIBERO_DATA_DIR}")" "${LIBERO_MIN_FREE_GB:-80}"
confirm_install_or_exit

init_conda
RUNNER="$(conda_runner)"
if conda_env_exists "${LIBERO_ENV_NAME}"; then
  info "Conda env ${LIBERO_ENV_NAME} already exists; skipping create."
else
  info "Creating conda env ${LIBERO_ENV_NAME} with Python 3.8.13"
  "${RUNNER}" create -y -n "${LIBERO_ENV_NAME}" python=3.8.13
fi

conda activate "${LIBERO_ENV_NAME}"
python -m pip install --upgrade pip setuptools wheel 2>&1 | tee -a "${LOG_DIR}/libero_pip_bootstrap_$(timestamp).log"

info "Installing LIBERO requirements."
python -m pip install -r "${LIBERO_REPO}/requirements.txt" 2>&1 | tee -a "${LOG_DIR}/libero_requirements_$(timestamp).log"

info "Installing official old torch cu113 wheels."
python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  2>&1 | tee -a "${LOG_DIR}/libero_torch_cu113_$(timestamp).log" || {
    warn "Official torch 1.11/cu113 install failed. Fallback idea: use CPU-only LIBERO import/render checks, or install a newer torch only if LingBot client code permits it."
    false
  }

info "Installing LIBERO editable."
python -m pip install -e "${LIBERO_REPO}" 2>&1 | tee -a "${LOG_DIR}/libero_editable_$(timestamp).log"

info "Installing lightweight client-side packages needed by LingBot LIBERO client."
python -m pip install websockets msgpack msgpack-numpy imageio "huggingface_hub[cli]" \
  2>&1 | tee -a "${LOG_DIR}/libero_client_extra_deps_$(timestamp).log"
python -m pip install lerobot==0.3.3 --no-deps \
  2>&1 | tee -a "${LOG_DIR}/libero_lerobot_nodeps_$(timestamp).log" || {
    warn "lerobot==0.3.3 install failed. Patching LingBot LIBERO client to use a stdlib JSON fallback."
    bash "${SCRIPT_DIR}/16_patch_lingbot_libero_client_fallback.sh"
  }

if [[ "${DOWNLOAD_LIBERO_DATA:-0}" == "1" ]]; then
  bash "${SCRIPT_DIR}/18_download_libero_datasets.sh"
else
  warn "Dataset download skipped. Set DOWNLOAD_LIBERO_DATA=1 after confirming LIBERO_DATA_DIR and disk space."
fi

info "LIBERO setup finished."
