#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
start_log "pack_lingbot_env"

load_config
init_conda

PERSIST_ROOT="${PERSIST_ROOT:-/workspace/persist}"
ENV_NAME="${LINGBOT_ENV_NAME:-lingbot_va}"
ENV_DIR="${PERSIST_ROOT}/envs"
PACK_FILE="${ENV_DIR}/${ENV_NAME}.tar.gz"
FREEZE_FILE="${ENV_DIR}/${ENV_NAME}_pip_freeze.txt"
CONDA_FILE="${ENV_DIR}/${ENV_NAME}_conda_env.yml"

mkdir -p "${ENV_DIR}"

if ! conda_env_exists "${ENV_NAME}"; then
  die "conda env ${ENV_NAME} does not exist. Finish setup first, then pack it."
fi

info "Packing conda env: ${ENV_NAME}"
info "Output: ${PACK_FILE}"

if ! command -v conda-pack >/dev/null 2>&1; then
  info "Installing conda-pack into base env."
  conda install -y -c conda-forge conda-pack || python -m pip install -U conda-pack
fi

conda run --no-capture-output -n "${ENV_NAME}" python -m pip freeze > "${FREEZE_FILE}" || true
conda env export -n "${ENV_NAME}" > "${CONDA_FILE}" || true

conda-pack -n "${ENV_NAME}" -o "${PACK_FILE}" --force

info "Packed env saved."
ls -lh "${PACK_FILE}" "${FREEZE_FILE}" "${CONDA_FILE}" 2>/dev/null || true

cat <<EOF

To restore this env on a fresh GPU container:
  mkdir -p /root/miniforge3/envs/${ENV_NAME}
  tar -xzf ${PACK_FILE} -C /root/miniforge3/envs/${ENV_NAME}
  /root/miniforge3/envs/${ENV_NAME}/bin/conda-unpack

EOF
