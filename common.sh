#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTO_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/config.env}"
EXAMPLE_CONFIG_FILE="${SCRIPT_DIR}/config.example.env"

timestamp() {
  date +"%Y%m%d_%H%M%S"
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

warn() {
  echo "[WARN] $*" >&2
}

info() {
  echo "[INFO] $*"
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

load_config() {
  if [[ -f "${CONFIG_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${CONFIG_FILE}"
  elif [[ -f "${EXAMPLE_CONFIG_FILE}" ]]; then
    warn "config.env not found; using config.example.env defaults. Copy it to config.env and edit paths before setup/eval."
    # shellcheck disable=SC1090
    source "${EXAMPLE_CONFIG_FILE}"
  else
    die "No config.env or config.example.env found under ${SCRIPT_DIR}"
  fi

  PROJECT_ROOT="${PROJECT_ROOT:-${AUTO_PROJECT_ROOT}}"
  SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/sources}"
  LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
  LINGBOT_REPO="${LINGBOT_REPO:-${SOURCE_ROOT}/lingbot-va}"
  DREAMZERO_REPO="${DREAMZERO_REPO:-${SOURCE_ROOT}/dreamzero}"
  LIBERO_REPO="${LIBERO_REPO:-${SOURCE_ROOT}/LIBERO}"
  LINGBOT_ENV_NAME="${LINGBOT_ENV_NAME:-lingbot_va}"
  DREAMZERO_ENV_NAME="${DREAMZERO_ENV_NAME:-dreamzero}"
  LIBERO_ENV_NAME="${LIBERO_ENV_NAME:-libero}"
  LINGBOT_CKPT_DIR="${LINGBOT_CKPT_DIR:-${PROJECT_ROOT}/checkpoints/lingbot-va-posttrain-libero-long}"
  DREAMZERO_CKPT_DIR="${DREAMZERO_CKPT_DIR:-${PROJECT_ROOT}/checkpoints/DreamZero-DROID}"
  DREAMZERO_AGIBOT_CKPT_DIR="${DREAMZERO_AGIBOT_CKPT_DIR:-${PROJECT_ROOT}/checkpoints/DreamZero-AgiBot}"
  LIBERO_DATA_DIR="${LIBERO_DATA_DIR:-${PROJECT_ROOT}/data/libero}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  DREAMZERO_CUDA_VISIBLE_DEVICES="${DREAMZERO_CUDA_VISIBLE_DEVICES:-0,1}"
  mkdir -p "${LOG_DIR}"
}

start_log() {
  local name="$1"
  load_config
  local ts
  ts="$(timestamp)"
  RUN_LOG="${LOG_DIR}/${name}_${ts}.log"
  mkdir -p "${LOG_DIR}"
  exec > >(tee -a "${RUN_LOG}") 2>&1
  info "Log: ${RUN_LOG}"
}

find_conda_base() {
  if command_exists conda; then
    conda info --base 2>/dev/null || true
  fi
}

init_conda() {
  local base
  base="$(find_conda_base)"
  [[ -n "${base}" ]] || die "conda not found. Install Miniconda/Mambaforge first or load the module that provides conda."
  # shellcheck disable=SC1091
  source "${base}/etc/profile.d/conda.sh"
}

conda_runner() {
  if command_exists mamba; then
    echo "mamba"
  elif command_exists micromamba; then
    echo "micromamba"
  else
    echo "conda"
  fi
}

conda_env_exists() {
  local env_name="$1"
  conda env list | awk '{print $1}' | grep -Fxq "${env_name}"
}

activate_env() {
  local env_name="$1"
  init_conda
  conda_env_exists "${env_name}" || die "conda env ${env_name} does not exist."
  conda activate "${env_name}"
}

require_dir() {
  local path="$1"
  local name="$2"
  [[ -d "${path}" ]] || die "${name} not found: ${path}"
}

require_file() {
  local path="$1"
  local name="$2"
  [[ -f "${path}" ]] || die "${name} not found: ${path}"
}

require_var() {
  local var_name="$1"
  local value="${!var_name:-}"
  [[ -n "${value}" ]] || die "${var_name} is empty. Set it in config.env."
}

driver_major() {
  if command_exists nvidia-smi; then
    nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | cut -d. -f1 || true
  fi
}

gpu_names() {
  if command_exists nvidia-smi; then
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd "," - || true
  fi
}

gpu_count() {
  if command_exists nvidia-smi; then
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ' || true
  fi
}

min_free_gb() {
  local path="$1"
  mkdir -p "${path}"
  df -Pk "${path}" | awk 'NR==2 {printf "%.0f", $4/1024/1024}'
}

require_free_gb() {
  local path="$1"
  local need_gb="$2"
  local free_gb
  free_gb="$(min_free_gb "${path}")"
  info "Disk free at ${path}: ${free_gb} GB; required: ${need_gb} GB"
  (( free_gb >= need_gb )) || die "Not enough disk space at ${path}. Free ${free_gb} GB, need ${need_gb} GB."
}

port_in_use() {
  local port="$1"
  if command_exists ss; then
    ss -ltn | awk '{print $4}' | grep -Eq "[:.]${port}$"
  elif command_exists lsof; then
    lsof -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
  else
    return 1
  fi
}

diagnose_log() {
  local file="$1"
  echo
  echo "===== quick diagnosis from ${file} ====="
  if grep -Eiq "CUDA out of memory|out of memory|CUBLAS_STATUS_ALLOC_FAILED" "${file}"; then
    echo "- Looks like GPU memory is insufficient. Try fewer GPUs per process, smaller inputs, offload mode, or a smaller checkpoint."
  fi
  if grep -Eiq "driver.*too old|CUDA driver version is insufficient|unsupported.*CUDA" "${file}"; then
    echo "- CUDA driver/runtime mismatch. Check nvidia-smi driver and install a torch wheel compatible with that driver."
  fi
  if grep -Eiq "flash_attn|flash-attn|No module named 'flash_attn'" "${file}"; then
    echo "- flash-attn problem. Reinstall after torch is installed: MAX_JOBS=8 pip install --no-build-isolation flash-attn"
  fi
  if grep -Eiq "No such file or directory|not found|does not exist" "${file}"; then
    echo "- Missing file/path. Recheck config.env checkpoint and repository paths."
  fi
  if grep -Eiq "Address already in use|port.*in use|OSError:.*98" "${file}"; then
    echo "- Port conflict. Change the *_PORT value in config.env or stop the old process."
  fi
  if grep -Eiq "Connection refused|timed out|WebSocket|websockets" "${file}"; then
    echo "- Server/client connection problem. Confirm the server is still running and host/port match."
  fi
  echo "Full log: ${file}"
}

confirm_install_or_exit() {
  if [[ "${CONFIRM_INSTALL:-0}" != "1" ]]; then
    echo
    warn "Dry-run safety stop. Re-run with CONFIRM_INSTALL=1 to actually create envs/install packages."
    echo "Example: CONFIRM_INSTALL=1 bash $0"
    exit 0
  fi
}
