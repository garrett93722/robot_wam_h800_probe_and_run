#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "${SCRIPT_DIR}/config.env" ]]; then
  bash "${SCRIPT_DIR}/make_config_for_current_tree.sh"
fi

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
start_log "bootstrap_h800_dreamzero_env"

cat <<'EOF'
This bootstrap script rebuilds the 2xH800 DreamZero environment from a fresh container.

It applies the failure fixes learned from previous runs:
- install/repair Miniforge under /root/miniforge3
- keep DreamZero on Python 3.11
- install PyTorch 2.8.0 cu129 and flash-attn for H800/H100
- avoid global pip constraints from managed containers
- download DreamZero-DROID checkpoint
- patch DreamZero checkpoint loading to stream safetensor shards
- install IsaacSim system OpenGL libraries
- prepare sim-evals with NVIDIA PyPI and flatdict build fix

Useful toggles:
  PREPARE_SIM_EVALS=0          skip IsaacSim/sim-evals preparation
  DOWNLOAD_DROID_SIM_ASSETS=0  skip DROID sim assets download
  RUN_DREAMZERO_SMOKE=1        run server/client smoke after setup
  RESET_DREAMZERO_ENV=1        remove and recreate the dreamzero conda env
EOF

confirm_install_or_exit

MINIFORGE_PREFIX="${MINIFORGE_PREFIX:-/root/miniforge3}"
MINIFORGE_INSTALLER="${PROJECT_ROOT}/installers/Miniforge3-Linux-x86_64.sh"
MINIFORGE_URL="${MINIFORGE_URL:-https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh}"
DREAMZERO_GIT_URL="${DREAMZERO_GIT_URL:-https://github.com/dreamzero0/dreamzero.git}"

install_system_packages() {
  if [[ "$(id -u)" != "0" || ! -x /usr/bin/apt-get ]]; then
    warn "Skipping apt system packages because this shell is not root or apt-get is unavailable."
    return 0
  fi

  if [[ "${BOOTSTRAP_INSTALL_SYSTEM_DEPS:-1}" != "1" ]]; then
    warn "BOOTSTRAP_INSTALL_SYSTEM_DEPS=0; skipping apt system packages."
    return 0
  fi

  info "Installing base and IsaacSim system libraries with apt."
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ca-certificates curl wget git git-lfs bzip2 \
    libgl1 libegl1 libopengl0 libglib2.0-0 \
    libx11-6 libxext6 libxrender1 libsm6 libice6 libxkbcommon0
}

download_file() {
  local url="$1"
  local output="$2"
  mkdir -p "$(dirname "${output}")"
  if command_exists wget; then
    wget -c --tries=20 --timeout=30 --read-timeout=60 -O "${output}" "${url}"
  elif command_exists curl; then
    curl -L --retry 20 --retry-delay 3 --connect-timeout 30 -C - -o "${output}" "${url}"
  else
    die "Neither wget nor curl is available to download ${url}."
  fi
}

install_miniforge_if_needed() {
  local existing
  existing="$(find_conda_base || true)"
  if [[ -n "${existing}" && -x "${existing}/bin/conda" ]]; then
    info "Conda found at ${existing}; skipping Miniforge install."
    return 0
  fi

  if [[ -e "${MINIFORGE_PREFIX}" && ! -x "${MINIFORGE_PREFIX}/bin/conda" ]]; then
    local backup="${MINIFORGE_PREFIX}.broken.$(timestamp)"
    warn "Found incomplete Miniforge directory ${MINIFORGE_PREFIX}; moving to ${backup}"
    mv "${MINIFORGE_PREFIX}" "${backup}"
  fi

  info "Downloading Miniforge from ${MINIFORGE_URL}"
  download_file "${MINIFORGE_URL}" "${MINIFORGE_INSTALLER}"
  info "Installing Miniforge to ${MINIFORGE_PREFIX}"
  bash "${MINIFORGE_INSTALLER}" -b -p "${MINIFORGE_PREFIX}"
  export PATH="${MINIFORGE_PREFIX}/bin:${MINIFORGE_PREFIX}/condabin:${PATH}"
  # shellcheck disable=SC1091
  source "${MINIFORGE_PREFIX}/etc/profile.d/conda.sh"
  conda --version
}

clone_dreamzero_if_needed() {
  mkdir -p "${SOURCE_ROOT}"
  if [[ -d "${DREAMZERO_REPO}/.git" ]]; then
    info "DreamZero repo already exists at ${DREAMZERO_REPO}; fetching latest refs."
    git -C "${DREAMZERO_REPO}" fetch --all --prune || warn "DreamZero fetch failed; continuing with existing checkout."
  else
    info "Cloning DreamZero repo to ${DREAMZERO_REPO}"
    git clone "${DREAMZERO_GIT_URL}" "${DREAMZERO_REPO}"
  fi
}

maybe_reset_dreamzero_env() {
  if [[ "${RESET_DREAMZERO_ENV:-0}" != "1" ]]; then
    return 0
  fi
  init_conda
  if conda_env_exists "${DREAMZERO_ENV_NAME}"; then
    warn "RESET_DREAMZERO_ENV=1; removing conda env ${DREAMZERO_ENV_NAME}"
    conda env remove -y -n "${DREAMZERO_ENV_NAME}"
  fi
}

install_system_packages
install_miniforge_if_needed
clone_dreamzero_if_needed

info "Running H800 readiness probe."
bash "${SCRIPT_DIR}/02_probe_h800_readiness.sh" || warn "Readiness probe reported warnings; continuing because bootstrap is explicit."

maybe_reset_dreamzero_env

info "Setting up DreamZero conda env."
CONFIRM_INSTALL=1 bash "${SCRIPT_DIR}/20_setup_dreamzero.sh"

info "Downloading DreamZero checkpoint(s)."
bash "${SCRIPT_DIR}/21_download_dreamzero_ckpt.sh"

info "Applying streaming checkpoint load patch."
bash "${SCRIPT_DIR}/24_patch_dreamzero_streaming_load.sh"

if [[ "${PREPARE_SIM_EVALS:-1}" == "1" ]]; then
  info "Preparing sim-evals/IsaacSim dependencies and assets."
  INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-1}" \
  INSTALL_UV="${INSTALL_UV:-1}" \
  DOWNLOAD_DROID_SIM_ASSETS="${DOWNLOAD_DROID_SIM_ASSETS:-1}" \
  DREAMZERO_SIM_PREPARE_ONLY=1 \
  CONFIRM_INSTALL=1 \
  bash "${SCRIPT_DIR}/23_run_dreamzero_droid_sim_eval.sh"
else
  warn "PREPARE_SIM_EVALS=0; skipping sim-evals/IsaacSim preparation."
fi

if [[ "${RUN_DREAMZERO_SMOKE:-0}" == "1" ]]; then
  info "Running DreamZero minimal server/client smoke."
  DREAMZERO_CLIENT_NUM_CHUNKS="${DREAMZERO_CLIENT_NUM_CHUNKS:-1}" \
  DREAMZERO_ENABLE_DIT_CACHE="${DREAMZERO_ENABLE_DIT_CACHE:-0}" \
  DREAMZERO_DISABLE_TORCH_COMPILE="${DREAMZERO_DISABLE_TORCH_COMPILE:-1}" \
  DREAMZERO_SERVER_WARMUP_SECONDS="${DREAMZERO_SERVER_WARMUP_SECONDS:-7200}" \
  bash "${SCRIPT_DIR}/22_run_dreamzero_server_smoke.sh"
else
  warn "RUN_DREAMZERO_SMOKE=0; skipping server smoke."
fi

cat <<EOF

Bootstrap finished.

Recommended verification:
  cd ${SCRIPT_DIR}
  DREAMZERO_CLIENT_NUM_CHUNKS=1 DREAMZERO_ENABLE_DIT_CACHE=0 DREAMZERO_DISABLE_TORCH_COMPILE=1 bash 22_run_dreamzero_server_smoke.sh

For keeping the server alive for sim eval:
  KEEP_DREAMZERO_SERVER=1 DREAMZERO_NPROC_PER_NODE=1 DREAMZERO_CUDA_VISIBLE_DEVICES=0 \\
  DREAMZERO_SERVER_WARMUP_SECONDS=7200 DREAMZERO_ENABLE_DIT_CACHE=0 DREAMZERO_DISABLE_TORCH_COMPILE=1 \\
  bash 22_run_dreamzero_server_smoke.sh

Then run sim eval in another terminal on the other GPU:
  CUDA_VISIBLE_DEVICES=1 DREAMZERO_SIM_EPISODES=1 DREAMZERO_SIM_SCENE=1 \\
  CONFIRM_INSTALL=1 INSTALL_UV=1 DOWNLOAD_DROID_SIM_ASSETS=0 \\
  bash 23_run_dreamzero_droid_sim_eval.sh
EOF
