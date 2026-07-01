#!/usr/bin/env bash
set -euo pipefail

# Fresh-GPU bootstrap for LingBot-VA + libero-long-lerobot.
# It prefers restoring a packed env from /workspace/persist/envs/lingbot_va.tar.gz.
# If no packed env exists, it installs Miniforge from a fast mirror, configures
# conda/pip mirrors, and runs 10_setup_lingbot_va.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
start_log "bootstrap_lingbot_libero_long"

PERSIST_ROOT="${PERSIST_ROOT:-/workspace/persist}"
MINIFORGE_PREFIX="${MINIFORGE_PREFIX:-/root/miniforge3}"
MINIFORGE_URL="${MINIFORGE_URL:-https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh}"
PIP_INDEX_URL_FAST="${PIP_INDEX_URL_FAST:-https://mirrors.tencent.com/pypi/simple/}"
PIP_TRUSTED_HOST_FAST="${PIP_TRUSTED_HOST_FAST:-mirrors.tencent.com}"
ENV_NAME="${LINGBOT_ENV_NAME:-lingbot_va}"
PACK_FILE="${PERSIST_ROOT}/envs/${ENV_NAME}.tar.gz"

mkdir -p \
  "${PERSIST_ROOT}/code" \
  "${PERSIST_ROOT}/sources" \
  "${PERSIST_ROOT}/data/lerobot" \
  "${PERSIST_ROOT}/checkpoints" \
  "${PERSIST_ROOT}/runs" \
  "${PERSIST_ROOT}/logs" \
  "${PERSIST_ROOT}/envs" \
  "${PERSIST_ROOT}/wheels" \
  "${PERSIST_ROOT}/.cache"

link_or_keep() {
  local target="$1"
  local link="$2"
  if [[ -L "${link}" ]]; then
    ln -sfnT "${target}" "${link}"
  elif [[ -e "${link}" ]]; then
    info "Keeping existing non-symlink path: ${link}"
  else
    ln -sfnT "${target}" "${link}"
  fi
}

link_or_keep "${PERSIST_ROOT}/sources" /workspace/sources
link_or_keep "${PERSIST_ROOT}/data" /workspace/data
link_or_keep "${PERSIST_ROOT}/checkpoints" /workspace/checkpoints
link_or_keep "${PERSIST_ROOT}/runs" /workspace/runs

ensure_miniforge() {
  if [[ -x "${MINIFORGE_PREFIX}/bin/conda" ]]; then
    info "Miniforge exists: ${MINIFORGE_PREFIX}"
    return
  fi
  info "Installing Miniforge from ${MINIFORGE_URL}"
  cd /workspace
  curl -L -o Miniforge3.sh "${MINIFORGE_URL}" || wget -O Miniforge3.sh "${MINIFORGE_URL}"
  rm -rf "${MINIFORGE_PREFIX}"
  bash Miniforge3.sh -b -p "${MINIFORGE_PREFIX}"
}

configure_mirrors() {
  cat > /root/.condarc <<'EOF'
channels:
  - conda-forge
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF
  python -m pip config set global.index-url "${PIP_INDEX_URL_FAST}" || true
  python -m pip config set global.trusted-host "${PIP_TRUSTED_HOST_FAST}" || true
}

ensure_miniforge
# shellcheck disable=SC1091
source "${MINIFORGE_PREFIX}/etc/profile.d/conda.sh"
configure_mirrors

restore_packed_env() {
  if [[ ! -f "${PACK_FILE}" ]]; then
    return 1
  fi
  if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    info "Conda env ${ENV_NAME} already exists; not restoring packed env."
    return 0
  fi
  info "Restoring ${ENV_NAME} from ${PACK_FILE}"
  mkdir -p "${MINIFORGE_PREFIX}/envs/${ENV_NAME}"
  tar -xzf "${PACK_FILE}" -C "${MINIFORGE_PREFIX}/envs/${ENV_NAME}"
  "${MINIFORGE_PREFIX}/envs/${ENV_NAME}/bin/conda-unpack" || true
}

if ! restore_packed_env; then
  info "No packed env found. Running normal LingBot setup with torch attention fallback."
  cd "${SCRIPT_DIR}"
  CONFIRM_INSTALL=1 SKIP_FLASH_ATTN=1 DOWNLOAD_LINGBOT_CKPT=0 bash "${SCRIPT_DIR}/10_setup_lingbot_va.sh"
fi

conda activate "${ENV_NAME}"
python -m pip config set global.index-url "${PIP_INDEX_URL_FAST}" || true
python -m pip config set global.trusted-host "${PIP_TRUSTED_HOST_FAST}" || true

info "Installing/checking LingBot training extras."
PIP_CONSTRAINT= PIP_CONSTRAINTS= python -m pip install \
  scipy wandb "lerobot==0.3.3" --no-deps \
  -i "${PIP_INDEX_URL_FAST}" || \
PIP_CONSTRAINT= PIP_CONSTRAINTS= python -m pip install \
  scipy wandb "lerobot==0.3.3" --no-deps \
  -i https://pypi.org/simple

info "Checking required persistent assets."
BASE_CKPT="${LINGBOT_BASE_CKPT:-/workspace/checkpoints/lingbot-va-base}"
DATASET_DIR="${LINGBOT_TRAIN_DATASET_DIR:-/workspace/data/lerobot/libero-long-lerobot}"

[[ -f "${BASE_CKPT}/transformer/config.json" ]] || die "Missing base checkpoint: ${BASE_CKPT}/transformer/config.json"
[[ -f "${DATASET_DIR}/meta/info.json" ]] || die "Missing dataset metadata: ${DATASET_DIR}/meta/info.json"
[[ -d "${DATASET_DIR}/latents" ]] || die "Missing dataset latents: ${DATASET_DIR}/latents"

info "Import check."
PYTHONPATH="/workspace/sources/lingbot-va:${PYTHONPATH:-}" python - <<'PY'
import torch
from wan_va.configs import VA_CONFIGS
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("has libero_train", "libero_train" in VA_CONFIGS)
PY

cat <<EOF

LingBot libero-long bootstrap is ready.

Run a 1-step smoke:
  cd /workspace/robot_wam_h800_probe_and_run
  LINGBOT_CKPT_DIR=${BASE_CKPT} \\
  LINGBOT_TRAIN_DATASET_DIR=${DATASET_DIR} \\
  LINGBOT_TRAIN_CONFIG=libero_train \\
  LINGBOT_TRAIN_STEPS=1 \\
  LINGBOT_TRAIN_NGPU=1 \\
  LINGBOT_TRAIN_SAVE_ROOT=/workspace/runs/lingbot_libero_long_base_smoke_\$(date +%Y%m%d_%H%M%S) \\
  CUDA_VISIBLE_DEVICES=0 \\
  CONFIRM_INSTALL=1 \\
  SKIP_FLASH_ATTN=1 \\
  bash 30_run_lingbot_single_task_train_smoke.sh

Before stopping the GPU, save the env:
  bash /workspace/robot_wam_h800_probe_and_run/07_pack_lingbot_env_to_persist.sh

EOF
