#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
start_log "setup_dreamzero"

require_dir "${DREAMZERO_REPO}" "DreamZero repo"
info "Repo: ${DREAMZERO_REPO}"
info "Conda env: ${DREAMZERO_ENV_NAME}"

DRV_MAJOR="$(driver_major || true)"
GPU_NAMES="$(gpu_names || true)"
COUNT="$(gpu_count || true)"
info "GPU names: ${GPU_NAMES:-unknown}"
info "GPU count: ${COUNT:-unknown}"
info "NVIDIA driver major: ${DRV_MAJOR:-unknown}"

if [[ -n "${DRV_MAJOR}" && "${DRV_MAJOR}" =~ ^[0-9]+$ && "${DRV_MAJOR}" -lt 575 ]]; then
  die "DreamZero official path expects CUDA 12.9+ wheels. Driver major ${DRV_MAJOR} may be too old. Run 00_probe_env.sh and upgrade/load a newer driver first."
fi
if [[ -n "${COUNT}" && "${COUNT}" =~ ^[0-9]+$ && "${COUNT}" -lt 2 ]]; then
  warn "DreamZero server smoke expects 2 GPUs. Setup can continue, but inference will likely fail."
fi

cat <<'PLAN'

DreamZero official baseline:
- Python 3.11
- PyTorch 2.8+ with CUDA 12.9+ wheel path
- flash-attn, preferring a resumable prebuilt wheel download
- 2 GPUs for distributed inference server

Important:
- H20/H800/H100: use the general path. Do not install GB200-only TensorRT/TransformerEngine acceleration.
- GB200 only: set ENABLE_GB200_OPT=1 if you intentionally want those extra acceleration packages.
PLAN

confirm_install_or_exit

init_conda
RUNNER="$(conda_runner)"
if conda_env_exists "${DREAMZERO_ENV_NAME}"; then
  info "Conda env ${DREAMZERO_ENV_NAME} already exists; skipping create."
else
  info "Creating conda env ${DREAMZERO_ENV_NAME} with Python 3.11"
  "${RUNNER}" create -y -n "${DREAMZERO_ENV_NAME}" python=3.11
fi

conda activate "${DREAMZERO_ENV_NAME}"

# Some managed GPU containers export global pip constraints for their system
# Python. They can conflict with DreamZero's pinned env dependencies inside this
# isolated conda env, e.g. datasets requiring dill<0.3.9 while the host pins
# dill==0.3.9.
unset PIP_CONSTRAINT PIP_CONSTRAINTS

python -m pip install --upgrade pip setuptools wheel 2>&1 | tee -a "${LOG_DIR}/dreamzero_pip_bootstrap_$(timestamp).log"

info "Installing PyTorch 2.8 CUDA 12.9 wheels."
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu129 \
  2>&1 | tee -a "${LOG_DIR}/dreamzero_torch_$(timestamp).log"

info "Installing DreamZero repo without dependency auto-pull so GB200-only accelerators are not installed by accident."
python -m pip install -e "${DREAMZERO_REPO}" --no-deps 2>&1 | tee -a "${LOG_DIR}/dreamzero_editable_$(timestamp).log"

info "Installing DreamZero common runtime dependencies."
python -m pip install \
  av==15.0.0 pyttsx3==2.90 scipy==1.15.3 numpy==1.26.4 matplotlib hydra-core "ray[default]==2.47.1" \
  click gymnasium mujoco termcolor flask "python-socketio>=5.13.0" flask_socketio loguru lmdb meshcat meshcat-shapes \
  rerun-sdk==0.21.0 pygame sshkeyboard msgpack msgpack-numpy peft==0.5.0 pyzmq pin pin-pink timm tyro redis lark \
  datasets==3.6.0 pandas dm_tree openai transformers==4.51.3 albumentations==1.4.18 einops==0.8.1 \
  tianshou==0.5.1 imageio==2.34.2 imageio-ffmpeg wandb opencv-python==4.8.0.74 diffusers==0.30.2 ftfy \
  nvidia-modelopt nvidia-modelopt-core openpi-client==0.1.1 huggingface_hub decord2 deepspeed tiktoken sentencepiece \
  2>&1 | tee -a "${LOG_DIR}/dreamzero_common_deps_$(timestamp).log"

FLASH_LOG="${LOG_DIR}/dreamzero_flash_attn_$(timestamp).log"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3.post1}"
FLASH_ATTN_WHEEL_DIR="${FLASH_ATTN_WHEEL_DIR:-${PROJECT_ROOT}/wheelhouse}"
mkdir -p "${FLASH_ATTN_WHEEL_DIR}"
FLASH_ATTN_WHEEL_URL="$(
  python - "${FLASH_ATTN_VERSION}" <<'PY'
import sys
import torch

version = sys.argv[1]
py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
torch_version = torch.__version__.split("+", 1)[0]
torch_major_minor = ".".join(torch_version.split(".")[:2])
print(
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    f"v{version}/flash_attn-{version}+cu12torch{torch_major_minor}"
    f"cxx11abiTRUE-{py_tag}-{py_tag}-linux_x86_64.whl"
)
PY
)"
FLASH_ATTN_WHEEL_PATH="${FLASH_ATTN_WHEEL_DIR}/$(basename "${FLASH_ATTN_WHEEL_URL}")"

info "Installing flash-attn. First try resumable prebuilt wheel: ${FLASH_ATTN_WHEEL_URL}"
{
  set +e
  DOWNLOAD_STATUS=1
  if command_exists wget; then
    wget -c --tries=20 --timeout=30 --read-timeout=60 -O "${FLASH_ATTN_WHEEL_PATH}" "${FLASH_ATTN_WHEEL_URL}"
    DOWNLOAD_STATUS=$?
  elif command_exists curl; then
    curl -L --retry 20 --retry-delay 3 --connect-timeout 30 -C - -o "${FLASH_ATTN_WHEEL_PATH}" "${FLASH_ATTN_WHEEL_URL}"
    DOWNLOAD_STATUS=$?
  else
    echo "No wget/curl found; skipping prebuilt wheel download."
  fi

  if [[ "${DOWNLOAD_STATUS}" -eq 0 ]]; then
    python -m pip install "${FLASH_ATTN_WHEEL_PATH}"
    INSTALL_STATUS=$?
  else
    echo "Prebuilt wheel download failed with status ${DOWNLOAD_STATUS}; falling back to source build."
    INSTALL_STATUS=1
  fi

  if [[ "${INSTALL_STATUS}" -ne 0 ]]; then
    echo "Source fallback: compile only H800/H100 sm90 with low parallelism."
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}" \
      MAX_JOBS="${FLASH_ATTN_MAX_JOBS:-2}" \
      python -m pip install --no-build-isolation flash-attn
    INSTALL_STATUS=$?
  fi

  python - <<'PY'
import flash_attn
print("flash_attn import OK", getattr(flash_attn, "__version__", "unknown"))
PY
  IMPORT_STATUS=$?
  set -e
  exit $(( INSTALL_STATUS != 0 || IMPORT_STATUS != 0 ))
} 2>&1 | tee "${FLASH_LOG}" || {
  diagnose_log "${FLASH_LOG}"
  die "flash-attn install failed."
}

if echo "${GPU_NAMES}" | grep -Eiq 'GB200' && [[ "${ENABLE_GB200_OPT:-0}" == "1" ]]; then
  info "GB200 detected and ENABLE_GB200_OPT=1; installing GB200-only acceleration packages."
  python -m pip install --no-build-isolation "transformer_engine[pytorch]" \
    2>&1 | tee -a "${LOG_DIR}/dreamzero_gb200_transformer_engine_$(timestamp).log"
  python -m pip install tensorrt==10.13.2.6 tensorrt_cu13==10.13.2.6 tensorrt_cu13_libs==10.13.2.6 tensorrt_cu13_bindings==10.13.2.6 --no-deps \
    2>&1 | tee -a "${LOG_DIR}/dreamzero_gb200_tensorrt_$(timestamp).log"
else
  info "Skipping GB200-only TensorRT/TransformerEngine acceleration."
fi

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
PY

info "DreamZero setup finished. Next: bash ${SCRIPT_DIR}/21_download_dreamzero_ckpt.sh"
