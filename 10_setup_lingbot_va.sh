#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
start_log "setup_lingbot_va"

info "Preparing LingBot-VA environment plan."
require_dir "${LINGBOT_REPO}" "LingBot-VA repo"
info "Repo: ${LINGBOT_REPO}"
info "Conda env: ${LINGBOT_ENV_NAME}"
info "Checkpoint dir: ${LINGBOT_CKPT_DIR}"

DRV_MAJOR="$(driver_major || true)"
GPU_NAMES="$(gpu_names || true)"
info "GPU names: ${GPU_NAMES:-unknown}"
info "NVIDIA driver major: ${DRV_MAJOR:-unknown}"
if [[ -n "${DRV_MAJOR}" && "${DRV_MAJOR}" =~ ^[0-9]+$ && "${DRV_MAJOR}" -lt 560 ]]; then
  die "Driver appears older than the usual CUDA 12.6 wheel requirement. Run 00_probe_env.sh and upgrade/load a newer driver before installing torch cu126."
fi

cat <<'PLAN'

LingBot-VA official baseline:
- Python == 3.10.16
- torch/vision/audio == 2.9.0 / 0.24.0 / 2.9.0 from cu126 index
- diffusers==0.36.0 transformers==4.55.2 accelerate websockets msgpack opencv-python matplotlib ftfy easydict
- flash-attn installed after torch with --no-build-isolation

This script does not use sudo and keeps LingBot-VA separate from LIBERO.
PLAN

confirm_install_or_exit

init_conda
RUNNER="$(conda_runner)"
if conda_env_exists "${LINGBOT_ENV_NAME}"; then
  info "Conda env ${LINGBOT_ENV_NAME} already exists; skipping create."
else
  info "Creating conda env ${LINGBOT_ENV_NAME} with Python 3.10.16"
  "${RUNNER}" create -y -n "${LINGBOT_ENV_NAME}" python=3.10.16
fi

conda activate "${LINGBOT_ENV_NAME}"
python -m pip install --upgrade pip setuptools wheel 2>&1 | tee -a "${LOG_DIR}/lingbot_pip_bootstrap_$(timestamp).log"

info "Installing PyTorch cu126 wheels."
python -m pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu126 \
  2>&1 | tee -a "${LOG_DIR}/lingbot_torch_$(timestamp).log"

info "Installing LingBot-VA runtime dependencies."
python -m pip install websockets einops diffusers==0.36.0 transformers==4.55.2 accelerate msgpack opencv-python matplotlib ftfy easydict \
  2>&1 | tee -a "${LOG_DIR}/lingbot_runtime_deps_$(timestamp).log"

info "Registering LingBot-VA repo on PYTHONPATH."
LINGBOT_PTH_LOG="${LOG_DIR}/lingbot_repo_path_$(timestamp).log"
LINGBOT_REPO="${LINGBOT_REPO}" python - <<'PY' 2>&1 | tee -a "${LINGBOT_PTH_LOG}"
import os
import site
from pathlib import Path

repo = Path(os.environ["LINGBOT_REPO"]).resolve()
if not (repo / "wan_va").is_dir():
    raise SystemExit(f"wan_va package directory not found under {repo}. Check LINGBOT_REPO/config.env.")
target_dir = Path(site.getsitepackages()[0])
target_dir.mkdir(parents=True, exist_ok=True)
pth = target_dir / "lingbot_va_repo.pth"
pth.write_text(str(repo) + "\n", encoding="utf-8")
print(f"wrote {pth} -> {repo}")
print("wan_va directory exists. Full import is checked after flash-attn is installed.")
PY

info "Trying editable install only as an optional metadata step."
EDITABLE_LOG="${LOG_DIR}/lingbot_editable_$(timestamp).log"
if python -m pip install -e "${LINGBOT_REPO}" 2>&1 | tee -a "${EDITABLE_LOG}"; then
  info "Editable install succeeded."
else
  warn "Editable install failed because upstream pyproject currently points to package 'lingbot_va', while the repo code imports as 'wan_va'. Continuing with .pth PYTHONPATH registration."
fi

if [[ "${SKIP_FLASH_ATTN:-0}" == "1" ]]; then
  warn "SKIP_FLASH_ATTN=1; patching LingBot to use torch attention fallback."
  bash "${SCRIPT_DIR}/14_patch_lingbot_torch_attention_fallback.sh"
  USED_TORCH_ATTN_FALLBACK=1
else
  FLASH_LOG="${LOG_DIR}/lingbot_flash_attn_$(timestamp).log"
  info "Installing flash-attn. This can take a while."
  MAX_JOBS="${MAX_JOBS:-1}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" \
    python -m pip install flash-attn --no-build-isolation \
    2>&1 | tee -a "${FLASH_LOG}" || {
      diagnose_log "${FLASH_LOG}"
      if [[ "${ALLOW_TORCH_ATTN_FALLBACK:-1}" == "1" ]]; then
        warn "flash-attn failed. Patching LingBot to use torch attention fallback and continuing."
        bash "${SCRIPT_DIR}/14_patch_lingbot_torch_attention_fallback.sh"
        USED_TORCH_ATTN_FALLBACK=1
      else
        die "flash-attn install failed. See log above."
      fi
    }
fi

info "Checking LingBot-VA imports."
PYTHONPATH="${LINGBOT_REPO}:${PYTHONPATH:-}" python - <<'PY'
import torch
from wan_va.configs import VA_CONFIGS
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("LingBot configs:", sorted(VA_CONFIGS.keys()))
PY

if [[ "${DOWNLOAD_LINGBOT_CKPT:-0}" == "1" ]]; then
  require_free_gb "$(dirname "${LINGBOT_CKPT_DIR}")" 60
  mkdir -p "${HF_HOME}" "${MODELSCOPE_CACHE}" "$(dirname "${LINGBOT_CKPT_DIR}")"
  export HF_HOME HF_TOKEN
  if [[ -z "${HF_TOKEN:-}" ]]; then
    warn "HF_TOKEN is empty. Public LingBot checkpoints may still work; private/gated access will fail."
  fi
  info "Downloading LingBot checkpoint from HuggingFace: ${LINGBOT_MODEL_ID}"
  if python -m pip show huggingface_hub >/dev/null 2>&1; then
    hf download "${LINGBOT_MODEL_ID}" --local-dir "${LINGBOT_CKPT_DIR}" \
      2>&1 | tee -a "${LOG_DIR}/lingbot_hf_download_$(timestamp).log" || HF_FAILED=1
  else
    python -m pip install "huggingface_hub[cli]"
    hf download "${LINGBOT_MODEL_ID}" --local-dir "${LINGBOT_CKPT_DIR}" \
      2>&1 | tee -a "${LOG_DIR}/lingbot_hf_download_$(timestamp).log" || HF_FAILED=1
  fi
  if [[ "${HF_FAILED:-0}" == "1" && "${USE_MODELSCOPE:-0}" == "1" ]]; then
    info "Trying ModelScope fallback: ${LINGBOT_MODELSCOPE_ID}"
    python -m pip install modelscope
    modelscope download --model "${LINGBOT_MODELSCOPE_ID}" --local_dir "${LINGBOT_CKPT_DIR}" \
      2>&1 | tee -a "${LOG_DIR}/lingbot_modelscope_download_$(timestamp).log"
  fi
  if [[ ! -f "${LINGBOT_CKPT_DIR}/transformer/config.json" ]]; then
    die "LingBot checkpoint download did not produce ${LINGBOT_CKPT_DIR}/transformer/config.json. Re-run download manually or set USE_MODELSCOPE=1."
  fi
fi

if [[ "${USED_TORCH_ATTN_FALLBACK:-0}" == "1" ]]; then
  info "Re-applying torch attention fallback after checkpoint download."
  bash "${SCRIPT_DIR}/14_patch_lingbot_torch_attention_fallback.sh"
fi

cat <<EOF

LingBot-VA setup finished.
Before inference/eval, check:
- ${LINGBOT_CKPT_DIR}/transformer/config.json has "attn_mode": "torch" for Tencent A100 fallback, or "flashattn" if flash-attn is installed.
- Run: bash ${SCRIPT_DIR}/11_run_lingbot_va_smoke.sh
EOF
