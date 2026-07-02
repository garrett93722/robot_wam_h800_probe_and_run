#!/usr/bin/env bash
set -euo pipefail

# One-command setup + smoke train for:
#   base model: robbyant/lingbot-va-base
#   dataset:    robbyant/libero-long-lerobot
#
# This script runs real LingBot-VA fine-tune smoke work. It deliberately does
# not include artificial GPU idling/anti-reclaim behavior.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
start_log "setup_run_lingbot_libero_long"

confirm_install_or_exit

PROJECT_ROOT="${PROJECT_ROOT:-/workspace}"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/sources}"
LINGBOT_REPO="${LINGBOT_REPO:-${SOURCE_ROOT}/lingbot-va}"
# This script is specifically for the base-model libero-long task. Do not let
# an older config.env posttrain checkpoint path silently redirect it.
LINGBOT_CKPT_DIR="${LINGBOT_BASE_CKPT_DIR:-${PROJECT_ROOT}/checkpoints/lingbot-va-base}"
LINGBOT_ENV_NAME="${LINGBOT_ENV_NAME:-lingbot_va}"
DATASET_DIR="${LINGBOT_TRAIN_DATASET_DIR:-${PROJECT_ROOT}/data/lerobot/libero-long-lerobot}"
SMOKE_STEPS="${LINGBOT_TRAIN_STEPS:-1}"
SMOKE_NGPU="${LINGBOT_TRAIN_NGPU:-1}"
SMOKE_SAVE_ROOT="${LINGBOT_TRAIN_SAVE_ROOT:-${PROJECT_ROOT}/runs/lingbot_libero_long_base_smoke_$(timestamp)}"

MINIFORGE_PREFIX="${MINIFORGE_PREFIX:-/root/miniforge3}"
MINIFORGE_URL="${MINIFORGE_URL:-https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh}"
PIP_INDEX_URL_FAST="${PIP_INDEX_URL_FAST:-https://mirrors.tencent.com/pypi/simple/}"
PIP_TRUSTED_HOST_FAST="${PIP_TRUSTED_HOST_FAST:-mirrors.tencent.com}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"

mkdir -p "${SOURCE_ROOT}" "${PROJECT_ROOT}/checkpoints" "${PROJECT_ROOT}/data/lerobot" "${PROJECT_ROOT}/runs" "${HF_HOME}"

section() {
  echo
  echo "===== $* ====="
}

install_miniforge() {
  if [[ -x "${MINIFORGE_PREFIX}/bin/conda" ]]; then
    info "Miniforge exists: ${MINIFORGE_PREFIX}"
    return
  fi
  info "Installing Miniforge from mirror."
  cd "${PROJECT_ROOT}"
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
  "${MINIFORGE_PREFIX}/bin/python" -m pip config set global.index-url "${PIP_INDEX_URL_FAST}" || true
  "${MINIFORGE_PREFIX}/bin/python" -m pip config set global.trusted-host "${PIP_TRUSTED_HOST_FAST}" || true
}

clone_repos() {
  if [[ ! -d "${LINGBOT_REPO}/.git" ]]; then
    info "Cloning LingBot-VA."
    git clone https://github.com/Robbyant/lingbot-va.git "${LINGBOT_REPO}"
  else
    info "LingBot-VA repo exists; pulling latest."
    git -C "${LINGBOT_REPO}" pull --ff-only || true
  fi
}

download_assets() {
  export HF_HOME HF_ENDPOINT HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
  info "HF_ENDPOINT=${HF_ENDPOINT}"
  info "HF_HOME=${HF_HOME}"
  info "HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET}; HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER}"
  "${MINIFORGE_PREFIX}/bin/python" -m pip install -U "huggingface_hub[cli]" hf-xet -i "${PIP_INDEX_URL_FAST}" || \
    "${MINIFORGE_PREFIX}/bin/python" -m pip install -U "huggingface_hub[cli]" hf-xet -i https://pypi.org/simple

  if [[ ! -f "${LINGBOT_CKPT_DIR}/transformer/config.json" ]]; then
    info "Downloading lingbot-va-base checkpoint to ${LINGBOT_CKPT_DIR}"
    "${MINIFORGE_PREFIX}/bin/python" - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="robbyant/lingbot-va-base",
    repo_type="model",
    local_dir="${LINGBOT_CKPT_DIR}",
    ignore_patterns=["**/.DS_Store", ".DS_Store"],
    max_workers=1,
)
print("base checkpoint downloaded")
PY
  else
    info "Base checkpoint exists: ${LINGBOT_CKPT_DIR}"
  fi

  if [[ ! -f "${DATASET_DIR}/meta/info.json" || ! -d "${DATASET_DIR}/latents" ]]; then
    info "Downloading libero-long-lerobot dataset to ${DATASET_DIR}"
    "${MINIFORGE_PREFIX}/bin/python" - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="robbyant/libero-long-lerobot",
    repo_type="dataset",
    local_dir="${DATASET_DIR}",
    allow_patterns=["meta/**", "data/**", "latents/**", "empty_emb.pt"],
    ignore_patterns=["**/.DS_Store", ".DS_Store"],
    max_workers=1,
)
print("dataset downloaded")
PY
  else
    info "Dataset exists: ${DATASET_DIR}"
  fi
}

ensure_empty_emb() {
  if [[ -f "${DATASET_DIR}/empty_emb.pt" ]]; then
    info "empty_emb.pt exists."
    return
  fi
  info "Generating empty_emb.pt from base checkpoint."
  conda activate "${LINGBOT_ENV_NAME}"
  export PATH="${MINIFORGE_PREFIX}/envs/${LINGBOT_ENV_NAME}/bin:${PATH}"
  PYTHONPATH="${LINGBOT_REPO}:${PYTHONPATH:-}" python - <<PY
import torch
from pathlib import Path
from wan_va.modules.utils import load_text_encoder, load_tokenizer

ckpt = Path("${LINGBOT_CKPT_DIR}")
dataset = Path("${DATASET_DIR}")
out = dataset / "empty_emb.pt"
dtype = torch.bfloat16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = load_tokenizer(ckpt / "tokenizer")
text_encoder = load_text_encoder(ckpt / "text_encoder", torch_dtype=dtype, torch_device=device).eval()
text_inputs = tokenizer([""], padding="max_length", max_length=226, truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
input_ids = text_inputs.input_ids
mask = text_inputs.attention_mask
seq_lens = mask.gt(0).sum(dim=1).long()
enc_device = next(text_encoder.parameters()).device
with torch.no_grad():
    embeds = text_encoder(input_ids.to(enc_device), mask.to(enc_device)).last_hidden_state
embeds = embeds.to(dtype=dtype, device=device)
embeds = [u[:v] for u, v in zip(embeds, seq_lens)]
embeds = torch.stack([torch.cat([u, u.new_zeros(226 - u.size(0), u.size(1))]) for u in embeds], dim=0)
torch.save(embeds[0].cpu(), out)
print("saved", out, tuple(embeds[0].shape), embeds[0].dtype)
PY
}

section "GPU check"
nvidia-smi || warn "nvidia-smi failed. This must run in the GPU training terminal."

section "Miniforge and mirrors"
install_miniforge
# shellcheck disable=SC1091
source "${MINIFORGE_PREFIX}/etc/profile.d/conda.sh"
configure_mirrors

section "Repos and assets"
clone_repos
download_assets

section "LingBot env"
cd "${SCRIPT_DIR}"
if ! conda env list | awk '{print $1}' | grep -Fxq "${LINGBOT_ENV_NAME}"; then
  info "Creating LingBot env via 10_setup_lingbot_va.sh"
  LINGBOT_REPO="${LINGBOT_REPO}" LINGBOT_CKPT_DIR="${LINGBOT_CKPT_DIR}" \
    CONFIRM_INSTALL=1 SKIP_FLASH_ATTN=1 DOWNLOAD_LINGBOT_CKPT=0 \
    bash "${SCRIPT_DIR}/10_setup_lingbot_va.sh"
else
  info "LingBot env exists: ${LINGBOT_ENV_NAME}"
fi

conda activate "${LINGBOT_ENV_NAME}"
export PATH="${MINIFORGE_PREFIX}/envs/${LINGBOT_ENV_NAME}/bin:${PATH}"
hash -r
python -m pip config set global.index-url "${PIP_INDEX_URL_FAST}" || true
python -m pip config set global.trusted-host "${PIP_TRUSTED_HOST_FAST}" || true
PIP_CONSTRAINT= PIP_CONSTRAINTS= python -m pip install \
  scipy wandb lerobot==0.3.3 datasets==3.6.0 "dill<0.3.9" "multiprocess<0.70.17" \
  "packaging>=24.2" jsonlines av \
  -i "${PIP_INDEX_URL_FAST}" || \
PIP_CONSTRAINT= PIP_CONSTRAINTS= python -m pip install \
  scipy wandb lerobot==0.3.3 datasets==3.6.0 "dill<0.3.9" "multiprocess<0.70.17" \
  "packaging>=24.2" jsonlines av \
  -i https://pypi.org/simple

section "empty_emb"
ensure_empty_emb

section "Final checks"
[[ -f "${LINGBOT_CKPT_DIR}/transformer/config.json" ]] || die "missing ${LINGBOT_CKPT_DIR}/transformer/config.json"
[[ -f "${DATASET_DIR}/meta/info.json" ]] || die "missing ${DATASET_DIR}/meta/info.json"
[[ -d "${DATASET_DIR}/latents" ]] || die "missing ${DATASET_DIR}/latents"
[[ -f "${DATASET_DIR}/empty_emb.pt" ]] || die "missing ${DATASET_DIR}/empty_emb.pt"
PYTHONPATH="${LINGBOT_REPO}:${PYTHONPATH:-}" python - <<'PY'
import torch
from wan_va.configs import VA_CONFIGS
print("python import OK")
print("torch", torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())
print("libero_train", "libero_train" in VA_CONFIGS)
PY

section "Run smoke"
if [[ "${RUN_LINGBOT_LIBERO_LONG_SMOKE:-1}" == "1" ]]; then
  cd "${SCRIPT_DIR}"
  LINGBOT_CKPT_DIR="${LINGBOT_CKPT_DIR}" \
  LINGBOT_REPO="${LINGBOT_REPO}" \
  LINGBOT_TRAIN_DATASET_DIR="${DATASET_DIR}" \
  LINGBOT_TRAIN_CONFIG=libero_train \
  LINGBOT_TRAIN_STEPS="${SMOKE_STEPS}" \
  LINGBOT_TRAIN_NGPU="${SMOKE_NGPU}" \
  LINGBOT_TRAIN_SAVE_ROOT="${SMOKE_SAVE_ROOT}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  CONFIRM_INSTALL=1 \
  SKIP_FLASH_ATTN=1 \
  bash "${SCRIPT_DIR}/30_run_lingbot_single_task_train_smoke.sh"
else
  info "RUN_LINGBOT_LIBERO_LONG_SMOKE=0; skipping train smoke."
fi

cat <<EOF

Done.
Checkpoint: ${LINGBOT_CKPT_DIR}
Dataset:    ${DATASET_DIR}
Output:     ${SMOKE_SAVE_ROOT}

EOF
