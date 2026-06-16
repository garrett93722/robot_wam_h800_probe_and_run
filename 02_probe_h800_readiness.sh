#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
start_log "h800_readiness"

section() {
  echo
  echo "===== $* ====="
}

run_or_note() {
  local label="$1"
  shift
  echo
  echo "--- ${label} ---"
  if "$@"; then
    true
  else
    echo "[not available or failed] command: $*"
  fi
}

section "config"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "SOURCE_ROOT=${SOURCE_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "DREAMZERO_CUDA_VISIBLE_DEVICES=${DREAMZERO_CUDA_VISIBLE_DEVICES}"
echo "DREAMZERO_REPO=${DREAMZERO_REPO}"
echo "DREAMZERO_CKPT_DIR=${DREAMZERO_CKPT_DIR}"
echo "LOG_DIR=${LOG_DIR}"

section "gpu"
if command_exists nvidia-smi; then
  nvidia-smi || true
  echo
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,cuda_version --format=csv,noheader || true
  echo
  run_or_note "nvidia-smi topo -m" nvidia-smi topo -m
else
  echo "nvidia-smi not found"
fi

section "h800 verdict"
COUNT="$(gpu_count || true)"
NAMES="$(gpu_names || true)"
DRV_MAJOR="$(driver_major || true)"
MIN_MEM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1{m=$1} $1<m{m=$1} END{print m+0}' || true)"
echo "gpu_count=${COUNT:-unknown}"
echo "gpu_names=${NAMES:-unknown}"
echo "driver_major=${DRV_MAJOR:-unknown}"
echo "min_gpu_mem_mb=${MIN_MEM:-unknown}"

if [[ "${COUNT:-0}" =~ ^[0-9]+$ && "${COUNT}" -ge 2 ]] && echo "${NAMES}" | grep -Eiq "H800|H100"; then
  echo "READY_GPU=YES"
else
  echo "READY_GPU=NO"
  warn "Expected at least 2 visible H800/H100-class GPUs."
fi

if [[ "${DRV_MAJOR:-0}" =~ ^[0-9]+$ && "${DRV_MAJOR}" -ge 575 ]]; then
  echo "READY_DRIVER_FOR_CU129=YES"
else
  echo "READY_DRIVER_FOR_CU129=CHECK"
  warn "DreamZero script installs PyTorch cu129; driver major >=575 is the safer target."
fi

section "tools"
for cmd in python python3 pip pip3 conda git git-lfs curl wget unzip tar tmux nvcc gcc g++ cmake ninja; do
  if command_exists "${cmd}"; then
    echo "${cmd}: $(command -v "${cmd}")"
    "${cmd}" --version 2>&1 | head -n 1 || true
  else
    echo "${cmd}: not found"
  fi
done

section "repos"
for path in "${DREAMZERO_REPO}" "${LINGBOT_REPO}" "${LIBERO_REPO}"; do
  if [[ -d "${path}" ]]; then
    echo "OK ${path}"
    git -C "${path}" rev-parse --short HEAD 2>/dev/null || true
  else
    echo "MISSING ${path}"
  fi
done

section "network"
for url in \
  "https://github.com" \
  "https://huggingface.co" \
  "https://download.pytorch.org/whl/cu129" \
  "https://pypi.org/simple/"; do
  if command_exists curl && curl -L -I --connect-timeout 8 --max-time 20 -sS "${url}" >/dev/null; then
    echo "OK ${url}"
  else
    echo "FAILED ${url}"
  fi
done

section "ports"
for port in "${DREAMZERO_PORT:-5000}" "${LINGBOT_PORT:-29056}" "${LINGBOT_MASTER_PORT:-29061}"; do
  if port_in_use "${port}"; then
    echo "IN_USE port ${port}"
  else
    echo "FREE port ${port}"
  fi
done

section "torch if installed"
PY="$(command -v python3 || command -v python || true)"
if [[ -n "${PY}" ]]; then
  "${PY}" <<'PY' || true
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    print("available", torch.cuda.is_available(), "count", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(i, p.name, round(p.total_memory / 1024**3, 1), "GB")
except Exception as exc:
    print("torch probe failed", type(exc).__name__, exc)
PY
else
  echo "No python found."
fi

section "recommended first commands"
cat <<'EOF'
If READY_GPU=YES and network is OK:
  bash make_config_for_current_tree.sh
  # clone/upload sources if missing
  CONFIRM_INSTALL=1 bash 20_setup_dreamzero.sh
  bash 21_download_dreamzero_ckpt.sh
  bash 22_run_dreamzero_server_smoke.sh

If DreamZero setup is blocked, use LingBot/LIBERO as the fallback route:
  CONFIRM_INSTALL=1 bash 10_setup_lingbot_va.sh
  bash 11_run_lingbot_va_smoke.sh
EOF
