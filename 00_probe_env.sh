#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
mkdir -p "${LOG_DIR}"
TS="$(date +"%Y%m%d_%H%M%S")"
REPORT="${LOG_DIR}/env_report_${TS}.txt"

exec > >(tee "${REPORT}") 2>&1

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

first_python() {
  if command -v python3 >/dev/null 2>&1; then
    echo python3
  elif command -v python >/dev/null 2>&1; then
    echo python
  else
    echo ""
  fi
}

probe_url() {
  local name="$1"
  local url="$2"
  printf "%-28s %s " "${name}" "${url}"
  if command -v curl >/dev/null 2>&1; then
    if curl -L -I --connect-timeout 8 --max-time 15 -sS "${url}" >/dev/null; then
      echo "OK"
    else
      echo "FAILED"
    fi
  elif command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1; then
    local py
    py="$(first_python)"
    "${py}" - "${url}" <<'PY' && echo "OK" || echo "FAILED"
import sys, urllib.request
url = sys.argv[1]
req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "robot-wam-probe"})
with urllib.request.urlopen(req, timeout=15) as r:
    pass
PY
  else
    echo "SKIPPED no curl/python"
  fi
}

section "basic"
echo "report_file=${REPORT}"
echo "time=$(date -Is)"
echo "hostname=$(hostname 2>/dev/null || true)"
echo "user=$(whoami 2>/dev/null || true)"
echo "pwd=$(pwd)"
echo "shell=${SHELL:-unknown}"
echo "kernel=$(uname -a 2>/dev/null || true)"

section "os and libc"
run_or_note "os-release" bash -lc 'cat /etc/os-release'
run_or_note "glibc" bash -lc 'ldd --version | head -n 3'

section "gpu and driver"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
  echo
  echo "--- gpu query csv ---"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,cuda_version --format=csv,noheader || true
else
  echo "nvidia-smi not found"
fi

section "cuda runtime and nvcc"
echo "CUDA_HOME=${CUDA_HOME:-}"
echo "CUDA_PATH=${CUDA_PATH:-}"
run_or_note "nvcc -V" nvcc -V
run_or_note "cuda libs in ldconfig" bash -lc "ldconfig -p 2>/dev/null | grep -Ei 'cuda|cudnn|nccl' | head -n 80"

section "python package managers"
for cmd in python python3 pip pip3 conda mamba micromamba; do
  if command -v "${cmd}" >/dev/null 2>&1; then
    echo "${cmd}: $(command -v "${cmd}")"
    "${cmd}" --version 2>&1 || true
  else
    echo "${cmd}: not found"
  fi
done

section "developer tools"
for cmd in git git-lfs gcc g++ cmake ninja make curl wget unzip tar tmux uv; do
  if command -v "${cmd}" >/dev/null 2>&1; then
    echo "${cmd}: $(command -v "${cmd}")"
    "${cmd}" --version 2>&1 | head -n 2 || true
  else
    echo "${cmd}: not found"
  fi
done

section "disk"
df -h || true

section "memory"
free -h || true

section "pip and conda sources"
run_or_note "pip config" bash -lc 'python3 -m pip config list 2>/dev/null || python -m pip config list 2>/dev/null || pip config list 2>/dev/null'
run_or_note "conda channels" bash -lc 'conda config --show channels 2>/dev/null || true'

section "network"
probe_url "GitHub" "https://github.com"
probe_url "HuggingFace" "https://huggingface.co"
probe_url "ModelScope" "https://modelscope.cn"
probe_url "PyPI" "https://pypi.org/simple/"
probe_url "PyTorch cu126 index" "https://download.pytorch.org/whl/cu126"
probe_url "PyTorch cu129 index" "https://download.pytorch.org/whl/cu129"

section "torch import"
PY="$(first_python)"
if [[ -n "${PY}" ]]; then
  "${PY}" <<'PY' || true
import json
try:
    import torch
    print("torch_import=OK")
    print("torch_version=" + str(torch.__version__))
    print("torch_cuda_version=" + str(torch.version.cuda))
    print("torch_cuda_available=" + str(torch.cuda.is_available()))
    print("torch_gpu_count=" + str(torch.cuda.device_count()))
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"torch_gpu_{i}_name={props.name}")
        print(f"torch_gpu_{i}_memory_gb={props.total_memory/1024**3:.1f}")
except Exception as exc:
    print("torch_import=FAILED")
    print(type(exc).__name__ + ": " + str(exc))
PY
else
  echo "No python found; torch import skipped"
fi

section "important python package imports"
if [[ -n "${PY}" ]]; then
  "${PY}" <<'PY' || true
mods = [
    "flash_attn", "transformers", "diffusers", "accelerate", "websockets",
    "cv2", "robosuite", "libero", "lerobot", "huggingface_hub", "modelscope",
]
import importlib
for mod in mods:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "unknown")
        print(f"{mod}=OK version={ver}")
    except Exception as exc:
        print(f"{mod}=MISSING_OR_FAILED {type(exc).__name__}: {exc}")
PY
fi

section "cuda-related environment"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
echo "CUDA_HOME=${CUDA_HOME:-}"
echo "PATH=${PATH:-}"
echo
echo "--- PATH entries containing cuda ---"
printf '%s\n' "${PATH:-}" | tr ':' '\n' | grep -Ei 'cuda|nvidia' || true
echo
echo "--- LD_LIBRARY_PATH entries containing cuda ---"
printf '%s\n' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -Ei 'cuda|nvidia|cudnn|nccl' || true

section "automatic route suggestion"
GPU_NAMES=""
GPU_COUNT="0"
MIN_MEM_MB="0"
DRIVER_MAJOR="0"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd ',' - || true)"
  GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ' || true)"
  MIN_MEM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1{m=$1} $1<m{m=$1} END{print m+0}' || true)"
  DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | cut -d. -f1 || true)"
fi
echo "detected_gpu_names=${GPU_NAMES:-none}"
echo "detected_gpu_count=${GPU_COUNT}"
echo "detected_min_gpu_mem_mb=${MIN_MEM_MB}"
echo "detected_driver_major=${DRIVER_MAJOR}"

if [[ "${GPU_COUNT}" == "0" || -z "${GPU_NAMES}" ]]; then
  echo "recommendation=No visible NVIDIA GPU. Do only repo/config checks; do not start LingBot/DreamZero inference."
elif echo "${GPU_NAMES}" | grep -Eiq 'H800|H100|GB200'; then
  if (( DRIVER_MAJOR >= 560 )); then
    echo "recommendation=2xH800/H100/GB200-class GPU detected with likely new enough driver. Prioritize DreamZero server smoke, then LingBot-VA i2av, then LIBERO eval."
  else
    echo "recommendation=Strong GPU detected but driver may be too old for cu126/cu129 wheels. Fix driver/module first, then run LingBot/DreamZero setup."
  fi
elif echo "${GPU_NAMES}" | grep -Eiq 'H20'; then
  echo "recommendation=H20 detected. Prefer LingBot-VA i2av and LIBERO route first. Try DreamZero 2-GPU server smoke only after confirming driver >=560 and enough free VRAM."
elif echo "${GPU_NAMES}" | grep -Eiq 'A100'; then
  if (( GPU_COUNT >= 2 && MIN_MEM_MB >= 70000 && DRIVER_MAJOR >= 575 )); then
    echo "recommendation=2xA100 80GB-class setup with likely CUDA 12.9-capable driver. You can try DreamZero server smoke, but LingBot-VA i2av remains the lower-risk first run."
  elif (( DRIVER_MAJOR >= 560 && MIN_MEM_MB >= 39000 )); then
    echo "recommendation=A100 detected. Prioritize LingBot-VA i2av smoke on one GPU, then LIBERO. DreamZero needs 2 visible GPUs and driver >=575 for the official cu129 path."
  else
    echo "recommendation=A100 detected, but driver/VRAM may not satisfy LingBot cu126 or DreamZero cu129. Run LIBERO/native checks first and inspect driver before installing heavy packages."
  fi
elif (( MIN_MEM_MB >= 70000 && GPU_COUNT >= 2 )); then
  echo "recommendation=2+ high-memory GPUs detected. Try LingBot-VA first; DreamZero may work if driver supports CUDA 12.9 wheels."
elif (( MIN_MEM_MB >= 24000 )); then
  echo "recommendation=Single/medium VRAM GPU. LingBot-VA i2av may work with offload; LIBERO env can be prepared; DreamZero 14B server is high risk."
else
  echo "recommendation=GPU VRAM seems limited. Focus on LIBERO environment checks and avoid 14B model inference."
fi

section "next command"
echo "Run: python 01_summarize_env.py ${REPORT}"

if [[ -n "${PY}" && -f "${SCRIPT_DIR}/01_summarize_env.py" ]]; then
  echo
  echo "Generating markdown summary..."
  "${PY}" "${SCRIPT_DIR}/01_summarize_env.py" "${REPORT}" || true
fi

echo
echo "Probe complete: ${REPORT}"
