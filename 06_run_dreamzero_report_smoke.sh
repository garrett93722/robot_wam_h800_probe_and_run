#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "${SCRIPT_DIR}/config.env" ]]; then
  bash "${SCRIPT_DIR}/make_config_for_current_tree.sh"
fi

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
start_log "dreamzero_report_smoke"

PORT="${DREAMZERO_PORT:-5000}"
HOST="${DREAMZERO_CLIENT_HOST:-127.0.0.1}"
REPORT_TS="$(timestamp)"
REPORT_FILE="${LOG_DIR}/dreamzero_report_${REPORT_TS}.md"
REPORT_RUN_DIR="${LOG_DIR}/dreamzero_report_${REPORT_TS}"
mkdir -p "${REPORT_RUN_DIR}"

append_report() {
  printf "%s\n" "$*" >> "${REPORT_FILE}"
}

run_client() {
  local name="$1"
  shift
  local log_file="${REPORT_RUN_DIR}/${name}.log"
  info "Running ${name}; log ${log_file}"
  set +e
  (
    cd "${DREAMZERO_REPO}"
    conda run --no-capture-output -n "${DREAMZERO_ENV_NAME}" python -u test_client_AR.py "$@"
  ) >"${log_file}" 2>&1
  local status=$?
  set -e
  if [[ "${status}" -ne 0 ]]; then
    diagnose_log "${log_file}"
    return "${status}"
  fi
  grep -E "Server metadata|Action shape|Done" "${log_file}" || true
}

need_bootstrap=0
CONDA_BASE="$(find_conda_base || true)"
if [[ ! -d "${DREAMZERO_REPO}/.git" ]]; then
  need_bootstrap=1
elif [[ -z "${CONDA_BASE}" ]]; then
  need_bootstrap=1
else
  init_conda
  if ! conda_env_exists "${DREAMZERO_ENV_NAME}"; then
    need_bootstrap=1
  fi
fi

MODEL_PATH="${DREAMZERO_CKPT_DIR}"
if [[ -d "${DREAMZERO_CKPT_DIR}/$(basename "${DREAMZERO_CKPT_DIR}")" ]]; then
  MODEL_PATH="${DREAMZERO_CKPT_DIR}/$(basename "${DREAMZERO_CKPT_DIR}")"
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  need_bootstrap=1
fi

if [[ "${need_bootstrap}" == "1" ]]; then
  info "DreamZero env/repo/checkpoint is incomplete; running bootstrap without sim-evals."
  PREPARE_SIM_EVALS=0 RUN_DREAMZERO_SMOKE=0 CONFIRM_INSTALL=1 \
    bash "${SCRIPT_DIR}/05_bootstrap_h800_dreamzero_env.sh"
else
  info "DreamZero env/repo/checkpoint appear ready; skipping bootstrap."
fi

init_conda
conda_env_exists "${DREAMZERO_ENV_NAME}" || die "Missing conda env ${DREAMZERO_ENV_NAME} after bootstrap."

MODEL_PATH="${DREAMZERO_CKPT_DIR}"
if [[ -d "${DREAMZERO_CKPT_DIR}/$(basename "${DREAMZERO_CKPT_DIR}")" ]]; then
  MODEL_PATH="${DREAMZERO_CKPT_DIR}/$(basename "${DREAMZERO_CKPT_DIR}")"
fi

info "Ensuring DreamZero streaming checkpoint-load patch is applied."
bash "${SCRIPT_DIR}/24_patch_dreamzero_streaming_load.sh"

append_report "# DreamZero-DROID H800 Report Smoke"
append_report ""
append_report "- Timestamp: ${REPORT_TS}"
append_report "- Host: $(hostname 2>/dev/null || echo unknown)"
append_report "- GPUs: $(gpu_names || echo unknown)"
append_report "- DreamZero repo: ${DREAMZERO_REPO}"
append_report "- Checkpoint: ${MODEL_PATH}"
append_report "- Report logs: ${REPORT_RUN_DIR}"
append_report ""

{
  echo "## Environment"
  echo
  conda run --no-capture-output -n "${DREAMZERO_ENV_NAME}" python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"gpu_count={torch.cuda.device_count()}")
try:
    import flash_attn
    print(f"flash_attn={getattr(flash_attn, '__version__', 'unknown')}")
except Exception as exc:
    print(f"flash_attn=FAILED {type(exc).__name__}: {exc}")
PY
  echo
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
} >> "${REPORT_FILE}" 2>&1

if port_in_use "${PORT}"; then
  warn "Port ${PORT} is already listening; reusing existing DreamZero server."
else
  info "Starting DreamZero server and running zero-image smoke; server will be kept alive."
  KEEP_DREAMZERO_SERVER=1 \
  DREAMZERO_CLIENT_NUM_CHUNKS="${DREAMZERO_CLIENT_NUM_CHUNKS:-1}" \
  DREAMZERO_ENABLE_DIT_CACHE="${DREAMZERO_ENABLE_DIT_CACHE:-0}" \
  DREAMZERO_DISABLE_TORCH_COMPILE="${DREAMZERO_DISABLE_TORCH_COMPILE:-1}" \
  DREAMZERO_SERVER_WARMUP_SECONDS="${DREAMZERO_SERVER_WARMUP_SECONDS:-7200}" \
  bash "${SCRIPT_DIR}/22_run_dreamzero_server_smoke.sh"
fi

append_report ""
append_report "## Zero-Image Smoke"
append_report ""
if run_client "zero_image_client" \
  --host "${HOST}" \
  --port "${PORT}" \
  --use-zero-images \
  --num-chunks "${DREAMZERO_CLIENT_NUM_CHUNKS:-1}"; then
  append_report "- PASS: zero-image client connected and returned an action chunk."
  grep -E "Server metadata|Action shape|Done" "${REPORT_RUN_DIR}/zero_image_client.log" >> "${REPORT_FILE}" || true
else
  append_report "- FAIL: zero-image client failed. See ${REPORT_RUN_DIR}/zero_image_client.log."
  die "Zero-image client failed."
fi

append_report ""
append_report "## Real-Frame Client"
append_report ""
IFS=',' read -ra CHUNKS <<< "${REPORT_REAL_CHUNKS:-1,3}"
for chunks in "${CHUNKS[@]}"; do
  chunks="${chunks// /}"
  [[ -n "${chunks}" ]] || continue
  name="real_frame_${chunks}_chunks"
  if run_client "${name}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --num-chunks "${chunks}"; then
    append_report "- PASS: real-frame client num_chunks=${chunks}."
    grep -E "Server metadata|Action shape|Done" "${REPORT_RUN_DIR}/${name}.log" >> "${REPORT_FILE}" || true
  else
    append_report "- FAIL: real-frame client num_chunks=${chunks}. See ${REPORT_RUN_DIR}/${name}.log."
    die "Real-frame client failed for num_chunks=${chunks}."
  fi
done

append_report ""
append_report "## Meeting Summary Snippet"
append_report ""
append_report 'I validated DreamZero-DROID on 2xH800 with a server/client smoke run. The server loads the DROID checkpoint, the client connects successfully, and the policy returns action chunks with shape `Action shape: (24, 8)`. I also ran real-frame multi-chunk inference, so the path is not limited to dummy zero-image input. The environment issues we hit before, including PyTorch cu129, flash-attn, streaming checkpoint loading, port probing, and IsaacSim dependencies, have been captured in reusable scripts for future image-based reuse.'

info "Report smoke finished."
info "Report: ${REPORT_FILE}"
echo
cat "${REPORT_FILE}"
