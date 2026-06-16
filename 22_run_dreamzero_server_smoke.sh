#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
start_log "dreamzero_server_smoke"

require_dir "${DREAMZERO_REPO}" "DreamZero repo"
require_dir "${DREAMZERO_CKPT_DIR}" "DreamZero checkpoint dir"
activate_env "${DREAMZERO_ENV_NAME}"

cd "${DREAMZERO_REPO}"
PORT="${DREAMZERO_PORT:-5000}"
RUN_DIR="${LOG_DIR}/dreamzero_server_$(timestamp)"
mkdir -p "${RUN_DIR}"
SERVER_LOG="${RUN_DIR}/server.log"
CLIENT_LOG="${RUN_DIR}/client.log"
PID_FILE="${RUN_DIR}/server.pid"

if port_in_use "${PORT}"; then
  die "Port ${PORT} is already in use. Change DREAMZERO_PORT or stop the old server."
fi

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
assert torch.cuda.device_count() >= 2, "DreamZero smoke expects at least 2 visible GPUs"
import websockets, msgpack_numpy
print("basic imports OK")
PY

info "Starting DreamZero distributed server on 2 GPUs."
(
  export CUDA_VISIBLE_DEVICES="${DREAMZERO_CUDA_VISIBLE_DEVICES:-0,1}"
  python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=2 \
    socket_test_optimized_AR.py \
    --port "${PORT}" \
    --enable-dit-cache \
    --model-path "${DREAMZERO_CKPT_DIR}"
) >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "${SERVER_PID}" > "${PID_FILE}"
info "Server PID ${SERVER_PID}; log ${SERVER_LOG}"

cleanup() {
  if [[ "${KEEP_DREAMZERO_SERVER:-0}" != "1" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    info "Stopping DreamZero server PID ${SERVER_PID}"
    kill "${SERVER_PID}" || true
  fi
}
trap cleanup EXIT

info "Waiting for warmup. First load may take several minutes."
sleep "${DREAMZERO_SERVER_WARMUP_SECONDS:-120}"
if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
  diagnose_log "${SERVER_LOG}"
  die "DreamZero server exited during warmup."
fi

info "Running DreamZero test client."
set +e
python test_client_AR.py --port "${PORT}" >"${CLIENT_LOG}" 2>&1
STATUS=$?
set -e

if [[ "${STATUS}" -ne 0 ]]; then
  diagnose_log "${SERVER_LOG}"
  diagnose_log "${CLIENT_LOG}"
  die "DreamZero client smoke failed. Logs are in ${RUN_DIR}"
fi

info "DreamZero server smoke succeeded. Logs are in ${RUN_DIR}"
if [[ "${KEEP_DREAMZERO_SERVER:-0}" == "1" ]]; then
  info "Server left running. PID file: ${PID_FILE}"
fi
