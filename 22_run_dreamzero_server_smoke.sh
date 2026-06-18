#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
start_log "dreamzero_server_smoke"

require_dir "${DREAMZERO_REPO}" "DreamZero repo"
require_dir "${DREAMZERO_CKPT_DIR}" "DreamZero checkpoint dir"
init_conda
conda_env_exists "${DREAMZERO_ENV_NAME}" || die "Missing conda env ${DREAMZERO_ENV_NAME}. Run CONFIRM_INSTALL=1 bash 20_setup_dreamzero.sh first."
CONDA_PY=(conda run --no-capture-output -n "${DREAMZERO_ENV_NAME}" python)

MODEL_PATH="${DREAMZERO_CKPT_DIR}"
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  nested="${MODEL_PATH}/$(basename "${MODEL_PATH}")"
  if [[ -f "${nested}/config.json" ]]; then
    warn "Checkpoint appears nested; using ${nested}"
    MODEL_PATH="${nested}"
  fi
fi
[[ -f "${MODEL_PATH}/config.json" ]] || die "Checkpoint config.json not found under ${DREAMZERO_CKPT_DIR}. Check DREAMZERO_CKPT_DIR in config.env."

cd "${DREAMZERO_REPO}"
PORT="${DREAMZERO_PORT:-5000}"
NPROC_PER_NODE="${DREAMZERO_NPROC_PER_NODE:-2}"
RUN_DIR="${LOG_DIR}/dreamzero_server_$(timestamp)"
mkdir -p "${RUN_DIR}"
SERVER_LOG="${RUN_DIR}/server.log"
CLIENT_LOG="${RUN_DIR}/client.log"
RESOURCE_LOG="${RUN_DIR}/resource.log"
PID_FILE="${RUN_DIR}/server.pid"
SITECUSTOMIZE_DIR="${RUN_DIR}/python_startup"

if port_in_use "${PORT}"; then
  die "Port ${PORT} is already in use. Change DREAMZERO_PORT or stop the old server."
fi

"${CONDA_PY[@]}" - "${NPROC_PER_NODE}" <<'PY'
import sys
import torch
expected = int(sys.argv[1])
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
assert torch.cuda.device_count() >= expected, f"DreamZero smoke expects at least {expected} visible GPU(s)"
import websockets, msgpack_numpy
print("basic imports OK")
PY

{
  echo "===== resource snapshot before server start ====="
  date
  echo "--- free -h ---"
  free -h || true
  echo "--- cgroup memory ---"
  for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.high /sys/fs/cgroup/memory/memory.limit_in_bytes /sys/fs/cgroup/memory/memory.usage_in_bytes; do
    [[ -e "${f}" ]] && printf "%s: %s\n" "${f}" "$(cat "${f}" 2>/dev/null || true)"
  done
  echo "--- nvidia-smi ---"
  nvidia-smi || true
} > "${RESOURCE_LOG}" 2>&1

info "Starting DreamZero distributed server on ${NPROC_PER_NODE} GPU process(es)."
if [[ "${DREAMZERO_DISABLE_TORCH_COMPILE:-1}" == "1" ]]; then
  mkdir -p "${SITECUSTOMIZE_DIR}"
  cat > "${SITECUSTOMIZE_DIR}/sitecustomize.py" <<'PY'
import os

if os.environ.get("DREAMZERO_DISABLE_TORCH_COMPILE", "1") == "1":
    try:
        import torch

        def _identity_compile(fn=None, *args, **kwargs):
            if fn is None:
                return lambda real_fn: real_fn
            return fn

        torch.compile = _identity_compile
        try:
            import torch._dynamo
            torch._dynamo.config.disable = True
        except Exception:
            pass
        print("[sitecustomize] torch.compile disabled for DreamZero smoke", flush=True)
    except Exception as exc:
        print(f"[sitecustomize] failed to disable torch.compile: {exc}", flush=True)
PY
  info "DREAMZERO_DISABLE_TORCH_COMPILE=1; disabling torch.compile for conservative smoke."
else
  info "DREAMZERO_DISABLE_TORCH_COMPILE=${DREAMZERO_DISABLE_TORCH_COMPILE:-0}; torch.compile remains enabled."
fi
SERVER_ARGS=(
  socket_test_optimized_AR.py
  --port "${PORT}"
  --model-path "${MODEL_PATH}"
)
if [[ "${DREAMZERO_ENABLE_DIT_CACHE:-0}" == "1" ]]; then
  warn "DREAMZERO_ENABLE_DIT_CACHE=1; enabling DIT cache. This can increase memory pressure."
  SERVER_ARGS+=(--enable-dit-cache)
else
  info "DREAMZERO_ENABLE_DIT_CACHE=${DREAMZERO_ENABLE_DIT_CACHE:-0}; DIT cache disabled for conservative smoke."
fi
(
  export CUDA_VISIBLE_DEVICES="${DREAMZERO_CUDA_VISIBLE_DEVICES:-0,1}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
  export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-${DREAMZERO_DISABLE_TORCH_COMPILE:-1}}"
  export DREAMZERO_DISABLE_TORCH_COMPILE="${DREAMZERO_DISABLE_TORCH_COMPILE:-1}"
  if [[ "${DREAMZERO_DISABLE_TORCH_COMPILE}" == "1" ]]; then
    export PYTHONPATH="${SITECUSTOMIZE_DIR}:${PYTHONPATH:-}"
  fi
  conda run --no-capture-output -n "${DREAMZERO_ENV_NAME}" python -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    "${SERVER_ARGS[@]}"
) >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "${SERVER_PID}" > "${PID_FILE}"
info "Server PID ${SERVER_PID}; log ${SERVER_LOG}"
info "Resource monitor log ${RESOURCE_LOG}"

(
  while kill -0 "${SERVER_PID}" >/dev/null 2>&1; do
    {
      echo
      echo "===== resource sample $(date '+%F %T') ====="
      free -h || true
      for f in /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.usage_in_bytes /sys/fs/cgroup/memory/memory.limit_in_bytes; do
        [[ -e "${f}" ]] && printf "%s: %s\n" "${f}" "$(cat "${f}" 2>/dev/null || true)"
      done
      nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
      ps -eo pid,ppid,pcpu,pmem,rss,vsz,comm,args | grep -E "socket_test_optimized_AR|torch.distributed.run|python" | grep -v grep || true
    } >> "${RESOURCE_LOG}" 2>&1
    sleep 5
  done
) &
MONITOR_PID=$!

cleanup() {
  if [[ -n "${MONITOR_PID:-}" ]] && kill -0 "${MONITOR_PID}" >/dev/null 2>&1; then
    kill "${MONITOR_PID}" || true
  fi
  if [[ "${KEEP_DREAMZERO_SERVER:-0}" != "1" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    info "Stopping DreamZero server PID ${SERVER_PID}"
    kill "${SERVER_PID}" || true
  fi
}
trap cleanup EXIT

WARMUP_SECONDS="${DREAMZERO_SERVER_WARMUP_SECONDS:-600}"
info "Waiting up to ${WARMUP_SECONDS}s for server port ${PORT} to listen. First load may take several minutes."
ready=0
for ((i=0; i<WARMUP_SECONDS; i+=5)); do
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    diagnose_log "${SERVER_LOG}"
    die "DreamZero server exited before opening port ${PORT}."
  fi
  if port_in_use "${PORT}"; then
    ready=1
    break
  fi
  sleep 5
done
if [[ "${ready}" != "1" ]]; then
  diagnose_log "${SERVER_LOG}"
  die "DreamZero server did not open port ${PORT} within ${WARMUP_SECONDS}s."
fi
info "Server port ${PORT} is listening."

info "Running DreamZero test client."
set +e
conda run --no-capture-output -n "${DREAMZERO_ENV_NAME}" python test_client_AR.py --port "${PORT}" >"${CLIENT_LOG}" 2>&1
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
