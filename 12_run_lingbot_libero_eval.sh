#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
start_log "lingbot_libero_eval"

require_dir "${LINGBOT_REPO}" "LingBot-VA repo"
require_dir "${LINGBOT_CKPT_DIR}" "LingBot checkpoint dir"
init_conda
conda_env_exists "${LINGBOT_ENV_NAME}" || die "Missing server env ${LINGBOT_ENV_NAME}. Run 10_setup_lingbot_va.sh first."
LIBERO_CLIENT_ENV_NAME="${LIBERO_CLIENT_ENV_NAME:-${LIBERO_ENV_NAME}}"
conda_env_exists "${LIBERO_CLIENT_ENV_NAME}" || die "Missing client env ${LIBERO_CLIENT_ENV_NAME}. Run CONFIRM_INSTALL=1 bash 13_setup_libero_env.sh first."

cd "${LINGBOT_REPO}"
export PYTHONPATH="${LINGBOT_REPO}:${PYTHONPATH:-}"
conda run -n "${LIBERO_CLIENT_ENV_NAME}" python - <<PY
import sys
sys.path.insert(0, "${LINGBOT_REPO}")
sys.path.insert(0, "${LIBERO_REPO}")
import libero, robosuite
print("LIBERO client imports OK")
PY

PORT="${LINGBOT_PORT:-29056}"
MASTER_PORT="${LINGBOT_MASTER_PORT:-29061}"
RUN_DIR="${LOG_DIR}/lingbot_libero_$(timestamp)"
mkdir -p "${RUN_DIR}"
SERVER_LOG="${RUN_DIR}/server.log"
CLIENT_LOG="${RUN_DIR}/client.log"
PID_FILE="${RUN_DIR}/server.pid"
RUNNER="${RUN_DIR}/lingbot_libero_server_runner.py"

if port_in_use "${PORT}"; then
  die "Port ${PORT} is already in use. Stop old server or change LINGBOT_PORT in config.env."
fi

info "Starting LingBot LIBERO server in background."
cat > "${RUNNER}" <<'PY'
import os
from types import SimpleNamespace

from wan_va.configs import VA_CONFIGS
import wan_va.wan_va_server as server

cfg = server.VA_CONFIGS["libero"]
cfg.wan22_pretrained_model_name_or_path = os.environ["LINGBOT_CKPT_DIR"]
cfg.enable_offload = os.environ.get("LINGBOT_ENABLE_OFFLOAD", "1") == "1"
cfg.num_inference_steps = int(os.environ.get("LINGBOT_VIDEO_STEPS", str(getattr(cfg, "num_inference_steps", 20))))
cfg.action_num_inference_steps = int(os.environ.get("LINGBOT_ACTION_STEPS", str(getattr(cfg, "action_num_inference_steps", 50))))
VA_CONFIGS["libero"] = cfg
server.VA_CONFIGS["libero"] = cfg
print("effective config: libero")
print("effective checkpoint:", cfg.wan22_pretrained_model_name_or_path)

args = SimpleNamespace(config_name="libero", port=int(os.environ["LINGBOT_PORT"]), save_root=os.environ["LINGBOT_LIBERO_SAVE_ROOT"])
server.init_logger()
server.run(args)
PY

(
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
  export TOKENIZERS_PARALLELISM=false
  export PYTHONPATH="${LINGBOT_REPO}:${PYTHONPATH:-}"
  export LINGBOT_CKPT_DIR LINGBOT_PORT
  export LINGBOT_LIBERO_SAVE_ROOT="${RUN_DIR}/visualization"
  conda run --no-capture-output -n "${LINGBOT_ENV_NAME}" python -u -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port "${MASTER_PORT}" \
    "${RUNNER}"
) >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "${SERVER_PID}" > "${PID_FILE}"
info "Server PID ${SERVER_PID}; log ${SERVER_LOG}"

info "Waiting for server port ${PORT} to listen."
SERVER_WAIT_SECONDS="${LINGBOT_SERVER_WARMUP_SECONDS:-300}"
START_TS="$(date +%s)"
while true; do
  if port_in_use "${PORT}"; then
    info "Server port ${PORT} is listening."
    break
  fi
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    diagnose_log "${SERVER_LOG}"
    die "LingBot server exited before opening port ${PORT}."
  fi
  NOW_TS="$(date +%s)"
  if (( NOW_TS - START_TS > SERVER_WAIT_SECONDS )); then
    echo "===== server log tail ====="
    tail -n 120 "${SERVER_LOG}" || true
    die "Timed out waiting for server port ${PORT}. PID file: ${PID_FILE}"
  fi
  sleep 5
done

info "Running official LIBERO client."
set +e
MUJOCO_GL="${MUJOCO_GL:-egl}" \
PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" \
LINGBOT_SERVER_HOST="${LINGBOT_SERVER_HOST:-127.0.0.1}" \
PYTHONPATH="${LINGBOT_REPO}:${LIBERO_REPO}:${PYTHONPATH:-}" \
conda run --no-capture-output -n "${LIBERO_CLIENT_ENV_NAME}" python -u evaluation/libero/client.py \
  --libero-benchmark "${LINGBOT_LIBERO_BENCHMARK:-libero_10}" \
  --port "${PORT}" \
  --test-num "${LINGBOT_LIBERO_TEST_NUM:-2}" \
  --task-range "${LINGBOT_LIBERO_START:-0}" "${LINGBOT_LIBERO_END:-1}" \
  --out-dir "${RUN_DIR}/outputs" \
  >"${CLIENT_LOG}" 2>&1
STATUS=$?
set -e

if [[ "${STATUS}" -ne 0 ]]; then
  diagnose_log "${SERVER_LOG}"
  diagnose_log "${CLIENT_LOG}"
  die "LingBot LIBERO eval failed. Stop server with bash stop_lingbot_server.sh ${PID_FILE}"
fi

info "LingBot LIBERO eval completed. Stop server with: bash ${SCRIPT_DIR}/stop_lingbot_server.sh ${PID_FILE}"
