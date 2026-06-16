#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
start_log "lingbot_va_smoke"

require_dir "${LINGBOT_REPO}" "LingBot-VA repo"
require_dir "${LINGBOT_CKPT_DIR}" "LingBot checkpoint dir"
activate_env "${LINGBOT_ENV_NAME}"

cd "${LINGBOT_REPO}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${LINGBOT_REPO}:${PYTHONPATH:-}"

info "Import test in env ${LINGBOT_ENV_NAME}"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "count", torch.cuda.device_count())
import transformers, diffusers, accelerate, websockets, cv2
from wan_va.configs import VA_CONFIGS
print("available_configs", sorted(VA_CONFIGS.keys()))
PY

if [[ -f "${LINGBOT_CKPT_DIR}/transformer/config.json" ]]; then
  info "Checkpoint config found: ${LINGBOT_CKPT_DIR}/transformer/config.json"
  grep -n '"attn_mode"' "${LINGBOT_CKPT_DIR}/transformer/config.json" || true
else
  warn "No transformer/config.json under checkpoint. Confirm LINGBOT_CKPT_DIR points to the actual model folder."
fi

CONFIG_NAME="${LINGBOT_SMOKE_CONFIG:-libero_i2av}"
SMOKE_LOG="${LOG_DIR}/lingbot_i2av_${CONFIG_NAME}_$(timestamp).log"
RUNNER="${LOG_DIR}/lingbot_i2av_runner_$(timestamp).py"
info "Running official image-to-video-action smoke with one GPU: CONFIG_NAME=${CONFIG_NAME}"
info "If this fails, the log will include next-step hints."

cat > "${RUNNER}" <<'PY'
import os
from types import SimpleNamespace

from wan_va.configs import VA_CONFIGS
import wan_va.wan_va_server as server

config_name = os.environ.get("LINGBOT_SMOKE_CONFIG", "libero_i2av")
cfg = server.VA_CONFIGS[config_name]
cfg.wan22_pretrained_model_name_or_path = os.environ["LINGBOT_CKPT_DIR"]
cfg.num_chunks_to_infer = int(os.environ.get("LINGBOT_NUM_CHUNKS", "1"))
cfg.num_inference_steps = int(os.environ.get("LINGBOT_VIDEO_STEPS", str(getattr(cfg, "num_inference_steps", 5))))
cfg.action_num_inference_steps = int(os.environ.get("LINGBOT_ACTION_STEPS", str(getattr(cfg, "action_num_inference_steps", 10))))
cfg.enable_offload = os.environ.get("LINGBOT_ENABLE_OFFLOAD", "1") == "1"
VA_CONFIGS[config_name] = cfg
server.VA_CONFIGS[config_name] = cfg
print("effective config:", config_name)
print("effective checkpoint:", cfg.wan22_pretrained_model_name_or_path)
print("effective chunks/video_steps/action_steps:", cfg.num_chunks_to_infer, cfg.num_inference_steps, cfg.action_num_inference_steps)

args = SimpleNamespace(config_name=config_name, port=None, save_root=os.environ["LINGBOT_SMOKE_SAVE_ROOT"])
server.init_logger()
server.run(args)
PY

set +e
LINGBOT_CKPT_DIR="${LINGBOT_CKPT_DIR}" \
LINGBOT_SMOKE_CONFIG="${CONFIG_NAME}" \
LINGBOT_NUM_CHUNKS="${LINGBOT_NUM_CHUNKS:-1}" \
LINGBOT_VIDEO_STEPS="${LINGBOT_VIDEO_STEPS:-5}" \
LINGBOT_ACTION_STEPS="${LINGBOT_ACTION_STEPS:-10}" \
LINGBOT_SMOKE_SAVE_ROOT="${LOG_DIR}/lingbot_i2av_outputs" \
PYTHONPATH="${LINGBOT_REPO}:${PYTHONPATH:-}" \
python -m torch.distributed.run \
  --nproc_per_node 1 \
  --master_port "${LINGBOT_MASTER_PORT:-29061}" \
  "${RUNNER}" \
  2>&1 | tee "${SMOKE_LOG}"
STATUS=${PIPESTATUS[0]}
set -e

if [[ "${STATUS}" -ne 0 ]]; then
  diagnose_log "${SMOKE_LOG}"
  die "LingBot-VA smoke failed. Check ${SMOKE_LOG}"
fi

info "LingBot-VA smoke completed. Log: ${SMOKE_LOG}"
