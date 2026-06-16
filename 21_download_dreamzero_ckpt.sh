#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
start_log "download_dreamzero_ckpt"

mkdir -p "${HF_HOME}" "$(dirname "${DREAMZERO_CKPT_DIR}")" "$(dirname "${DREAMZERO_AGIBOT_CKPT_DIR}")"
export HF_HOME HF_TOKEN

if [[ -z "${HF_TOKEN:-}" ]]; then
  warn "HF_TOKEN is empty. If the model or assets are gated/private, run: export HF_TOKEN=... or edit config.env."
fi

activate_env "${DREAMZERO_ENV_NAME}"
if ! command -v hf >/dev/null 2>&1; then
  python -m pip install "huggingface_hub[cli]"
fi

TARGETS="${DREAMZERO_CKPT_TARGETS:-DROID}"
IFS=',' read -ra ITEMS <<< "${TARGETS}"
for item in "${ITEMS[@]}"; do
  case "${item}" in
    DROID|droid)
      require_free_gb "$(dirname "${DREAMZERO_CKPT_DIR}")" 80
      info "Downloading ${DREAMZERO_DROID_MODEL_ID} to ${DREAMZERO_CKPT_DIR}"
      hf download "${DREAMZERO_DROID_MODEL_ID}" --repo-type model --local-dir "${DREAMZERO_CKPT_DIR}" \
        2>&1 | tee -a "${LOG_DIR}/dreamzero_droid_download_$(timestamp).log"
      ;;
    AGIBOT|AgiBot|agibot)
      require_free_gb "$(dirname "${DREAMZERO_AGIBOT_CKPT_DIR}")" 80
      info "Downloading ${DREAMZERO_AGIBOT_MODEL_ID} to ${DREAMZERO_AGIBOT_CKPT_DIR}"
      hf download "${DREAMZERO_AGIBOT_MODEL_ID}" --repo-type model --local-dir "${DREAMZERO_AGIBOT_CKPT_DIR}" \
        2>&1 | tee -a "${LOG_DIR}/dreamzero_agibot_download_$(timestamp).log"
      ;;
    *)
      die "Unknown DREAMZERO_CKPT_TARGETS entry: ${item}. Use DROID, AGIBOT, or DROID,AGIBOT."
      ;;
  esac
done

info "DreamZero checkpoint download finished."
