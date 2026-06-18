#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
start_log "dreamzero_droid_sim_eval"

require_dir "${DREAMZERO_REPO}" "DreamZero repo"

SIM_EVALS_DIR="${SIM_EVALS_DIR:-${PROJECT_ROOT}/sources/sim-evals}"
API_HOST="${DREAMZERO_API_HOST:-127.0.0.1}"
API_PORT="${DREAMZERO_API_PORT:-5000}"

cat <<EOF
DreamZero DROID sim eval needs:
- sim-evals cloned at: ${SIM_EVALS_DIR}
- uv environment synced
- DROID sim assets downloaded
- A reachable policy API/server at ${API_HOST}:${API_PORT}

If using NVIDIA hosted policy, set DREAMZERO_API_HOST and DREAMZERO_API_PORT in config.env.
If using local server, start 22_run_dreamzero_server_smoke.sh with KEEP_DREAMZERO_SERVER=1 first.
EOF

confirm_install_or_exit
require_free_gb "$(dirname "${SIM_EVALS_DIR}")" "${DROID_SIM_MIN_FREE_GB:-120}"

if [[ ! -d "${SIM_EVALS_DIR}/.git" ]]; then
  info "Cloning sim-evals."
  git clone --recurse-submodules https://github.com/arhanjain/sim-evals.git "${SIM_EVALS_DIR}"
else
  info "sim-evals already cloned; updating submodules."
  git -C "${SIM_EVALS_DIR}" submodule update --init --recursive
fi

if ! command -v uv >/dev/null 2>&1; then
  if [[ "${INSTALL_UV:-0}" == "1" ]]; then
    info "Installing uv without sudo."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  else
    die "uv not found. Install uv or rerun with INSTALL_UV=1 CONFIRM_INSTALL=1."
  fi
fi

cd "${SIM_EVALS_DIR}"
if [[ "${PATCH_SIM_EVALS_UV_BUILD_DEPS:-1}" == "1" && -f "pyproject.toml" ]]; then
  info "Patching sim-evals uv build deps for flatdict/pkg_resources compatibility."
  python - "pyproject.toml" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
lines = text.splitlines()
header = "[tool.uv.extra-build-dependencies]"
if header not in lines:
    if lines and lines[-1].strip():
        lines.append("")
    lines.extend([header, 'flatdict = ["pkg_resources"]'])
else:
    idx = lines.index(header) + 1
    end = idx
    while end < len(lines) and not lines[end].startswith("["):
        end += 1
    section = "\n".join(lines[idx:end])
    if "flatdict" not in section:
        lines.insert(end, 'flatdict = ["pkg_resources"]')
    else:
        lines[idx:end] = [
            'flatdict = ["pkg_resources"]' if line.strip().startswith("flatdict") else line
            for line in lines[idx:end]
        ]
path.write_text("\n".join(lines) + "\n")
PY
fi
info "Syncing sim-evals uv environment."
uv sync 2>&1 | tee -a "${LOG_DIR}/sim_evals_uv_sync_$(timestamp).log"

if [[ "${DOWNLOAD_DROID_SIM_ASSETS:-0}" == "1" ]]; then
  if [[ -z "${HF_TOKEN:-}" ]]; then
    warn "HF_TOKEN is empty. DROID sim asset download may fail if gated."
  fi
  export HF_TOKEN HF_HOME
  info "Downloading DROID sim assets."
  uvx hf download owhan/DROID-sim-environments --repo-type dataset --local-dir assets \
    2>&1 | tee -a "${LOG_DIR}/sim_evals_assets_$(timestamp).log"
else
  warn "Asset download skipped. Set DOWNLOAD_DROID_SIM_ASSETS=1 after confirming disk and HF access."
fi

cd "${DREAMZERO_REPO}"
EVAL_LOG="${LOG_DIR}/dreamzero_droid_sim_eval_$(timestamp).log"
info "Running DreamZero sim eval against ${API_HOST}:${API_PORT}"
set +e
python eval_utils/run_sim_eval.py --host "${API_HOST}" --port "${API_PORT}" 2>&1 | tee "${EVAL_LOG}"
STATUS=${PIPESTATUS[0]}
set -e

if [[ "${STATUS}" -ne 0 ]]; then
  diagnose_log "${EVAL_LOG}"
  die "DreamZero DROID sim eval failed."
fi

info "DreamZero DROID sim eval finished. Log: ${EVAL_LOG}"
