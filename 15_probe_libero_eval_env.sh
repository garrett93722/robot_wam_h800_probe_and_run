#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
start_log "probe_libero_eval_env"

info "This is a lightweight LIBERO evaluation probe. It does not install packages and does not start LingBot inference."

require_dir "${LINGBOT_REPO}" "LingBot-VA repo"
require_dir "${LIBERO_REPO}" "LIBERO repo"
info "Project root: ${PROJECT_ROOT}"
info "LingBot repo: ${LINGBOT_REPO}"
info "LIBERO repo: ${LIBERO_REPO}"
info "LIBERO data dir: ${LIBERO_DATA_DIR}"

section() {
  echo
  echo "===== $* ====="
}

section "basic files"
for path in \
  "${LINGBOT_REPO}/evaluation/libero/client.py" \
  "${LINGBOT_REPO}/evaluation/libero/launch_server.sh" \
  "${LINGBOT_REPO}/evaluation/libero/launch_client.sh" \
  "${LIBERO_REPO}/requirements.txt" \
  "${LIBERO_REPO}/setup.py" \
  "${LIBERO_REPO}/benchmark_scripts/download_libero_datasets.py"; do
  if [[ -e "${path}" ]]; then
    echo "OK ${path}"
  else
    echo "MISSING ${path}"
  fi
done

section "ports"
for port in "${LINGBOT_PORT:-29056}" "${LINGBOT_MASTER_PORT:-29061}"; do
  if port_in_use "${port}"; then
    echo "IN_USE port ${port}"
  else
    echo "FREE port ${port}"
  fi
done

section "rendering environment"
echo "MUJOCO_GL=${MUJOCO_GL:-}"
echo "PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-}"
echo "DISPLAY=${DISPLAY:-}"
if [[ "${MUJOCO_GL:-}" != "egl" ]]; then
  echo "SUGGEST export MUJOCO_GL=egl"
fi
if [[ "${PYOPENGL_PLATFORM:-}" != "egl" ]]; then
  echo "SUGGEST export PYOPENGL_PLATFORM=egl"
fi

section "conda envs"
init_conda
conda env list

probe_env_imports() {
  local env_name="$1"
  echo
  echo "--- probing conda env: ${env_name} ---"
  if ! conda_env_exists "${env_name}"; then
    echo "MISSING conda env ${env_name}"
    return 0
  fi
  local probe_py="${LOG_DIR}/probe_imports_${env_name}_$(timestamp).py"
  cat > "${probe_py}" <<'PY'
import importlib
mods = ["torch", "numpy", "cv2", "mujoco", "OpenGL", "robosuite", "libero", "websockets", "msgpack"]
for mod in mods:
    try:
        m = importlib.import_module(mod)
        print(f"{mod}=OK version={getattr(m, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{mod}=FAILED {type(exc).__name__}: {exc}")
PY
  conda run --no-capture-output -n "${env_name}" python "${probe_py}" || true
}

section "python imports"
probe_env_imports "${LINGBOT_ENV_NAME}"
probe_env_imports "${LIBERO_ENV_NAME}"

section "libero task metadata"
if conda_env_exists "${LIBERO_ENV_NAME}"; then
  META_PROBE_PY="${LOG_DIR}/probe_libero_metadata_$(timestamp).py"
  cat > "${META_PROBE_PY}" <<'PY'
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
lingbot = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repo))
sys.path.insert(0, str(lingbot))
try:
    from libero.libero import benchmark
    import importlib.util
    client_path = lingbot / "evaluation" / "libero" / "client.py"
    spec = importlib.util.spec_from_file_location("lingbot_libero_client_probe", client_path)
    client_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client_mod)
    d = benchmark.get_benchmark_dict()
    print("benchmarks:", sorted(d.keys()))
    suite = d["libero_10"]()
    task = suite.get_task(0)
    print("libero_10 task0:", task.name)
    print("language:", task.language)
    print("websocket client import OK:", client_mod.WebsocketClientPolicy)
except Exception as exc:
    print(f"LIBERO metadata probe FAILED {type(exc).__name__}: {exc}")
PY
  MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" \
  PYTHONPATH="${LINGBOT_REPO}:${LIBERO_REPO}:${PYTHONPATH:-}" \
  conda run --no-capture-output -n "${LIBERO_ENV_NAME}" python "${META_PROBE_PY}" "${LIBERO_REPO}" "${LINGBOT_REPO}" || true
else
  echo "SKIP metadata probe because ${LIBERO_ENV_NAME} env is missing."
fi

section "recommendation"
if ! conda_env_exists "${LIBERO_ENV_NAME}"; then
  echo "NEXT: install native LIBERO env after smoke is stable:"
  echo "  CONFIRM_INSTALL=1 bash 13_setup_libero_env.sh"
fi
echo "Before real eval, use:"
echo "  export MUJOCO_GL=egl"
echo "  export PYOPENGL_PLATFORM=egl"
echo "  LINGBOT_LIBERO_TEST_NUM=1 LINGBOT_LIBERO_START=0 LINGBOT_LIBERO_END=1 bash 12_run_lingbot_libero_eval.sh"
