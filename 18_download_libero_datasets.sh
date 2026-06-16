#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
start_log "download_libero_datasets"

require_dir "${LIBERO_REPO}" "LIBERO repo"
require_free_gb "$(dirname "${LIBERO_DATA_DIR}")" "${LIBERO_MIN_FREE_GB:-80}"

init_conda
conda activate "${LIBERO_ENV_NAME}"

mkdir -p "${LIBERO_DATA_DIR}"
cd "${LIBERO_REPO}"

DATASET_NAME="${LIBERO_DATASET_NAME:-libero_100}"
DATA_SOURCE="${LIBERO_DATA_SOURCE:-box}"

info "LIBERO repo: ${LIBERO_REPO}"
info "Dataset dir: ${LIBERO_DATA_DIR}"
info "Dataset name: ${DATASET_NAME}"
info "Dataset source: ${DATA_SOURCE}"

info "Updating LIBERO config dataset path."
python - "${LIBERO_DATA_DIR}" <<'PY'
import sys
import yaml
from pathlib import Path
from libero.libero import utils as libero_utils

dataset_dir = sys.argv[1]
config_path = Path(libero_utils.config_file)
config_path.parent.mkdir(parents=True, exist_ok=True)
if config_path.exists():
    config = yaml.safe_load(config_path.read_text()) or {}
else:
    config = libero_utils.get_path_dict()
config["datasets"] = dataset_dir
config_path.write_text(yaml.safe_dump(config, sort_keys=False))
print(f"set {config_path} datasets: {dataset_dir}")
PY

if [[ "${DATA_SOURCE}" == "huggingface" ]]; then
  info "Using direct HuggingFace snapshot download instead of LIBERO's old libero_100 allow-pattern."
  python - "${LIBERO_DATA_DIR}" "${DATASET_NAME}" <<'PY' \
    2>&1 | tee -a "${LOG_DIR}/libero_dataset_download_$(timestamp).log"
import sys
from huggingface_hub import snapshot_download

download_dir = sys.argv[1]
dataset_name = sys.argv[2]
repo_id = "yifengzhu-hf/LIBERO-datasets"

if dataset_name == "libero_100":
    # The HF repo stores LIBERO-100 as libero_10 + libero_90.
    # For quick downstream eval, libero_10 is the part the client needs.
    patterns = ["libero_10/*"]
elif dataset_name in {"libero_10", "libero_90", "libero_goal", "libero_object", "libero_spatial"}:
    patterns = [f"{dataset_name}/*"]
else:
    raise SystemExit(f"Unsupported LIBERO_DATASET_NAME for HF direct mode: {dataset_name}")

print(f"repo={repo_id}")
print(f"local_dir={download_dir}")
print(f"allow_patterns={patterns}")
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=download_dir,
    allow_patterns=patterns,
    local_dir_use_symlinks=False,
    resume_download=True,
)
PY
else
  info "Using official LIBERO Box links. For LIBERO-10 eval, download libero_100; it extracts libero_10 and libero_90."
  python benchmark_scripts/download_libero_datasets.py \
    --datasets "${DATASET_NAME}" \
    --download-dir "${LIBERO_DATA_DIR}" \
    2>&1 | tee -a "${LOG_DIR}/libero_dataset_download_$(timestamp).log"
fi

info "Checking downloaded LIBERO datasets."
python - "${LIBERO_DATA_DIR}" <<'PY'
import sys
from libero.libero.utils.download_utils import check_libero_dataset
check_libero_dataset(sys.argv[1])
PY

info "LIBERO dataset download/check finished."
