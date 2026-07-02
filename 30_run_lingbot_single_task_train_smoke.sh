#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
start_log "lingbot_single_task_train_smoke"

info "LingBot-VA single-task fine-tune smoke."
info "This only checks whether the training path can load one prepared dataset and run a few optimizer steps."

confirm_install_or_exit

require_dir "${LINGBOT_REPO}" "LingBot-VA repo"
require_dir "${LINGBOT_CKPT_DIR}" "LingBot checkpoint"

LINGBOT_TRAIN_DATASET_DIR="${LINGBOT_TRAIN_DATASET_DIR:-}"
[[ -n "${LINGBOT_TRAIN_DATASET_DIR}" ]] || die "Set LINGBOT_TRAIN_DATASET_DIR to a prepared LingBot/LeRobot training dataset directory."
require_dir "${LINGBOT_TRAIN_DATASET_DIR}" "LingBot train dataset"

SMOKE_STEPS="${LINGBOT_TRAIN_STEPS:-2}"
SMOKE_NGPU="${LINGBOT_TRAIN_NGPU:-1}"
SMOKE_MASTER_PORT="${LINGBOT_TRAIN_MASTER_PORT:-29631}"
SMOKE_CONFIG="${LINGBOT_TRAIN_CONFIG:-libero_train}"
SMOKE_SAVE_ROOT="${LINGBOT_TRAIN_SAVE_ROOT:-${PROJECT_ROOT}/outputs/lingbot_train_smoke_$(timestamp)}"

if [[ "${SMOKE_CONFIG}" != "libero_train" && "${SMOKE_CONFIG}" != "robotwin_train" && "${SMOKE_CONFIG}" != "demo_train" ]]; then
  die "Unsupported LINGBOT_TRAIN_CONFIG=${SMOKE_CONFIG}. Use libero_train, robotwin_train, or demo_train."
fi

INFO_COUNT="$(find "${LINGBOT_TRAIN_DATASET_DIR}" -path "*/meta/info.json" -type f | wc -l | tr -d ' ')"
[[ "${INFO_COUNT}" != "0" ]] || die "No LeRobot meta/info.json found under ${LINGBOT_TRAIN_DATASET_DIR}."
[[ -d "${LINGBOT_TRAIN_DATASET_DIR}/latents" ]] || die "Missing ${LINGBOT_TRAIN_DATASET_DIR}/latents. LingBot training needs pre-extracted VAE latents, not only raw videos."
[[ -f "${LINGBOT_TRAIN_DATASET_DIR}/empty_emb.pt" ]] || die "Missing ${LINGBOT_TRAIN_DATASET_DIR}/empty_emb.pt. Generate/copy it before training."

info "Dataset root: ${LINGBOT_TRAIN_DATASET_DIR}"
info "Found ${INFO_COUNT} LeRobot dataset(s) under dataset root."
info "Config: ${SMOKE_CONFIG}; steps: ${SMOKE_STEPS}; GPUs: ${SMOKE_NGPU}"
info "Output: ${SMOKE_SAVE_ROOT}"

init_conda
if ! conda_env_exists "${LINGBOT_ENV_NAME}" 2>/dev/null; then
  info "LingBot env ${LINGBOT_ENV_NAME} does not exist; running setup first."
  CONFIRM_INSTALL=1 SKIP_FLASH_ATTN="${SKIP_FLASH_ATTN:-1}" DOWNLOAD_LINGBOT_CKPT="${DOWNLOAD_LINGBOT_CKPT:-0}" \
    bash "${SCRIPT_DIR}/10_setup_lingbot_va.sh"
fi

activate_env "${LINGBOT_ENV_NAME}"
ENV_PREFIX="$(conda env list | awk -v n="${LINGBOT_ENV_NAME}" '$1 == n {print $NF; exit}')"
if [[ -n "${ENV_PREFIX}" && -d "${ENV_PREFIX}/bin" ]]; then
  export CONDA_PREFIX="${ENV_PREFIX}"
  export PATH="${ENV_PREFIX}/bin:${PATH}"
  hash -r
fi
info "Python: $(which python)"

info "Installing LingBot post-training dependencies."
PIP_CONSTRAINT= PIP_CONSTRAINTS= python -m pip install \
  scipy wandb lerobot==0.3.3 datasets==3.6.0 "dill<0.3.9" "multiprocess<0.70.17" \
  "packaging>=24.2" jsonlines av \
  2>&1 | tee -a "${LOG_DIR}/lingbot_train_deps_$(timestamp).log" || {
    warn "Direct lerobot==0.3.3 install failed. Trying official PyPI without the current mirror."
    PIP_CONSTRAINT= PIP_CONSTRAINTS= python -m pip install \
      scipy wandb lerobot==0.3.3 datasets==3.6.0 "dill<0.3.9" "multiprocess<0.70.17" \
      "packaging>=24.2" jsonlines av \
      -i https://pypi.org/simple \
      2>&1 | tee -a "${LOG_DIR}/lingbot_train_deps_pypi_$(timestamp).log"
  }

info "Patching LingBot train config for a tiny local smoke."
LINGBOT_REPO="${LINGBOT_REPO}" \
LINGBOT_CKPT_DIR="${LINGBOT_CKPT_DIR}" \
LINGBOT_TRAIN_DATASET_DIR="${LINGBOT_TRAIN_DATASET_DIR}" \
SMOKE_STEPS="${SMOKE_STEPS}" \
SMOKE_SAVE_ROOT="${SMOKE_SAVE_ROOT}" \
SMOKE_CONFIG="${SMOKE_CONFIG}" \
python - <<'PY'
import json
import os
from pathlib import Path

repo = Path(os.environ["LINGBOT_REPO"])
ckpt = Path(os.environ["LINGBOT_CKPT_DIR"])
dataset = Path(os.environ["LINGBOT_TRAIN_DATASET_DIR"])
steps = int(os.environ["SMOKE_STEPS"])
save_root = Path(os.environ["SMOKE_SAVE_ROOT"])
config_name = os.environ["SMOKE_CONFIG"]

cfg_map = {
    "libero_train": repo / "wan_va" / "configs" / "va_libero_train_cfg.py",
    "robotwin_train": repo / "wan_va" / "configs" / "va_robotwin_train_cfg.py",
    "demo_train": repo / "wan_va" / "configs" / "va_demo_train_cfg.py",
}
cfg_path = cfg_map[config_name]
text = cfg_path.read_text(encoding="utf-8")
backup = cfg_path.with_suffix(cfg_path.suffix + ".bak_train_smoke")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")

append = f'''

# --- Codex H800 train-smoke overrides ---
{Path(cfg_path).stem}.dataset_path = r"{dataset}"
{Path(cfg_path).stem}.empty_emb_path = r"{dataset / "empty_emb.pt"}"
{Path(cfg_path).stem}.enable_wandb = False
{Path(cfg_path).stem}.load_worker = 0
{Path(cfg_path).stem}.save_interval = max(1, {steps})
{Path(cfg_path).stem}.gc_interval = 1
{Path(cfg_path).stem}.batch_size = 1
{Path(cfg_path).stem}.gradient_accumulation_steps = 1
{Path(cfg_path).stem}.num_steps = {steps}
{Path(cfg_path).stem}.save_root = r"{save_root}"
{Path(cfg_path).stem}.wan22_pretrained_model_name_or_path = r"{ckpt}"
# --- end Codex H800 train-smoke overrides ---
'''
marker = "# --- Codex H800 train-smoke overrides ---"
if marker in text:
    text = text.split(marker)[0].rstrip() + append
else:
    text = text.rstrip() + append
cfg_path.write_text(text + "\n", encoding="utf-8")
print(f"patched {cfg_path}")

t_cfg = ckpt / "transformer" / "config.json"
if not t_cfg.exists():
    raise SystemExit(f"missing transformer config: {t_cfg}")
data = json.loads(t_cfg.read_text(encoding="utf-8"))
old = data.get("attn_mode")
data["attn_mode"] = "flex"
t_cfg.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
print(f"set {t_cfg} attn_mode: {old!r} -> 'flex'")
PY

info "Patching LingBot/LeRobot compatibility helpers."
LINGBOT_REPO="${LINGBOT_REPO}" python - <<'PY'
import os
from pathlib import Path

repo = Path(os.environ["LINGBOT_REPO"])
p = repo / "wan_va" / "dataset" / "lerobot_latent_dataset.py"
if not p.exists():
    raise SystemExit(f"missing {p}")
s = p.read_text(encoding="utf-8")
start = "# --- Codex compatibility: lerobot local-path safe version ---"
old_start = "# --- Codex compatibility: lerobot>=0.3 may not expose get_safe_version here ---"
end = "# --- end Codex compatibility ---"
insert = '''

# --- Codex compatibility: lerobot local-path safe version ---
try:
    from lerobot.datasets.utils import get_safe_version as _hub_get_safe_version  # type: ignore
except Exception:
    _hub_get_safe_version = None

def get_safe_version(repo_id, revision=None):
    from pathlib import Path as _Path
    if isinstance(repo_id, (str, bytes)) and _Path(str(repo_id)).exists():
        return revision
    if _hub_get_safe_version is None:
        return revision
    return _hub_get_safe_version(repo_id, revision)
# --- end Codex compatibility ---
'''
for marker in (start, old_start):
    while marker in s and end in s.split(marker, 1)[1]:
        before = s.split(marker, 1)[0].rstrip()
        after = s.split(marker, 1)[1].split(end, 1)[1].lstrip()
        s = before + "\n" + after

lines = s.splitlines()
last_import = 0
for i, line in enumerate(lines):
    if line.startswith("import ") or line.startswith("from "):
        last_import = i + 1
lines.insert(last_import, insert)
p.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"patched local-path compatibility in {p}")
PY

info "Checking local LeRobot files before dataset construction."
LINGBOT_TRAIN_DATASET_DIR="${LINGBOT_TRAIN_DATASET_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["LINGBOT_TRAIN_DATASET_DIR"])
info = root / "meta" / "info.json"
if not info.exists():
    raise SystemExit(f"missing {info}")
data_files = sorted(root.glob("data/**/*.parquet"))
video_files = sorted(root.glob("videos/**/*"))
latent_files = sorted(root.glob("latents/**/*"))
latent_files = [p for p in latent_files if p.is_file()]
print("dataset_root", root)
print("meta/info.json OK")
print("data parquet files", len(data_files))
print("video files", len([p for p in video_files if p.is_file()]))
print("latent files", len(latent_files))
if data_files:
    print("first data file", data_files[0])
if latent_files:
    print("first latent file", latent_files[0])
if not data_files:
    raise SystemExit(
        "No data/**/*.parquet files found. The LeRobot dataset download is incomplete; "
        "re-run 09_setup_and_run_lingbot_libero_long.sh with RUN_LINGBOT_LIBERO_LONG_SMOKE=0."
    )

info_data = json.loads(info.read_text(encoding="utf-8"))
total = int(info_data.get("total_episodes") or info_data.get("total_episodes_count") or 0)
chunks_size = int(info_data.get("chunks_size") or info_data.get("chunk_size") or 1000)
data_path = info_data.get("data_path")
if total and data_path:
    missing = []
    for episode_index in range(total):
        episode_chunk = episode_index // chunks_size
        try:
            rel = data_path.format(episode_chunk=episode_chunk, episode_index=episode_index)
        except Exception:
            break
        if not (root / rel).is_file():
            missing.append(rel)
            if len(missing) >= 20:
                break
    if missing:
        print("expected total episodes", total)
        print("data_path template", data_path)
        print("missing data files sample:")
        for rel in missing:
            print("  ", rel)
        raise SystemExit(
            "LeRobot metadata points to missing parquet files. Dataset is incomplete or laid out differently; "
            "re-run 09_setup_and_run_lingbot_libero_long.sh with RUN_LINGBOT_LIBERO_LONG_SMOKE=0."
        )
PY

info "Checking train imports and dataset construction before launching distributed training."
cd "${LINGBOT_REPO}"
SMOKE_CONFIG="${SMOKE_CONFIG}" PYTHONPATH="${LINGBOT_REPO}:${PYTHONPATH:-}" python - <<'PY'
import torch
import os
from wan_va.configs import VA_CONFIGS
from wan_va.dataset import MultiLatentLeRobotDataset

config_name = os.environ["SMOKE_CONFIG"]
cfg = VA_CONFIGS[config_name]
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("config", config_name)
print("dataset_path", cfg.dataset_path)
print("empty_emb_path", cfg.empty_emb_path)
ds = MultiLatentLeRobotDataset(cfg, num_init_worker=1)
print("dataset_len", len(ds))
sample = ds[0]
for key, value in sample.items():
    shape = tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
    print(key, shape)
PY

info "Launching LingBot tiny train smoke."
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
python -m torch.distributed.run \
  --nproc_per_node="${SMOKE_NGPU}" \
  --local-ranks-filter=0 \
  --master_port "${SMOKE_MASTER_PORT}" \
  --tee 3 \
  -m wan_va.train --config-name "${SMOKE_CONFIG}" --save-root "${SMOKE_SAVE_ROOT}"

info "LingBot train smoke finished."
info "Output: ${SMOKE_SAVE_ROOT}"
warn "For inference/eval after training, reset ${LINGBOT_CKPT_DIR}/transformer/config.json attn_mode to torch or flashattn."
