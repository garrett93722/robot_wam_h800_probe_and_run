#!/usr/bin/env bash
set -euo pipefail

# Batch-select and run RoboChallenge Table30 LingBot-VA subtask fine-tune smokes.
#
# Required:
#   ROBOCHALLENGE_RAW_ROOT=/path/to/robochallenge/table30/raw/tasks
#
# Useful knobs:
#   MAX_TASKS=4                 number of tasks to select
#   MAX_EPISODES=10             episodes per task for quick expansion
#   TRAIN_STEPS=50              train steps per selected task
#   FRAME_INTERVAL=5            raw-frame downsample interval, 30fps/5 -> 6fps
#   TABLE30_CODE_REPO=/workspace/sources/lingbot-va-table30
#   LINGBOT_BASE_CKPT=/root/autodl-tmp/checkpoints/lingbot-va-base
#   HF_LEROBOT_HOME=/workspace/data/lerobot
#   TABLE30_PARALLEL=2          number of concurrent tasks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
start_log "table30_subtask_batch"

confirm_install_or_exit
load_config
init_conda

RAW_ROOT="${ROBOCHALLENGE_RAW_ROOT:-}"
[[ -n "${RAW_ROOT}" ]] || die "Set ROBOCHALLENGE_RAW_ROOT to the RoboChallenge raw task root."
require_dir "${RAW_ROOT}" "RoboChallenge raw root"

TABLE30_CODE_REPO="${TABLE30_CODE_REPO:-${SOURCE_ROOT}/lingbot-va-table30}"
TABLE30_REPO_URL="${TABLE30_REPO_URL:-https://github.com/AnonChongqing/lingbot-va-table30.git}"
LINGBOT_BASE_CKPT="${LINGBOT_BASE_CKPT:-/root/autodl-tmp/checkpoints/lingbot-va-base}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${PROJECT_ROOT}/data/lerobot}"
TABLE30_WORK_ROOT="${TABLE30_WORK_ROOT:-${PROJECT_ROOT}/runs/table30_subtasks_$(timestamp)}"
MAX_TASKS="${MAX_TASKS:-4}"
MAX_EPISODES="${MAX_EPISODES:-10}"
MAX_FRAMES_PER_EPISODE="${MAX_FRAMES_PER_EPISODE:-}"
FRAME_INTERVAL="${FRAME_INTERVAL:-5}"
TRAIN_STEPS="${TRAIN_STEPS:-50}"
TABLE30_PARALLEL="${TABLE30_PARALLEL:-2}"
TABLE30_CONDA_ENV="${TABLE30_CONDA_ENV:-}"

mkdir -p "${SOURCE_ROOT}" "${HF_LEROBOT_HOME}" "${TABLE30_WORK_ROOT}"
export HF_LEROBOT_HOME LINGBOT_BASE_CKPT

info "Raw root: ${RAW_ROOT}"
info "Code repo: ${TABLE30_CODE_REPO}"
info "HF_LEROBOT_HOME: ${HF_LEROBOT_HOME}"
info "Work root: ${TABLE30_WORK_ROOT}"
info "MAX_TASKS=${MAX_TASKS}; MAX_EPISODES=${MAX_EPISODES}; TRAIN_STEPS=${TRAIN_STEPS}; PARALLEL=${TABLE30_PARALLEL}"

if [[ ! -d "${TABLE30_CODE_REPO}/.git" ]]; then
  info "Cloning Table30 LingBot code."
  git clone "${TABLE30_REPO_URL}" "${TABLE30_CODE_REPO}"
else
  info "Table30 code repo exists; pulling latest."
  git -C "${TABLE30_CODE_REPO}" pull --ff-only || warn "git pull failed; using existing checkout."
fi

if [[ -n "${TABLE30_CONDA_ENV}" ]]; then
  conda activate "${TABLE30_CONDA_ENV}"
elif conda env list | awk '{print $1" "$NF}' | grep -Eq '(^lingbotva\s|/root/envs/lingbotva$)'; then
  conda activate /root/envs/lingbotva 2>/dev/null || conda activate lingbotva
elif conda_env_exists "${LINGBOT_ENV_NAME}"; then
  conda activate "${LINGBOT_ENV_NAME}"
else
  die "No LingBot conda env found. Set TABLE30_CONDA_ENV, or create /root/envs/lingbotva / ${LINGBOT_ENV_NAME}."
fi

info "Python: $(which python)"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
PY

TASKS_TSV="${TABLE30_WORK_ROOT}/selected_tasks.tsv"
info "Selecting candidate subtasks."
RAW_ROOT="${RAW_ROOT}" MAX_TASKS="${MAX_TASKS}" TASKS_TSV="${TASKS_TSV}" python - <<'PY'
import json
import os
import re
from pathlib import Path

raw_root = Path(os.environ["RAW_ROOT"])
max_tasks = int(os.environ["MAX_TASKS"])
out_path = Path(os.environ["TASKS_TSV"])

clean_words = {
    "clean", "wipe", "remove", "debris", "dirt", "dust", "stain",
    "brush", "sweep", "scrub", "sponge", "cloth", "towel", "lint",
    "roller", "clothing", "garment", "surface",
}
tool_words = {"tool", "brush", "sponge", "cloth", "towel", "roller", "scraper", "gripper"}
simple_words = {"pick", "place", "put", "move", "transfer", "hold", "push"}
hard_words = {"insert", "screw", "pour", "fold", "open", "close", "assemble", "plug", "thread"}

def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return text[:72] or "task"

rows = []
for info_path in raw_root.rglob("meta/task_info.json"):
    task_dir = info_path.parents[1]
    try:
        data = json.loads(info_path.read_text(encoding="utf-8"))
        desc = data.get("task_desc", {})
        prompt = str(desc.get("prompt") or "")
        task_tag = desc.get("task_tag") or []
        robot_tag = task_tag[-1] if task_tag else ""
    except Exception as exc:
        print(f"[skip] {info_path}: {exc}")
        continue
    if not (task_dir / "data").is_dir():
        continue
    text = f"{task_dir.name} {prompt}".lower()
    score = 0
    score += 10 * sum(w in text for w in clean_words)
    score += 4 * sum(w in text for w in tool_words)
    score += 2 * sum(w in text for w in simple_words)
    score -= 8 * sum(w in text for w in hard_words)
    if "lint" in text or "roller" in text:
        score += 5
    if "table" in text:
        score += 1
    if score <= 0:
        continue
    rows.append((score, task_dir.name, prompt, str(task_dir), robot_tag, slugify(task_dir.name)))

rows.sort(key=lambda x: (-x[0], x[1]))
selected = rows[:max_tasks]
if not selected:
    raise SystemExit(f"No candidate tasks found under {raw_root}. Check ROBOCHALLENGE_RAW_ROOT.")

with out_path.open("w", encoding="utf-8") as f:
    f.write("score\ttask_name\tprompt\ttask_dir\trobot_tag\tslug\n")
    for row in selected:
        f.write("\t".join(map(str, row)) + "\n")

print(f"selected {len(selected)} tasks:")
for score, name, prompt, path, robot, slug in selected:
    print(f"- score={score} name={name} robot={robot} slug={slug}")
    print(f"  prompt={prompt}")
PY

cat "${TASKS_TSV}"

add_action_config() {
  local repo_name="$1"
  local prompt="$2"
  python - "${repo_name}" "${prompt}" <<'PY'
import json
import os
import sys
from pathlib import Path

repo_name, prompt = sys.argv[1], sys.argv[2]
root = Path(os.environ["HF_LEROBOT_HOME"]) / repo_name
meta = root / "meta" / "episodes.jsonl"
if not meta.exists():
    raise SystemExit(f"missing episodes.jsonl: {meta}")

lower = prompt.lower()
if any(w in lower for w in ["clean", "wipe", "remove", "debris", "dirt", "dust", "stain", "brush", "sweep", "scrub", "sponge", "cloth", "towel", "lint", "roller"]):
    segments = [
        (0.00, 0.18, "Move to the workspace and stabilize the target object."),
        (0.18, 0.35, "Pick up or position the cleaning tool."),
        (0.35, 0.78, prompt),
        (0.78, 1.00, "Put down the tool and retract the grippers."),
    ]
elif any(w in lower for w in ["pick", "place", "put", "move", "transfer"]):
    segments = [
        (0.00, 0.25, "Move the grippers toward the target object."),
        (0.25, 0.45, "Grasp the target object."),
        (0.45, 0.78, prompt),
        (0.78, 1.00, "Release the object and retract the grippers."),
    ]
else:
    segments = [(0.00, 1.00, prompt)]

def bounds(length):
    vals = [int(round(length * s[0])) for s in segments] + [length]
    vals[0] = 0
    vals[-1] = length
    out = [0]
    for i, value in enumerate(vals[1:], 1):
        if i < len(vals) - 1:
            value = max(out[-1] + 1, min(value, length))
        else:
            value = length
        out.append(value)
    return out

new_lines = []
for line in meta.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    item = json.loads(line)
    length = int(item.get("length", 0))
    b = bounds(length)
    cfg = []
    for i, (_, _, text) in enumerate(segments):
        if b[i + 1] <= b[i]:
            continue
        cfg.append({"start_frame": b[i], "end_frame": b[i + 1], "action_text": text})
    item["action_config"] = cfg
    new_lines.append(json.dumps(item, ensure_ascii=False))
backup = meta.with_suffix(".jsonl.bak_before_batch_action_config")
if not backup.exists():
    backup.write_text(meta.read_text(encoding="utf-8"), encoding="utf-8")
meta.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
print(f"wrote action_config: {meta}")
PY
}

make_task_config() {
  local config_key="$1"
  local var_name="$2"
  local dataset_root="$3"
  local train_steps="$4"
  local cfg_file="${TABLE30_CODE_REPO}/wan_va/configs/va_${config_key}_train_cfg.py"
  local init_file="${TABLE30_CODE_REPO}/wan_va/configs/__init__.py"

  python - "${cfg_file}" "${init_file}" "${config_key}" "${var_name}" "${dataset_root}" "${train_steps}" <<'PY'
import sys
from pathlib import Path

cfg_file = Path(sys.argv[1])
init_file = Path(sys.argv[2])
config_key = sys.argv[3]
var_name = sys.argv[4]
dataset_root = sys.argv[5]
train_steps = int(sys.argv[6])

cfg_text = f'''from easydict import EasyDict
from .va_table30_lint_cfg import va_table30_lint_cfg
import os

{var_name} = EasyDict(__name__='Config: VA {config_key} train')
{var_name}.update(va_table30_lint_cfg)

{var_name}.dataset_path = r'{dataset_root}'
{var_name}.empty_emb_path = os.path.join({var_name}.dataset_path, 'empty_emb.pt')
{var_name}.wan22_pretrained_model_name_or_path = os.environ.get('LINGBOT_BASE_CKPT', r'{Path(dataset_root).parent.parent / "checkpoints" / "lingbot-va-base"}')
{var_name}.enable_wandb = False
{var_name}.num_init_worker = 1
{var_name}.load_worker = 2
{var_name}.save_interval = max(1, {train_steps})
{var_name}.gc_interval = 10
{var_name}.cfg_prob = 0.1

{var_name}.learning_rate = 1e-5
{var_name}.beta1 = 0.9
{var_name}.beta2 = 0.95
{var_name}.weight_decay = 0.1
{var_name}.warmup_steps = 10
{var_name}.batch_size = 1
{var_name}.gradient_accumulation_steps = 4
{var_name}.num_steps = {train_steps}
'''
cfg_file.write_text(cfg_text, encoding='utf-8')

init = init_file.read_text(encoding='utf-8')
imp = f"from .va_{config_key}_train_cfg import {var_name}"
entry = f"    '{config_key}_train': {var_name},"
if imp not in init:
    lines = init.splitlines()
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("from ."):
            insert_at = i + 1
    lines.insert(insert_at, imp)
    init = "\n".join(lines) + "\n"
if entry not in init:
    marker = "VA_CONFIGS = {"
    if marker not in init:
        raise SystemExit("Could not find VA_CONFIGS in __init__.py")
    lines = init.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "}":
            lines.insert(i, entry)
            break
    init = "\n".join(lines) + "\n"
init_file.write_text(init, encoding='utf-8')
print(f"wrote config {config_key}_train -> {cfg_file}")
PY
}

run_one_task() {
  local gpu_id="$1"
  local task_name="$2"
  local prompt="$3"
  local task_dir="$4"
  local slug="$5"

  local repo_name="table30_${slug}_fi${FRAME_INTERVAL}"
  local config_key="table30_${slug}"
  local var_name="va_${config_key}_train_cfg"
  local dataset_root="${HF_LEROBOT_HOME}/${repo_name}"
  local out_dir="${TABLE30_WORK_ROOT}/${slug}"
  local log_file="${out_dir}/run.log"
  mkdir -p "${out_dir}"

  {
    echo "[INFO] task=${task_name}"
    echo "[INFO] prompt=${prompt}"
    echo "[INFO] gpu=${gpu_id}"
    echo "[INFO] repo_name=${repo_name}"
    echo "[INFO] dataset_root=${dataset_root}"

    cd "${TABLE30_CODE_REPO}"
    local convert_args=(
      tools/convert_table30_dualarm_to_lerobot.py
      --repo-name "${repo_name}"
      --raw-task-dir "${task_dir}"
      --frame-interval "${FRAME_INTERVAL}"
      --max-episodes "${MAX_EPISODES}"
      --overwrite
    )
    if [[ -n "${MAX_FRAMES_PER_EPISODE}" ]]; then
      convert_args+=(--max-frames-per-episode "${MAX_FRAMES_PER_EPISODE}")
    fi
    python "${convert_args[@]}"

    add_action_config "${repo_name}" "${prompt}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" python tools/extract_table30_lingbot_latents.py \
      --lerobot-root "${dataset_root}" \
      --raw-task-dir "${task_dir}" \
      --checkpoint "${LINGBOT_BASE_CKPT}" \
      --max-episodes "${MAX_EPISODES}" \
      --device cuda \
      --overwrite

    local config_lock="${TABLE30_WORK_ROOT}/config.lock"
    while ! mkdir "${config_lock}" 2>/dev/null; do
      sleep 1
    done
    (
      trap 'rmdir "${config_lock}"' EXIT
      make_task_config "${config_key}" "${var_name}" "${dataset_root}" "${TRAIN_STEPS}"
    )

    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    NGPU=1 \
    CONFIG_NAME="${config_key}_train" \
    MASTER_PORT="$((29700 + gpu_id))" \
    LOG_RANK=0 \
    bash script/run_va_posttrain.sh \
      --save-root "${out_dir}/train_out"

    echo "[PASS] ${task_name}"
  } 2>&1 | tee "${log_file}"
}

info "Starting selected tasks."
tail -n +2 "${TASKS_TSV}" | nl -v 0 -w 1 -s $'\t' | while IFS=$'\t' read -r idx score task_name prompt task_dir robot_tag slug; do
  gpu_id=$((idx % 2))
  run_one_task "${gpu_id}" "${task_name}" "${prompt}" "${task_dir}" "${slug}" &
  while [[ "$(jobs -rp | wc -l | tr -d ' ')" -ge "${TABLE30_PARALLEL}" ]]; do
    sleep 10
  done
done

wait

info "Table30 subtask batch finished."
info "Selected tasks: ${TASKS_TSV}"
info "Outputs: ${TABLE30_WORK_ROOT}"
