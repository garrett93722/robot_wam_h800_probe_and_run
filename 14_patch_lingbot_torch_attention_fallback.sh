#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
load_config

require_dir "${LINGBOT_REPO}" "LingBot-VA repo"
MODEL_FILE="${LINGBOT_REPO}/wan_va/modules/model.py"
require_file "${MODEL_FILE}" "LingBot model.py"

python - "${MODEL_FILE}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

old_import = """try:
    from flash_attn_interface import flash_attn_func
except:
    from flash_attn import flash_attn_func
"""

new_import = """try:
    from flash_attn_interface import flash_attn_func
except Exception:
    try:
        from flash_attn import flash_attn_func
    except Exception:
        flash_attn_func = None
"""

old_attn = """        elif attn_mode == 'flashattn':
            self.attn_op = flash_attn_func
"""

new_attn = """        elif attn_mode == 'flashattn':
            # Tencent A100 smoke fallback: if flash-attn is unavailable, keep
            # inference debuggable with PyTorch SDPA. Slower, but avoids a
            # failed source build blocking the first run.
            self.attn_op = flash_attn_func if flash_attn_func is not None else custom_sdpa
"""

changed = False
if old_import in text:
    text = text.replace(old_import, new_import)
    changed = True
if old_attn in text:
    text = text.replace(old_attn, new_attn)
    changed = True

if "flash_attn_func = None" not in text:
    raise SystemExit("Could not verify flash-attn fallback patch in model.py")

if changed:
    backup = path.with_suffix(path.suffix + ".bak_flash_fallback")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    print(f"patched {path}")
else:
    print(f"already patched {path}")
PY

if [[ -f "${LINGBOT_CKPT_DIR}/transformer/config.json" ]]; then
  python - "${LINGBOT_CKPT_DIR}/transformer/config.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
cfg = json.loads(path.read_text(encoding="utf-8"))
old = cfg.get("attn_mode")
cfg["attn_mode"] = "torch"
path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"set {path} attn_mode: {old!r} -> 'torch'")
PY
else
  echo "[WARN] Checkpoint config not found yet: ${LINGBOT_CKPT_DIR}/transformer/config.json"
  echo "[WARN] After checkpoint download, re-run this patch script to force attn_mode=torch."
fi

echo "LingBot torch-attention fallback is ready."
