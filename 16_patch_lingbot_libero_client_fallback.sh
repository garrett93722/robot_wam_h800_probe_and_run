#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
load_config

CLIENT_FILE="${LINGBOT_REPO}/evaluation/libero/client.py"
require_file "${CLIENT_FILE}" "LingBot LIBERO client"

python - "${CLIENT_FILE}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
marker = "def save_video("
idx = text.find(marker)
if idx < 0:
    raise SystemExit("Could not find def save_video in client.py")

clean_header = '''import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import imageio
import numpy as np
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm

_SIMPLE_REMOTE_INFER_ROOT = Path(__file__).resolve().parents[2] / "wan_va" / "utils" / "Simple_Remote_Infer"
if str(_SIMPLE_REMOTE_INFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_SIMPLE_REMOTE_INFER_ROOT))
from deploy.websocket_client_policy import WebsocketClientPolicy


def write_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


'''

backup = path.with_suffix(path.suffix + ".bak_lerobot_fallback")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")
path.write_text(clean_header + text[idx:], encoding="utf-8")
text = path.read_text(encoding="utf-8")
text = text.replace(
    "model = WebsocketClientPolicy(port=port)",
    'model = WebsocketClientPolicy(host=os.environ.get("LINGBOT_SERVER_HOST", "127.0.0.1"), port=port)'
)
path.write_text(text, encoding="utf-8")
print(f"patched clean client header in {path}")
PY

echo "LingBot LIBERO client websocket/json fallback is ready."
