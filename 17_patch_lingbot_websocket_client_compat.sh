#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
load_config

WS_FILE="${LINGBOT_REPO}/wan_va/utils/Simple_Remote_Infer/deploy/websocket_client_policy.py"
require_file "${WS_FILE}" "LingBot websocket client"

python - "${WS_FILE}" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

start = text.find("    def _wait_for_server(")
if start < 0:
    raise SystemExit("Could not find _wait_for_server")
next_def = text.find("\n    @override\n    def infer", start)
if next_def < 0:
    raise SystemExit("Could not find infer after _wait_for_server")

new_block = r'''    def _wait_for_server(
            self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {
                    "Authorization": f"Api-Key {self._api_key}"
                } if self._api_key else None

                base_kwargs = dict(compression=None, max_size=None)
                candidates = [
                    dict(base_kwargs, ping_interval=None, close_timeout=10),
                    dict(base_kwargs, close_timeout=10),
                    dict(base_kwargs),
                    {},
                ]
                conn = None
                last_error = None
                for kwargs in candidates:
                    if headers:
                        kwargs = dict(kwargs)
                        kwargs["additional_headers"] = headers
                    try:
                        conn = websockets.sync.client.connect(self._uri, **kwargs)
                        break
                    except TypeError as err:
                        last_error = err
                        if "additional_headers" in kwargs:
                            kwargs = dict(kwargs)
                            kwargs["extra_headers"] = kwargs.pop("additional_headers")
                            try:
                                conn = websockets.sync.client.connect(self._uri, **kwargs)
                                break
                            except TypeError as err2:
                                last_error = err2
                if conn is None:
                    raise last_error or RuntimeError("websocket connect failed")

                metadata = unpackb(conn.recv())
                print(f"Connected to server at {self._uri}; metadata={metadata}", flush=True)
                return conn, metadata
            except Exception as e:
                print(f"Still waiting for server at {self._uri}... ({type(e).__name__}: {e})", flush=True)
                time.sleep(5)
'''

backup = path.with_suffix(path.suffix + ".bak_ws_compat")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")
path.write_text(text[:start] + new_block + text[next_def:], encoding="utf-8")
print(f"patched websocket client compatibility in {path}")
PY

echo "LingBot websocket client compatibility patch is ready."
