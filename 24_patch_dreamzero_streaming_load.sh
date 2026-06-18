#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

require_dir "${DREAMZERO_REPO}" "DreamZero repo"
TARGET="${DREAMZERO_REPO}/groot/vla/model/dreamzero/base_vla.py"
[[ -f "${TARGET}" ]] || die "Missing ${TARGET}"

python - "${TARGET}" <<'PY'
from pathlib import Path
import sys
import textwrap

p = Path(sys.argv[1])
s = p.read_text()

if "loading pretrained@@@@@ (streaming shards)" in s:
    print(f"already patched {p}")
    raise SystemExit(0)

start_marker = """    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path: str,
        config: VLAConfig = None
    ):
"""
start = s.rfind(start_marker)
if start < 0:
    raise SystemExit("Could not find target VLA.from_pretrained method")

end_marker = "\n    def post_initialize(self):\n"
end = s.find(end_marker, start)
if end < 0:
    raise SystemExit("Could not find end of VLA.from_pretrained method")

replacement = r'''    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        config: VLAConfig = None
    ):
        del config

        from safetensors.torch import load_file
        import os
        import json
        import gc

        print("loading pretrained@@@@@ (streaming shards)")
        safetensors_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        safetensors_index_path = os.path.join(pretrained_model_name_or_path, "model.safetensors.index.json")

        print("loading config@@")
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = VLAConfig(**config_dict)
        print("loading model")
        print("config.action_head_cfg", config.action_head_cfg)

        if 'config' in config.action_head_cfg and isinstance(config.action_head_cfg['config'], dict):
            if 'defer_lora_injection' in config.action_head_cfg['config']:
                config.action_head_cfg['config']['defer_lora_injection'] = False
                print("config.action_head_cfg['config']['defer_lora_injection'] disabled (set to False)")
        elif 'defer_lora_injection' in config.action_head_cfg:
            config.action_head_cfg['defer_lora_injection'] = False
            print("config.action_head_cfg['defer_lora_injection'] disabled (set to False)")

        model = cls(config)
        model_keys = set(model.state_dict().keys())
        loaded_keys = set()
        unexpected_keys_accum = set()

        def normalize_state_dict_keys(state_dict):
            if any(".base_layer." in key for key in state_dict.keys()):
                print("Removing '.base_layer' from state dict keys in current shard")
                return {k.replace(".base_layer.", "."): v for k, v in state_dict.items()}
            return state_dict

        if os.path.exists(safetensors_index_path):
            print(f"Loading sharded safetensors using streaming index: {safetensors_index_path}")
            with open(safetensors_index_path, "r") as f:
                index = json.load(f)
            shard_files = sorted(set(index["weight_map"].values()))
            for shard_file in shard_files:
                shard_path = os.path.join(pretrained_model_name_or_path, shard_file)
                print(f"Loading shard: {shard_path}")
                shard_state_dict = normalize_state_dict_keys(load_file(shard_path))
                loaded_keys.update(shard_state_dict.keys())
                unexpected_keys_accum.update(set(shard_state_dict.keys()) - model_keys)
                _missing_keys, unexpected_keys = model.load_state_dict(shard_state_dict, strict=False)
                unexpected_keys_accum.update(unexpected_keys)
                del shard_state_dict
                gc.collect()
        elif os.path.exists(safetensors_path):
            print(f"Loading weights from safetensors: {safetensors_path}")
            state_dict = normalize_state_dict_keys(load_file(safetensors_path))
            loaded_keys.update(state_dict.keys())
            unexpected_keys_accum.update(set(state_dict.keys()) - model_keys)
            _missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            unexpected_keys_accum.update(unexpected_keys)
            del state_dict
            gc.collect()
        else:
            raise FileNotFoundError(
                f"No weights found at '{pretrained_model_name_or_path}'. "
                "Expected 'model.safetensors' or 'model.safetensors.index.json'."
            )

        missing_keys = sorted(model_keys - loaded_keys)
        if missing_keys:
            preview = missing_keys[:50]
            print(f"Missing keys after streaming load: {preview} ... total={len(missing_keys)}")
        if unexpected_keys_accum:
            preview = sorted(unexpected_keys_accum)[:50]
            print(f"Unexpected keys after streaming load: {preview} ... total={len(unexpected_keys_accum)}")

        print("Successfully loaded pretrained weights with streaming shard load")
        print(f"{cls}\n")
        return model
'''

backup = p.with_suffix(p.suffix + ".bak_before_streaming_load")
if not backup.exists():
    backup.write_text(s)

p.write_text(s[:start] + replacement + s[end:])
print(f"patched streaming shard load in {p}")
print(f"backup: {backup}")
PY

echo "DreamZero streaming checkpoint load patch is ready."
