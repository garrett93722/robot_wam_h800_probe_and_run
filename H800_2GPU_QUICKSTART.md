# 2xH800 Remote SSH Quick Start

This folder is the small control surface to upload/open on the 2xH800 server.
It is safe to start with probe scripts; they do not install packages.

## Recommended Remote Layout

```text
/workspace/
  robot_wam_h800_probe_and_run/
  sources/
    dreamzero/
    lingbot-va/
    LIBERO/
  checkpoints/
  data/
```

## First Probe

```bash
cd /workspace/robot_wam_h800_probe_and_run
bash 00_probe_env.sh
python 01_summarize_env.py
bash make_config_for_current_tree.sh
bash 02_probe_h800_readiness.sh
```

Read the latest files under `logs/`.

## Source Repos

If `02_probe_h800_readiness.sh` reports missing repos:

```bash
mkdir -p /workspace/sources
cd /workspace/sources
git clone https://github.com/dreamzero0/dreamzero.git dreamzero
git clone https://github.com/Robbyant/lingbot-va.git lingbot-va
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git LIBERO
```

If any clone URL fails, upload the local `D:\worldmodel\sources\...` snapshot.

## DreamZero First Route

Use this on 2xH800 if the probe shows both GPUs and driver/network are OK:

```bash
cd /workspace/robot_wam_h800_probe_and_run
CONFIRM_INSTALL=1 bash 20_setup_dreamzero.sh
bash 21_download_dreamzero_ckpt.sh
bash 22_run_dreamzero_server_smoke.sh
```

The DreamZero setup uses a separate `dreamzero` conda env and defaults to
`DREAMZERO_CUDA_VISIBLE_DEVICES=0,1`.

If the server is `SIGKILL`ed while loading DreamZero-DROID and `resource.log`
shows `/sys/fs/cgroup/memory.current` reaching `/sys/fs/cgroup/memory.max`,
patch the checkpoint loader to stream safetensor shards instead of building one
large in-memory state dict:

```bash
bash 24_patch_dreamzero_streaming_load.sh
DREAMZERO_ENABLE_DIT_CACHE=0 DREAMZERO_DISABLE_TORCH_COMPILE=1 bash 22_run_dreamzero_server_smoke.sh
```

## Fallback Route

If DreamZero is blocked by package or checkpoint issues, use LingBot-VA first:

```bash
CONFIRM_INSTALL=1 bash 10_setup_lingbot_va.sh
bash 11_run_lingbot_va_smoke.sh
```

For LIBERO evaluation, install the separate `libero` env only after LingBot
smoke is stable.
