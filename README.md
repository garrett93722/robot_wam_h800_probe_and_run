# Robot WAM Probe and Run for 2xH800

This folder is meant to be uploaded to a remote 2xH800 GPU server through VSCode SSH.
Start with environment probing only; do not install anything until you have read the generated summary.

## Files

- `00_probe_env.sh`: collect GPU, CUDA, Python, package, disk, memory, network, and route-suggestion info. It does not install or modify packages.
- `01_summarize_env.py`: convert `logs/env_report_*.txt` into a clearer `logs/env_summary_*.md`.
- `02_probe_h800_readiness.sh`: focused 2xH800 readiness probe for GPUs, driver, topology, repos, network, and ports.
- `config.example.env`: copy to `config.env` and edit all paths/tokens locally on the server.
- `common.sh`: shared helpers for config loading, conda activation, logging, disk checks, port checks, and error diagnosis.
- `10_setup_lingbot_va.sh`: prepare a separate LingBot-VA conda env. Requires `CONFIRM_INSTALL=1` to install.
- `11_run_lingbot_va_smoke.sh`: run LingBot-VA import checks and a one-GPU image-to-video-action smoke test.
- `12_run_lingbot_libero_eval.sh`: start LingBot-VA LIBERO server/client with logs and PID tracking.
- `13_setup_libero_env.sh`: prepare a separate native LIBERO conda env. Requires `CONFIRM_INSTALL=1` to install.
- `14_patch_lingbot_torch_attention_fallback.sh`: patch LingBot-VA to use PyTorch attention when flash-attn cannot be built.
- `15_probe_libero_eval_env.sh`: lightweight LIBERO eval readiness probe; no install and no model server.
- `16_patch_lingbot_libero_client_fallback.sh`: patch LingBot LIBERO client to avoid requiring LeRobot just for JSON writing.
- `17_patch_lingbot_websocket_client_compat.sh`: patch websocket client connection compatibility and print connection errors.
- `18_download_libero_datasets.sh`: download/check LIBERO datasets, defaulting to the official Box links because the old HuggingFace allow-pattern path can fetch 0 files.
- `stop_lingbot_server.sh`: stop a background LingBot LIBERO server from its PID file.
- `20_setup_dreamzero.sh`: prepare a separate DreamZero conda env. Requires `CONFIRM_INSTALL=1` to install.
- `21_download_dreamzero_ckpt.sh`: download DreamZero-DROID and/or DreamZero-AgiBot checkpoints with resume support.
- `22_run_dreamzero_server_smoke.sh`: start DreamZero 2-GPU WebSocket server and run the official test client.
- `23_run_dreamzero_droid_sim_eval.sh`: prepare/run DreamZero DROID sim-evals route when host/port/assets are ready.
- `RUNBOOK.md`: beginner-friendly step-by-step operating guide.

## Quick Start

```bash
cd robot_wam_h800_probe_and_run
bash 00_probe_env.sh
python 01_summarize_env.py
ls -lt logs/env_summary_*.md | head
cp config.example.env config.env
nano config.env
```

For the tree where `robot_wam_h800_probe_and_run/` and `sources/` are siblings, generate config automatically:

```bash
bash make_config_for_current_tree.sh
```

After reading the summary, choose one route:

```bash
# 2xH800 recommended first route
CONFIRM_INSTALL=1 bash 20_setup_dreamzero.sh
bash 21_download_dreamzero_ckpt.sh
bash 22_run_dreamzero_server_smoke.sh

# Fallback route if DreamZero is blocked
CONFIRM_INSTALL=1 bash 10_setup_lingbot_va.sh
bash 11_run_lingbot_va_smoke.sh
```

For 2xH800 details, see `H800_2GPU_QUICKSTART.md`.
