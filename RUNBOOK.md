# RUNBOOK: DreamZero / LingBot-VA First Probe and Smoke Tests

## 1. Upload This Folder

From your local machine, upload `robot_wam_probe_and_run/` to the GPU server.
With VSCode SSH, the easiest method is usually dragging the folder into the remote Explorer, or using `scp`:

```bash
scp -r robot_wam_probe_and_run user@server:/home/user/worldmodel/
```

If you also have source snapshots, upload or clone them so the server has:

```text
~/worldmodel/sources/lingbot-va
~/worldmodel/sources/dreamzero
~/worldmodel/sources/LIBERO
```

For Tencent Cloud A100, see `TENCENT_A100_UPLOAD.md` for the exact minimal upload list.

## 2. Run the Probe First

On the server:

```bash
cd ~/worldmodel/robot_wam_probe_and_run
bash 00_probe_env.sh
```

This only reads environment information. It does not install packages, change system files, use sudo, or delete anything.

## 3. Read the Summary

After the probe:

```bash
ls -lt logs/env_summary_*.md | head
cat logs/env_summary_YYYYMMDD_HHMMSS.md
```

The raw full report is also saved as:

```bash
logs/env_report_YYYYMMDD_HHMMSS.txt
```

## 4. Create Your Config

```bash
cp config.example.env config.env
nano config.env
```

Fill in:

- `PROJECT_ROOT`
- `HF_HOME`
- `HF_TOKEN`
- `MODELSCOPE_CACHE`
- `CUDA_VISIBLE_DEVICES`
- `LINGBOT_CKPT_DIR`
- `DREAMZERO_CKPT_DIR`
- `LIBERO_DATA_DIR`
- `LOG_DIR`

Do not write a real token into a shared file or public repo.

## 5. If the Server Is 2xH800

Recommended order:

1. Try DreamZero server smoke if the driver is new enough for CUDA 12.9 wheels.
2. Try LingBot-VA image-to-video-action smoke.
3. Prepare LIBERO after you know the basic model inference path works.

Commands:

```bash
CONFIRM_INSTALL=1 bash 20_setup_dreamzero.sh
bash 21_download_dreamzero_ckpt.sh
bash 22_run_dreamzero_server_smoke.sh
```

H800 has enough class of hardware for DreamZero-style tests, but the driver and CUDA wheel compatibility still decide whether it works cleanly.

## 6. If the Server Is 2xH20

Recommended order:

1. LingBot-VA image-to-video-action smoke.
2. LingBot-VA LIBERO evaluation route.
3. DreamZero server smoke only after confirming driver >= 575 and enough free VRAM.

Commands:

```bash
CONFIRM_INSTALL=1 bash 10_setup_lingbot_va.sh
bash 11_run_lingbot_va_smoke.sh
CONFIRM_INSTALL=1 bash 13_setup_libero_env.sh
bash 12_run_lingbot_libero_eval.sh
```

H20 can be useful, but DreamZero 14B distributed inference is a more fragile first target than LingBot-VA or native LIBERO checks.

## 7. If the Server Is Tencent Cloud A100

Recommended order:

1. Run `00_probe_env.sh` first and check driver version.
2. If driver major is 560 or newer, try LingBot-VA image-to-video-action smoke on one GPU.
3. Prepare LIBERO only after LingBot imports and basic inference work.
4. Try DreamZero only if you have 2 visible A100 GPUs and driver major is 575 or newer.

For a single A100, keep this in `config.env`:

```bash
CUDA_VISIBLE_DEVICES="0"
DREAMZERO_CUDA_VISIBLE_DEVICES="0,1"
```

Commands:

```bash
cd ~/worldmodel/robot_wam_probe_and_run
bash 00_probe_env.sh
cp config.example.env config.env
nano config.env
CONFIRM_INSTALL=1 bash 10_setup_lingbot_va.sh
bash 11_run_lingbot_va_smoke.sh
```

If HuggingFace is hard to reach from Tencent Cloud, LingBot can try ModelScope:

```bash
USE_MODELSCOPE=1 DOWNLOAD_LINGBOT_CKPT=1 CONFIRM_INSTALL=1 bash 10_setup_lingbot_va.sh
```

## 8. Why LingBot-VA First for LIBERO

LingBot-VA officially provides a LIBERO server/client path:

```bash
bash evaluation/libero/launch_server.sh
bash evaluation/libero/launch_client.sh
```

The scripts here wrap that flow with logs, PID recording, port checks, and smaller default eval ranges. LIBERO dependencies are old and can conflict with modern Torch, so keep the native LIBERO env separate from LingBot-VA.

## 9. Why DreamZero Starts with DROID/Server Smoke

DreamZero officially ships a WebSocket inference server and DROID simulation route. It does not advertise LIBERO as the first evaluation path. For a first reproduction attempt, the safest order is:

1. Environment probe.
2. DreamZero checkpoint download.
3. Official 2-GPU server smoke.
4. Official DROID/sim eval only after API host/port and assets are ready.

Do not force DreamZero into LIBERO as step one. That creates two hard problems at once: model-server bring-up and simulator adaptation.

## 10. Common Errors

### CUDA driver too old

Symptom:

```text
CUDA driver version is insufficient
driver too old
```

Action:

- Check `nvidia-smi` in the probe report.
- LingBot-VA cu126 usually needs a newer driver.
- DreamZero cu129 is stricter; update/load a newer driver module before installing.

### flash-attn compile failed

Action:

```bash
python -m pip install --upgrade pip setuptools wheel
MAX_JOBS=8 python -m pip install --no-build-isolation flash-attn
```

Make sure torch is already installed before flash-attn.

### HuggingFace cannot be accessed

Action:

- Check the network section in `env_summary_*.md`.
- Set proxy variables if your server needs them.
- Set `HF_TOKEN` if the repo is gated.
- For LingBot-VA, try `USE_MODELSCOPE=1 DOWNLOAD_LINGBOT_CKPT=1 CONFIRM_INSTALL=1 bash 10_setup_lingbot_va.sh`.

### Disk is not enough

Action:

- Read `df -h` in the probe report.
- DreamZero and sim assets can easily require tens to hundreds of GB.
- Move `HF_HOME`, `DREAMZERO_CKPT_DIR`, and `LIBERO_DATA_DIR` to a large data disk.

### conda does not exist

Action:

- Install Miniconda/Mambaforge in your home directory, or load the cluster module that provides conda.
- Re-run `bash 00_probe_env.sh`.

### robosuite / mujoco / headless rendering error

Action:

Try:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

Also check whether the server/container exposes GPU rendering libraries.

### server/client port occupied

Action:

- Change `LINGBOT_PORT` or `DREAMZERO_PORT` in `config.env`.
- Or stop the old process.
- For LingBot: `bash stop_lingbot_server.sh logs/.../server.pid`.

### GPU memory insufficient

Action:

- Use the smallest smoke settings first.
- For LingBot-VA, start with `LINGBOT_NUM_CHUNKS=1`.
- For DreamZero, make sure exactly two high-memory GPUs are visible via `CUDA_VISIBLE_DEVICES=0,1`.
- Avoid launching multiple server copies.

## 11. Safety Rules

- No script uses `sudo`.
- Installation scripts stop unless `CONFIRM_INSTALL=1` is set.
- Logs are saved under `LOG_DIR`.
- Paths are read from `config.env`.
- Scripts are intended to be repeatable; existing envs/repos are reused.
