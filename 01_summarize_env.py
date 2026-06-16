#!/usr/bin/env python3
"""Summarize raw env probe logs into a beginner-friendly Markdown report."""
from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path


def latest_report(log_dir: Path) -> Path:
    reports = sorted(log_dir.glob("env_report_*.txt"), key=lambda p: p.stat().st_mtime)
    if not reports:
        raise SystemExit(f"No env_report_*.txt found in {log_dir}")
    return reports[-1]


def section(text: str, name: str) -> str:
    pattern = rf"^===== {re.escape(name)} =====\n(.*?)(?=^===== |\Z)"
    match = re.search(pattern, text, flags=re.M | re.S)
    return match.group(1).strip() if match else ""


def line_value(text: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}=(.*)$", text, flags=re.M)
    return match.group(1).strip() if match else ""


def status_word(section_text: str, needle: str) -> str:
    for line in section_text.splitlines():
        if needle in line:
            if " OK" in line or "=OK" in line:
                return "OK"
            if "FAILED" in line or "MISSING" in line:
                return "FAILED"
    return "UNKNOWN"


def bulletize(lines: list[str]) -> str:
    return "\n".join(f"- {line}" for line in lines if line)


def infer_risks(text: str) -> list[str]:
    risks: list[str] = []
    driver_major = line_value(text, "detected_driver_major")
    gpu_count = line_value(text, "detected_gpu_count")
    min_mem = line_value(text, "detected_min_gpu_mem_mb")
    recommendation = line_value(text, "recommendation")

    try:
        if int(driver_major or "0") < 560:
            risks.append("NVIDIA driver may be too old for LingBot cu126 or DreamZero cu129 wheels.")
    except ValueError:
        risks.append("Could not parse NVIDIA driver version.")

    try:
        if int(gpu_count or "0") < 2:
            risks.append("DreamZero distributed inference expects at least 2 visible GPUs.")
    except ValueError:
        pass

    try:
        if int(min_mem or "0") < 24000:
            risks.append("GPU VRAM looks below LingBot-VA's stated single-GPU i2av requirement.")
        elif int(min_mem or "0") < 70000:
            risks.append("DreamZero 14B inference may be VRAM constrained; start with smoke tests only.")
    except ValueError:
        pass

    for site in ["GitHub", "HuggingFace", "ModelScope", "PyPI", "PyTorch"]:
        if re.search(rf"{site}.*FAILED", text, flags=re.I):
            risks.append(f"{site} access failed; downloads may need mirrors, proxy, or offline package.")

    if "conda: not found" in text and "micromamba: not found" in text and "mamba: not found" in text:
        risks.append("No conda/mamba/micromamba found; setup scripts cannot create isolated envs yet.")

    if "nvidia-smi not found" in text:
        risks.append("nvidia-smi is missing; the GPU driver/container runtime is not visible.")

    if recommendation:
        risks.append(f"Probe route hint: {recommendation}")
    return risks


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    if len(sys.argv) >= 2:
        report = Path(sys.argv[1]).expanduser().resolve()
    else:
        report = latest_report(script_dir / "logs")
    text = report.read_text(encoding="utf-8", errors="replace")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = report.parent / f"env_summary_{ts}.md"

    gpu = section(text, "gpu and driver")
    cuda = section(text, "cuda runtime and nvcc")
    managers = section(text, "python package managers")
    network = section(text, "network")
    disk = section(text, "disk")
    torch = section(text, "torch import")
    pkgs = section(text, "important python package imports")

    gpu_lines = [
        f"GPU names: `{line_value(text, 'detected_gpu_names') or 'unknown'}`",
        f"GPU count: `{line_value(text, 'detected_gpu_count') or 'unknown'}`",
        f"Minimum VRAM: `{line_value(text, 'detected_min_gpu_mem_mb') or 'unknown'} MiB`",
        f"Driver major: `{line_value(text, 'detected_driver_major') or 'unknown'}`",
    ]

    cuda_lines = [
        f"Torch import: `{line_value(torch, 'torch_import') or status_word(torch, 'torch_import')}`",
        f"Torch version: `{line_value(torch, 'torch_version') or 'unknown'}`",
        f"Torch CUDA: `{line_value(torch, 'torch_cuda_version') or 'unknown'}`",
        f"Torch cuda available: `{line_value(torch, 'torch_cuda_available') or 'unknown'}`",
        "nvcc present: `" + ("yes" if "release" in cuda or "Cuda compilation tools" in cuda else "no/unknown") + "`",
    ]

    py_lines = []
    for cmd in ["python", "python3", "pip", "pip3", "conda", "mamba", "micromamba"]:
        found = "yes" if re.search(rf"^{cmd}: (?!not found)", managers, flags=re.M) else "no"
        py_lines.append(f"{cmd}: `{found}`")

    net_lines = []
    for name in ["GitHub", "HuggingFace", "ModelScope", "PyPI", "PyTorch cu126 index", "PyTorch cu129 index"]:
        m = re.search(rf"^{re.escape(name)}\s+.*\s+(OK|FAILED|SKIPPED.*)$", network, flags=re.M)
        net_lines.append(f"{name}: `{m.group(1) if m else 'UNKNOWN'}`")

    pkg_lines = []
    for mod in ["flash_attn", "transformers", "diffusers", "accelerate", "websockets", "cv2", "robosuite", "libero", "lerobot"]:
        m = re.search(rf"^{re.escape(mod)}=(.*)$", pkgs, flags=re.M)
        pkg_lines.append(f"{mod}: `{m.group(1) if m else 'UNKNOWN'}`")

    risks = infer_risks(text)
    next_steps = [
        "Copy `config.example.env` to `config.env` and edit repository, checkpoint, cache, and data paths.",
        "On Tencent Cloud A100, start with `CUDA_VISIBLE_DEVICES=0` and LingBot-VA i2av; only use DreamZero when 2 GPUs are visible and the driver supports CUDA 12.9 wheels.",
        "If this is 2xH800 with driver >=560, try DreamZero server smoke after downloading the DROID checkpoint.",
        "If this is 2xH20, try LingBot-VA i2av first, then LIBERO evaluation; treat DreamZero as a higher-risk smoke test.",
        "Keep LingBot-VA, DreamZero, and LIBERO in separate conda envs to avoid Torch/CUDA dependency conflicts.",
    ]

    md = f"""# Robot WAM Environment Summary

Raw report: `{report.name}`

## GPU situation
{bulletize(gpu_lines)}

## CUDA/PyTorch situation
{bulletize(cuda_lines)}

Important imports:
{bulletize(pkg_lines)}

## Python/Conda situation
{bulletize(py_lines)}

## Network access situation
{bulletize(net_lines)}

## Disk space situation

```text
{disk[:3000] if disk else 'No df output found.'}
```

## Risk points
{bulletize(risks) if risks else '- No obvious high-risk point found in the probe log.'}

## Recommended next step
{bulletize(next_steps)}

## Raw route hint

`{line_value(text, 'recommendation') or 'No recommendation line found.'}`
"""
    out.write_text(md, encoding="utf-8")
    print(f"Wrote summary: {out}")


if __name__ == "__main__":
    main()
