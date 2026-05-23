from __future__ import annotations

import os
import subprocess


def get_hardware_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return os.uname().machine


def get_gpu_power_watts() -> float | None:
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip().splitlines()

        if result:
            return float(result[0])
        return None
    except Exception:
        return None


def reset_peak_memory_stats() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def get_peak_vram_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
    except Exception:
        pass
    return None
