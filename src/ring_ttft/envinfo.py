"""Hardware and software details saved alongside every result."""

from __future__ import annotations

import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version


def _sysctl(key: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _ver(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "not installed"


def env_info() -> dict:
    mem = _sysctl("hw.memsize")
    return {
        "chip": _sysctl("machdep.cpu.brand_string"),
        "memory_gb": round(int(mem) / 2**30) if mem.isdigit() else None,
        "os": f"macOS {platform.mac_ver()[0]}" if platform.system() == "Darwin" else platform.platform(),
        "python": platform.python_version(),
        "mlx": _ver("mlx"),
        "mlx_lm": _ver("mlx-lm"),
    }
