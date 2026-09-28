"""Seeding and environment capture for reproducible runs."""

from __future__ import annotations

import importlib.metadata
import platform
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_TRACKED_PACKAGES = ("numpy", "pandas", "scipy", "astropy", "astroquery", "lightkurve", "torch")


def set_seeds(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch global RNGs.

    Library code should still pass explicit ``numpy.random.Generator`` objects; this only
    guards against accidental use of global state. Bitwise determinism across GPU
    environments is not guaranteed.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - intentional global seeding
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_commit(root: Path) -> dict[str, Any]:
    """Current commit hash and whether the working tree has uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_info(root: Path) -> dict[str, Any]:
    """Snapshot of software/hardware context. PyTorch/CUDA fields are null if torch is absent."""
    info: dict[str, Any] = {
        "captured_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git": git_commit(root),
        "packages": package_versions(),
        "torch_cuda_available": None,
        "gpu_name": None,
    }
    try:
        import torch
    except ImportError:
        return info
    info["torch_cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info
