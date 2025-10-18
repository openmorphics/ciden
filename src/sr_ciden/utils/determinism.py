"""
Determinism and reproducibility utilities for SR-CIDEN.

This module centralizes:
- Global seeding across Python, NumPy, and PyTorch
- PyTorch deterministic algorithm switches (CUDNN, algos)
- Default ODE solver tolerances (configurable via env)
- Environment capture to JSON for experiment artifacts
- Dataset cache root resolution via SR_CIDEN_DATA

Usage:
    from sr_ciden.utils.determinism import (
        seed_all, set_torch_determinism, get_default_ode_tolerances,
        capture_env, get_data_root
    )

Notes:
- Functions are CPU/GPU aware but degrade gracefully if torch is not available.
- Env variables:
    SR_CIDEN_RTOL, SR_CIDEN_ATOL  (floats for ODE tolerances)
    SR_CIDEN_DATA                 (dataset cache root)
"""

from __future__ import annotations

import json
import os
import pathlib
import platform
import random
import subprocess
from typing import Any, Dict, Optional


try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


__all__ = [
    "seed_all",
    "set_torch_determinism",
    "get_default_ode_tolerances",
    "capture_env",
    "get_data_root",
]


def seed_all(
    seed: int,
    *,
    deterministic_algos: bool = True,
    cudnn_benchmark: bool = False,
) -> None:
    """
    Seed Python, NumPy, and PyTorch (CPU/GPU).

    Args:
        seed: Global seed value.
        deterministic_algos: If True, enable PyTorch deterministic algorithms where supported.
        cudnn_benchmark: Torch CUDNN benchmark flag (generally False for determinism).
    """
    random.seed(seed)
    if np is not None:
        try:
            np.random.seed(seed)  # type: ignore[attr-defined]
        except Exception:
            pass

    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)  # type: ignore[attr-defined]
                torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
        except Exception:
            pass

        # cuDNN and deterministic algorithm switches
        try:
            if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):  # type: ignore[attr-defined]
                torch.backends.cudnn.deterministic = bool(deterministic_algos)  # type: ignore[attr-defined]
                torch.backends.cudnn.benchmark = bool(cudnn_benchmark)  # type: ignore[attr-defined]
        except Exception:
            pass

        # torch.use_deterministic_algorithms API varies across versions
        try:
            if deterministic_algos:
                # Prefer strict determinism, but some versions accept (enabled, warn_only)
                if hasattr(torch, "use_deterministic_algorithms"):
                    try:
                        torch.use_deterministic_algorithms(True, warn_only=False)  # type: ignore[call-arg]
                    except TypeError:
                        torch.use_deterministic_algorithms(True)  # type: ignore[call-arg]
            else:
                if hasattr(torch, "use_deterministic_algorithms"):
                    try:
                        torch.use_deterministic_algorithms(False, warn_only=True)  # type: ignore[call-arg]
                    except TypeError:
                        torch.use_deterministic_algorithms(False)  # type: ignore[call-arg]
        except Exception:
            pass


def set_torch_determinism(enabled: bool = True, warn_only: bool = False) -> None:
    """
    Toggle PyTorch deterministic algorithms globally.

    Args:
        enabled: If True, enable deterministic algorithms; else disable.
        warn_only: When disabling, prefer warn_only where supported.
    """
    if torch is None:
        return
    try:
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(bool(enabled), warn_only=bool(warn_only))  # type: ignore[call-arg]
            except TypeError:
                # Older torch does not support warn_only kwarg
                torch.use_deterministic_algorithms(bool(enabled))  # type: ignore[call-arg]
    except Exception:
        pass


def _get_env_float(name: str, default: float) -> float:
    val = os.environ.get(name, "")
    if not val:
        return default
    try:
        return float(val)
    except Exception:
        return default


def get_default_ode_tolerances() -> Dict[str, float]:
    """
    Return default ODE solver tolerances with env overrides.

    Env overrides:
        SR_CIDEN_RTOL (default 1e-5)
        SR_CIDEN_ATOL (default 1e-7)

    Returns:
        dict with keys {"rtol", "atol"}.
    """
    rtol = _get_env_float("SR_CIDEN_RTOL", 1e-5)
    atol = _get_env_float("SR_CIDEN_ATOL", 1e-7)
    return {"rtol": rtol, "atol": atol}


def get_data_root() -> str:
    """
    Resolve dataset cache root directory.

    Order of precedence:
      1) Env SR_CIDEN_DATA
      2) ~/.cache/sr_ciden

    Returns:
        Absolute path (string) to the dataset root. Directory is created if missing.
    """
    root = os.environ.get("SR_CIDEN_DATA", "").strip()
    if not root:
        root = os.path.join(pathlib.Path.home(), ".cache", "sr_ciden")
    os.makedirs(root, exist_ok=True)
    return os.path.abspath(root)


def _safe_version(mod_name: str) -> Optional[str]:
    try:
        mod = __import__(mod_name)
        return getattr(mod, "__version__", None)  # type: ignore[attr-defined]
    except Exception:
        return None


def _git_commit_sha() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def capture_env(out_path: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Capture runtime environment (OS, Python, key libs, hardware) and write JSON.

    Args:
        out_path: Destination file path for env JSON.
        extra: Optional dict merged into the root level of the JSON.

    Returns:
        The environment dictionary written to disk.
    """
    info: Dict[str, Any] = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "versions": {
            "numpy": _safe_version("numpy"),
            "torch": _safe_version("torch"),
            "matplotlib": _safe_version("matplotlib"),
            "scipy": _safe_version("scipy"),
        },
        "git": {"commit": _git_commit_sha()},
        "sr_ciden": {},
    }

    # Torch details if available
    if torch is not None:
        try:
            cuda_available = bool(torch.cuda.is_available())  # type: ignore[attr-defined]
        except Exception:
            cuda_available = False

        cuda_info: Dict[str, Any] = {
            "available": cuda_available,
            "device_count": int(torch.cuda.device_count()) if cuda_available else 0,  # type: ignore[attr-defined]
        }
        try:
            cuda_info["version"] = getattr(torch.version, "cuda", None)  # type: ignore[attr-defined]
        except Exception:
            cuda_info["version"] = None
        try:
            cuda_info["cudnn"] = {
                "enabled": bool(torch.backends.cudnn.enabled),  # type: ignore[attr-defined]
                "version": int(torch.backends.cudnn.version()) if hasattr(torch.backends.cudnn, "version") else None,  # type: ignore[attr-defined]
            }
        except Exception:
            pass
        try:
            if cuda_available:
                cuda_info["devices"] = [
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),  # type: ignore[attr-defined]
                        "capability": torch.cuda.get_device_capability(i),  # type: ignore[attr-defined]
                    }
                    for i in range(torch.cuda.device_count())  # type: ignore[attr-defined]
                ]
        except Exception:
            pass

        # Apple MPS (Metal) backend
        try:
            mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()  # type: ignore[attr-defined]
        except Exception:
            mps_available = False

        info["torch"] = {
            "cuda": cuda_info,
            "mps": {"available": bool(mps_available)},
            "deterministic_algorithms": _query_deterministic_algorithms(),
        }

    # Merge extras
    if extra:
        try:
            info.update(extra)
        except Exception:
            pass

    # Write JSON
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, sort_keys=True)
    return info


def _query_deterministic_algorithms() -> Optional[bool]:
    if torch is None:
        return None
    # torch does not expose a direct getter; infer via a small randomized op?
    # Return None to avoid side effects; presence of deterministic mode is recorded indirectly.
    return None