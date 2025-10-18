#!/usr/bin/env python3
"""
Configuration utilities for SR-CIDEN.

- Load YAML/JSON configs
- Argparse bridge with --config path and CLI key=val overrides (dot-notation for nested)
- Optional dependency on PyYAML (provide helpful error if missing)

Example:
    from sr_ciden.utils.config import load_config, apply_overrides

    cfg = load_config("configs/ks_timescaling.yaml")
    cfg = apply_overrides(cfg, ["seed=123", "bench.runs=1000"])
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


def _import_yaml():
    try:
        import yaml  # type: ignore
        return yaml
    except Exception as e:
        raise RuntimeError(
            "PyYAML is required to load YAML configs. Install with: pip install PyYAML"
        ) from e


def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    _, ext = os.path.splitext(path.lower())
    if ext in (".yml", ".yaml"):
        yaml = _import_yaml()
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    # JSON fallback
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _set_deep(cfg: Dict[str, Any], key_path: List[str], value: Any) -> None:
    cur = cfg
    for k in key_path[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[key_path[-1]] = value


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """
    Apply overrides like ["a.b=1", "bench.runs=100"] to a nested dict.
    Numbers and booleans are parsed; strings must be quoted: key="value".
    """
    import ast

    for ov in overrides:
        if "=" not in ov:
            continue
        k, v = ov.split("=", 1)
        kpath = k.strip().split(".")
        v_str = v.strip()
        try:
            parsed = ast.literal_eval(v_str)
        except Exception:
            parsed = v_str
        _set_deep(cfg, kpath, parsed)
    return cfg


def parse_args_with_config(description: str = "SR-CIDEN runner", default_config: str | None = None):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=str, default=default_config, help="Path to YAML/JSON config")
    p.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help='Key=Value overrides (dot notation), e.g. "bench.runs=100 seed=0"',
    )
    args, unknown = p.parse_known_args()
    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override or [])
    return args, cfg, unknown
