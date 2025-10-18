#!/usr/bin/env python3
"""
Readout acceptance benchmark: acceptance ratio vs λ_max strategies.

Outputs: results/benchmark_readout_acceptance.json
Schema: {task, seed, commit, env, metrics{rows: [[multiple, acceptance_ratio], ...]}}
"""
from __future__ import annotations

import json
import os
from typing import List

import torch

# Local import with fallback to source tree
try:
    from sr_ciden.readout import sample_ogata_with_stats
    from sr_ciden.utils.determinism import seed_all, capture_env
except Exception:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from sr_ciden.readout import sample_ogata_with_stats
    from sr_ciden.utils.determinism import seed_all, capture_env


def _intensity_fn(t: float) -> torch.Tensor:
    # Two-mark constant intensities
    return torch.tensor([0.5, 0.2], dtype=torch.get_default_dtype())


def main() -> None:
    seed_all(0)
    os.makedirs("results", exist_ok=True)
    env = capture_env(os.path.join("artifacts", "env.json"))

    true_max = 0.7
    T = 3.0
    multiples = [1.0, 1.2, 1.5, 2.0, 5.0]
    rows: List[List[str]] = []
    for m in multiples:
        lam_max = true_max * m
        _, stats = sample_ogata_with_stats(_intensity_fn, T=T, lam_max=lam_max, seed=123)
        acc = float(stats["acceptance_ratio"])
        rows.append([f"{m:.1f}x", f"{acc:.3f}"])

    payload = {
        "task": "readout_acceptance",
        "seed": 0,
        "commit": (env.get("git", {}) or {}).get("commit"),
        "env": {
            "platform": env.get("platform"),
            "python": env.get("python"),
            "versions": env.get("versions"),
            "torch": env.get("torch"),
        },
        "metrics": {
            "rows": rows
        },
    }
    with open(os.path.join("results", "benchmark_readout_acceptance.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
