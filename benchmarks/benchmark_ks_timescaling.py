#!/usr/bin/env python3
"""
KS time-rescaling diagnostics on synthetic spike trains.

Outputs: results/benchmark_ks_timescaling.json
Schema: {task, seed, commit, env, metrics{poisson_p, sinusoid_p}}
"""
from __future__ import annotations

import json
import math
import os
from typing import List, Tuple

import torch

# Local import with fallback to source tree
try:
    from sr_ciden.readout import sample_ogata
    from sr_ciden.validation import time_rescaling_test
    from sr_ciden.utils.determinism import seed_all, capture_env
except Exception:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from sr_ciden.readout import sample_ogata
    from sr_ciden.validation import time_rescaling_test
    from sr_ciden.utils.determinism import seed_all, capture_env


def _const_intensity(lam: float):
    lam_t = torch.tensor([lam], dtype=torch.get_default_dtype())
    def fn(t: float) -> torch.Tensor:
        return lam_t
    return fn


def _sin_intensity(lam0: float, amp: float, freq_hz: float):
    # lam(t) = lam0 * (1 + amp * sin(2π f t)), ensured positive if amp < 1
    def fn(t: float) -> torch.Tensor:
        val = lam0 * (1.0 + amp * math.sin(2.0 * math.pi * freq_hz * float(t)))
        return torch.tensor([max(val, 1e-6)], dtype=torch.get_default_dtype())
    return fn


def _to_events_list(spikes: List[Tuple[float, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(spikes) == 0:
        return torch.empty(0, dtype=torch.get_default_dtype()), torch.empty(0, dtype=torch.long)
    times = torch.tensor([t for (t, _) in spikes], dtype=torch.get_default_dtype())
    marks = torch.tensor([m for (_, m) in spikes], dtype=torch.long)
    return times, marks


def main() -> None:
    seed_all(0)
    os.makedirs("results", exist_ok=True)
    env = capture_env(os.path.join("artifacts", "env.json"))

    # Poisson process
    T = 3.0
    lam = 0.8
    spikes_poisson = sample_ogata(_const_intensity(lam), T=T, lam_max=lam * 1.2, seed=42)
    events_poisson = _to_events_list(spikes_poisson)
    res_poisson = time_rescaling_test(lambda t: torch.tensor([lam], dtype=torch.get_default_dtype()), events_poisson)

    # Sinusoidal intensity
    lam0, amp, f = 0.8, 0.5, 1.0
    spikes_sine = sample_ogata(_sin_intensity(lam0, amp, f), T=T, lam_max=lam0 * (1 + amp) * 1.1, seed=43)
    events_sine = _to_events_list(spikes_sine)
    # For testing, we pass the true intensity function used for simulation
    res_sine = time_rescaling_test(lambda t: torch.tensor([lam0 * (1.0 + amp * math.sin(2.0 * math.pi * f * float(t)))], dtype=torch.get_default_dtype()), events_sine)

    payload = {
        "task": "ks_timescaling",
        "seed": 0,
        "commit": (env.get("git", {}) or {}).get("commit"),
        "env": {
            "platform": env.get("platform"),
            "python": env.get("python"),
            "versions": env.get("versions"),
            "torch": env.get("torch"),
        },
        "metrics": {
            "poisson_p": float(res_poisson.get("p_value", float("nan"))),
            "sinusoid_p": float(res_sine.get("p_value", float("nan"))),
        },
    }
    with open(os.path.join("results", "benchmark_ks_timescaling.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
