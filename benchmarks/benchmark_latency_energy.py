#!/usr/bin/env python3
"""
Latency and throughput benchmark for readout (Ogata thinning) and training step.

Outputs: results/benchmark_latency_energy.json
Schema: {task, seed, commit, env, metrics{ns_per_event, events_per_second, est_energy_j_per_event, device}}
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Tuple

import torch

# Local import with fallback to source tree
try:
    from sr_ciden.adapters import CIDENContinuous
    from sr_ciden.solvers import ODESolverConfig
    from sr_ciden.readout import sample_ogata
    from sr_ciden.utils.determinism import seed_all, capture_env
except Exception:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from sr_ciden.adapters import CIDENContinuous
    from sr_ciden.solvers import ODESolverConfig
    from sr_ciden.readout import sample_ogata
    from sr_ciden.utils.determinism import seed_all, capture_env


def _constant_intensity(lam: float) -> Callable[[float], torch.Tensor]:
    lam_t = torch.tensor([lam], dtype=torch.get_default_dtype())
    def fn(t: float) -> torch.Tensor:
        return lam_t
    return fn


def main() -> None:
    seed_all(0)
    os.makedirs("results", exist_ok=True)
    env = capture_env(os.path.join("artifacts", "env.json"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = "CUDA" if device.type == "cuda" else "CPU"

    # Readout throughput: simulate many small sequences
    intensity_fn = _constant_intensity(0.8)  # total rate
    T = 1.0
    lam_max = 1.2
    runs = 500

    t0 = time.perf_counter()
    total_events = 0
    for r in range(runs):
        spikes = sample_ogata(intensity_fn, T=T, lam_max=lam_max, seed=1000 + r)
        total_events += len(spikes)
    t1 = time.perf_counter()
    elapsed = max(t1 - t0, 1e-12)
    eps_readout = total_events / elapsed
    ns_per_event = (elapsed / max(total_events, 1)) * 1e9

    # Training latency proxy on small batch
    cfg = ODESolverConfig(use_adjoint=False, fallback="rk4", dt=1e-2)
    model = CIDENContinuous(hidden_dim=8, num_marks=1, solver_config=cfg).to(device)
    B, N, T_end = 8, 64, 1.0
    dtype = torch.get_default_dtype()
    base_times = torch.linspace(1e-3, T_end - 1e-3, steps=N, device=device, dtype=dtype)
    events = [(base_times, torch.zeros(N, dtype=torch.long, device=device))]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Warm-up
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = model(events, T_end=T_end, reduction="mean")
        loss.backward()
        opt.step()
    # Timed
    steps = 20
    t2 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = model(events, T_end=T_end, reduction="mean")
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
    t3 = time.perf_counter()
    train_elapsed = max(t3 - t2, 1e-12)
    train_eps = (steps * B * N) / train_elapsed

    # Crude energy estimate using assumed power (J = W * s)
    assumed_power_watts = 30.0 if device.type == "cpu" else 60.0
    est_energy_j_per_event = (assumed_power_watts * elapsed) / max(total_events, 1)

    payload = {
        "task": "latency_energy",
        "seed": 0,
        "commit": (env.get("git", {}) or {}).get("commit"),
        "env": {
            "platform": env.get("platform"),
            "python": env.get("python"),
            "versions": env.get("versions"),
            "torch": env.get("torch"),
        },
        "metrics": {
            "device": device_str,
            "readout_events": int(total_events),
            "readout_runs": int(runs),
            "readout_events_per_second": float(eps_readout),
            "readout_ns_per_event": float(ns_per_event),
            "train_events_per_second": float(train_eps),
            "est_energy_j_per_event": float(est_energy_j_per_event),
            "assumed_power_watts": float(assumed_power_watts),
        },
    }
    with open(os.path.join("results", "benchmark_latency_energy.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
