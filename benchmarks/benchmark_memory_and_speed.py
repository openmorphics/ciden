#!/usr/bin/env python3
"""
Benchmarks for sr_ciden: memory, thinning efficiency, and training throughput.

Usage examples:
  - python benchmarks/benchmark_memory_and_speed.py --bench all
  - python benchmarks/benchmark_memory_and_speed.py --bench memory
  - python benchmarks/benchmark_memory_and_speed.py --bench speed
  - python benchmarks/benchmark_memory_and_speed.py --bench efficiency
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Tuple

import json
import torch

# Determinism utilities (support both installed and source layout)
try:
    from sr_ciden.utils.determinism import seed_all, capture_env
except Exception:
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", "src"))
    from sr_ciden.utils.determinism import seed_all, capture_env

# Ensure local editable install or source layout works
try:
    from sr_ciden.adapters import CIDENContinuous
    from sr_ciden.solvers import ODESolverConfig
    from sr_ciden.readout import sample_ogata_with_stats
except Exception:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from sr_ciden.adapters import CIDENContinuous
    from sr_ciden.solvers import ODESolverConfig
    from sr_ciden.readout import sample_ogata_with_stats


def _patch_ode_forward_accept_1d(model: CIDENContinuous) -> None:
    """
    Some solver paths may pass a 1D hidden state to the ODE field. This patch mirrors
    the tests to ensure robustness in standalone scripts.
    """
    import types

    orig_forward = model.ode_core.forward

    def _patched_forward(self, h, t):
        if isinstance(h, torch.Tensor) and h.ndim == 1:
            return orig_forward(h.unsqueeze(0), t).squeeze(0)
        return orig_forward(h, t)

    model.ode_core.forward = types.MethodType(_patched_forward, model.ode_core)


def _format_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    sep = "  "
    def fmt_row(vals: List[str]) -> str:
        return sep.join(str(v).ljust(widths[i]) for i, v in enumerate(vals))
    lines = [fmt_row(headers), fmt_row(["-" * w for w in widths])]
    for r in rows:
        lines.append(fmt_row(r))
    return "\n".join(lines)


def _generate_dummy_events(T: float, K: int = 1, num_events: int = 3, device=None, dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
    t0 = float(T) * 0.1
    t1 = float(T) * 0.9
    times = torch.linspace(t0, t1, steps=num_events, device=device, dtype=dtype)
    marks = torch.zeros(num_events, dtype=torch.long, device=device)
    return times, marks


def benchmark_adjoint_memory() -> dict:
    if not torch.cuda.is_available():
        print("Adjoint Memory Scaling: CUDA GPU not available. Skipping.")
        return {"task": "adjoint_memory", "skipped": "no_cuda"}

    device = torch.device("cuda")
    cfg = ODESolverConfig(use_adjoint=True)
    model = CIDENContinuous(hidden_dim=8, num_marks=1, solver_config=cfg).to(device)
    _patch_ode_forward_accept_1d(model)
    model.train()
    model.reset_state(batch_size=1, device=device)

    T_values = [100.0, 500.0, 1000.0, 2000.0]
    rows: List[List[str]] = []

    for T in T_values:
        # Prepare sparse dummy events up to horizon T
        times, marks = _generate_dummy_events(T, K=1, num_events=3, device=device, dtype=next(model.parameters()).dtype)

        # Clear and measure peak memory during one full training step
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        model.zero_grad(set_to_none=True)
        loss = model((times, marks), T_end=float(T))
        loss.backward()
        torch.cuda.synchronize()

        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_mb = peak_bytes / (1024.0 * 1024.0)
        rows.append([f"{T:.0f}", f"{peak_mb:.2f}"])

        # Cleanup
        del loss

    print("=== Adjoint Memory Scaling (GPU) ===")
    print(_format_table(["Horizon (T)", "Peak GPU Memory (MB)"], rows))
    return {"task": "adjoint_memory", "rows": rows}


def benchmark_thinning_efficiency() -> dict:
    # Constant per-mark intensities; true maximum of the total rate is 0.7
    true_max = 0.7

    def intensity_fn(t: float) -> torch.Tensor:
        return torch.tensor([0.5, 0.2])

    T = 3.0
    multiples = [1.0, 1.2, 1.5, 2.0, 5.0]
    rows: List[List[str]] = []
    for m in multiples:
        lam_max = true_max * m
        _, stats = sample_ogata_with_stats(intensity_fn, T=T, lam_max=lam_max, seed=123)
        acc = float(stats["acceptance_ratio"])
        rows.append([f"{m:.1f}x", f"{acc:.3f}"])

    print("=== Thinning Efficiency (Ogata) ===")
    print(_format_table(["λ_max Tightness", "Acceptance Ratio"], rows))
    return {"task": "thinning_efficiency", "rows": rows}


def benchmark_training_throughput() -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 32
    K = 2
    N = 100
    T_end = 1.0

    cfg = ODESolverConfig(use_adjoint=False, fallback="rk4", dt=1e-2)
    model = CIDENContinuous(hidden_dim=8, num_marks=K, solver_config=cfg).to(device)
    _patch_ode_forward_accept_1d(model)
    model.train()
    model.reset_state(batch_size=B, device=device)

    # Build a representative batch: B sequences with N events each in [0, T_end)
    dtype = next(model.parameters()).dtype
    base_times = torch.linspace(1e-3, T_end - 1e-3, steps=N, device=device, dtype=dtype)
    events: List[Tuple[torch.Tensor, torch.Tensor]] = []
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    for _ in range(B):
        marks = torch.randint(low=0, high=K, size=(N,), dtype=torch.long, generator=g).to(device)
        events.append((base_times, marks))

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Warm-up
    warmup = 5
    for _ in range(warmup):
        opt.zero_grad(set_to_none=True)
        loss = model(events, T_end=T_end, reduction="mean")
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timed loop
    num_steps = 100
    t0 = time.perf_counter()
    for _ in range(num_steps):
        opt.zero_grad(set_to_none=True)
        loss = model(events, T_end=T_end, reduction="mean")
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_events = num_steps * B * N
    elapsed = max(t1 - t0, 1e-12)
    eps = total_events / elapsed
    device_str = "CUDA" if device.type == "cuda" else "CPU"
    print(f"=== Training Throughput ({device_str}) ===")
    print(f"Processed {total_events} events in {elapsed:.3f} s &#45;> {eps:,.0f} events/second")
    return {
        "task": "training_throughput",
        "events_per_second": float(eps),
        "elapsed_sec": float(elapsed),
        "total_events": int(total_events),
        "device": device_str,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="sr_ciden benchmarks: memory, thinning efficiency, throughput")
    parser.add_argument(
        "--bench",
        choices=["all", "memory", "speed", "efficiency"],
        default="all",
        help="Which benchmark to run.",
    )
    args = parser.parse_args()

    # Determinism, env capture, and output dirs
    seed_all(0)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("artifacts", "profiles"), exist_ok=True)
    env = capture_env(os.path.join("artifacts", "env.json"))

    results = {}
    if args.bench in ("all", "memory"):
        results["memory"] = benchmark_adjoint_memory()
    if args.bench in ("all", "efficiency"):
        results["efficiency"] = benchmark_thinning_efficiency()
    if args.bench in ("all", "speed"):
        results["speed"] = benchmark_training_throughput()

    payload = {
        "task": "benchmarks",
        "seed": 0,
        "commit": (env.get("git", {}) or {}).get("commit"),
        "env": {
            "platform": env.get("platform"),
            "python": env.get("python"),
            "versions": env.get("versions"),
            "torch": env.get("torch"),
        },
        "metrics": results,
    }
    out_name = "benchmarks_all.json" if args.bench == "all" else f"benchmark_{args.bench}.json"
    with open(os.path.join("results", out_name), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()