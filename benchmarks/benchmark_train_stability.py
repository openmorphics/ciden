#!/usr/bin/env python3
"""
Train stability benchmark: track loss and gradient norms over epochs for SR-CIDEN.

Outputs: results/benchmark_train_stability.json
Schema: {task, seed, commit, env, metrics{loss_curve, grad_norm_curve, exploding_flags}}
"""
from __future__ import annotations

import json
import os
import time
from typing import List, Tuple

import torch

# Local import with fallback to source tree
try:
    from sr_ciden.adapters import CIDENContinuous
    from sr_ciden.solvers import ODESolverConfig
    from sr_ciden.utils.determinism import seed_all, capture_env
except Exception:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from sr_ciden.adapters import CIDENContinuous
    from sr_ciden.solvers import ODESolverConfig
    from sr_ciden.utils.determinism import seed_all, capture_env


def _make_batch(B: int, K: int, N: int, T_end: float, device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    dtype = torch.get_default_dtype()
    base_times = torch.linspace(1e-3, T_end - 1e-3, steps=N, device=device, dtype=dtype)
    events: List[Tuple[torch.Tensor, torch.Tensor]] = []
    g = torch.Generator(device="cpu").manual_seed(0)
    for _ in range(B):
        marks = torch.randint(low=0, high=K, size=(N,), dtype=torch.long, generator=g).to(device)
        events.append((base_times, marks))
    return events


def main() -> None:
    seed_all(0)
    os.makedirs("results", exist_ok=True)
    env = capture_env(os.path.join("artifacts", "env.json"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ODESolverConfig(use_adjoint=False, fallback="rk4", dt=1e-2)
    model = CIDENContinuous(hidden_dim=8, num_marks=2, solver_config=cfg).to(device)

    B, N, T_end = 16, 64, 1.0
    batch = _make_batch(B=B, K=2, N=N, T_end=T_end, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    loss_curve: List[float] = []
    grad_norm_curve: List[float] = []
    exploding_flags: List[bool] = []

    epochs = 25
    for ep in range(epochs):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = model(batch, T_end=T_end, reduction="mean")
        loss.backward()
        # Grad norm
        total_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_sq += float(p.grad.detach().norm().item() ** 2)
        grad_norm = total_sq ** 0.5
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        loss_curve.append(float(loss.detach().item()))
        grad_norm_curve.append(float(grad_norm))
        exploding_flags.append(bool(grad_norm > 1e3 or not torch.isfinite(loss)))

        # Optional: print a compact line
        print(f"ep={ep:02d} loss={loss_curve[-1]:.4f} grad_norm={grad_norm_curve[-1]:.2f} dt={t1-t0:.3f}s")

    payload = {
        "task": "train_stability",
        "seed": 0,
        "commit": (env.get("git", {}) or {}).get("commit"),
        "env": {
            "platform": env.get("platform"),
            "python": env.get("python"),
            "versions": env.get("versions"),
            "torch": env.get("torch"),
        },
        "metrics": {
            "loss_curve": loss_curve,
            "grad_norm_curve": grad_norm_curve,
            "exploding_flags": exploding_flags,
        },
    }
    with open(os.path.join("results", "benchmark_train_stability.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
