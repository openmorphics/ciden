#!/usr/bin/env python3
"""
Minimal training and inference example for sr_ciden.

This script demonstrates the decoupled API:
- Training uses CIDENContinuous.forward to compute the continuous-time NLL.
- Inference uses model.get_intensity_fn() with sampling utilities in sr_ciden.readout.
"""
import torch
import sr_ciden
from sr_ciden.adapters import CIDENContinuous
from sr_ciden.solvers import ODESolverConfig
from sr_ciden.readout import sample_ogata, sample_bernoulli


def main() -> None:
    torch.manual_seed(0)

    # Instantiate model (1 mark)
    model = CIDENContinuous(
        hidden_dim=8,
        num_marks=1,
        solver_config=ODESolverConfig(dt=1e-3, use_adjoint=False),
    )

    # Dummy event data: three events for mark 0
    dummy_times = torch.tensor([0.1, 0.4, 0.9], dtype=torch.get_default_dtype())
    dummy_marks = torch.zeros_like(dummy_times, dtype=torch.long)
    dummy_events = (dummy_times, dummy_marks)

    print(f"sr_ciden version: {sr_ciden.__version__}")

    # 1. Training
    # Reset the model's state and compute the loss with a single forward pass.
    print("### 1. Training ###")
    model.train()
    model.reset_state(batch_size=1)
    loss = model(dummy_events, T_end=1.0)
    print("Training entry point loss tensor:", loss)

    # 2. Inference (Decoupled from Training)
    print("### 2. Inference (Decoupled from Training) ###")
    model.eval()
    intensity_fn = model.get_intensity_fn()

    # Continuous-Time Sampling via Ogata's thinning
    spikes_ct = sample_ogata(intensity_fn, T=1.0, lam_max=10.0, seed=0)
    print("First few continuous-time spikes:", spikes_ct[:5])

    # Discrete-Time Sampling: Bernoulli on a dt grid
    spikes_dt = sample_bernoulli(intensity_fn, T=1.0, dt=0.01, seed=0)
    print("First few discrete-time spikes:", spikes_dt[:5])

    print("Note: readout functions do not support gradients and operate on the decoupled intensity_fn.")


if __name__ == "__main__":
    main()
