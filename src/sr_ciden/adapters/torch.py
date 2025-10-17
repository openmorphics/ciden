"""PyTorch adapter modules bridging sr_ciden functional core with nn.Module.

This adapter provides:
- CIDENContinuous: a stateful nn.Module that encapsulates the core continuous-time
  model components for training via exact likelihood (nll_continuous). It holds
  the ODE field, optional jump update, and a readout that maps hidden states to
  per-mark intensities. Its forward computes the loss; it does not perform sampling.
- SRReadoutTorch: a minimal discrete-time Bernoulli readout utility for inference
  that samples spikes from instantaneous intensities on a Δt grid. Sampling is
  non-differentiable and detached from autograd.

Training–inference decoupling
- CIDENContinuous.forward computes the continuous-time NLL for training.
- Inference should use CIDENContinuous.get_intensity_fn to obtain a forward-only
  callable intensity_function(t), which can be fed into sr_ciden.readout samplers.
- SRReadoutTorch is purely for discrete-time inference convenience and never
  participates in autograd.

No top-level code executes on import.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor

from sr_ciden.dynamics import LinearNonlinearODE, IntensityHead, JumpMLP
from sr_ciden.solvers import ODESolverConfig, odeint_wrapper
from sr_ciden.losses import nll_continuous

__all__ = ["CIDENContinuous", "SRReadoutTorch"]


class CIDENContinuous(nn.Module):
    """Stateful continuous-time training module for C-IDEN.

    This module wraps sr_ciden's functional components into a conventional
    nn.Module for ease of use in standard PyTorch training loops. It owns:
      - a Neural ODE field over hidden state h(t) via LinearNonlinearODE
      - a marked-intensity readout (K channels) via IntensityHead(s)
      - an optional jump update network via JumpMLP

    Responsibilities
    - forward(...): compute exact continuous-time negative log-likelihood (NLL)
      via sr_ciden.losses.nll_continuous, wiring its submodules to the loss.
    - get_intensity_fn(...): return an inference-only callable that evolves and
      updates an internal hidden state by integrating the ODE between calls and
      returns the current marked intensities. The callable is wrapped in
      torch.no_grad() to guarantee no gradient tracking.

    Notes
    - Training and inference are deliberately decoupled. This class does not
      implement any spike sampling; use sr_ciden.readout functions with the
      callable produced by get_intensity_fn.
    - The module holds a persistent buffer h0 that can be reset to shape [B, H]
      via reset_state; forward will pass this h0 to the loss.

    Parameters
    ----------
    hidden_dim:
        Hidden/state dimension H.
    num_marks:
        Number of marked intensity channels K.
    feature_dim:
        Event feature dimension X for jump updates U(x_i, h^-). If 0, no jump net.
    solver_config:
        Optional ODESolverConfig controlling integration backend/tolerances for
        likelihood evaluation. If None, a default config is used.
    **kwargs:
        Forwarded to LinearNonlinearODE (e.g., use_stable_A, nonlinearity, alpha_init, x_dim).

    Examples (not executed)
    -----------------------
    The typical training loop calls model(events, T_end) inside the forward pass
    to obtain a loss scalar, then optimizes parameters via backprop.

    Inference:
        intensity_fn = model.get_intensity_fn(h_init=None)
        # Pass intensity_fn(t: float) to readout.sample_ogata(...) or sample_bernoulli(...)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_marks: int,
        feature_dim: int = 0,
        solver_config: Optional[ODESolverConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_marks = int(num_marks)
        self.feature_dim = int(feature_dim)

        # Core ODE (captures parameters; optional kwargs for stability etc.)
        # If external continuous input is desired, set via ode_core.set_input_provider(...)
        self.ode_core = LinearNonlinearODE(dim=self.hidden_dim, **kwargs)

        # Intensity readout: for K==1 a single head; for K>1, a ModuleList of heads
        if self.num_marks <= 0:
            raise ValueError("num_marks must be a positive integer")
        if self.num_marks == 1:
            self.intensity_head: nn.Module = IntensityHead(state_dim=self.hidden_dim)
        else:
            self.intensity_head = nn.ModuleList(
                [IntensityHead(state_dim=self.hidden_dim) for _ in range(self.num_marks)]
            )

        # Optional jump network (per-event update). If feature_dim==0, no jump.
        self.jump_net: Optional[JumpMLP]
        if self.feature_dim > 0:
            self.jump_net = JumpMLP(state_dim=self.hidden_dim, input_dim=self.feature_dim)
        else:
            self.jump_net = None

        # Solver configuration retained for loss and inference integration
        self.config: ODESolverConfig = solver_config if solver_config is not None else ODESolverConfig()

        # Persistent initial hidden state buffer (resettable)
        self.register_buffer("h0", torch.zeros(1, self.hidden_dim))

    def reset_state(self, batch_size: int, device: Optional[torch.device] = None) -> Tensor:
        """Reset the persistent initial hidden state buffer.

        Parameters
        ----------
        batch_size:
            Desired batch size B for the reset state.
        device:
            Optional device to place the state on. Defaults to the module's
            parameter device if not provided.

        Returns
        -------
        Tensor
            The new h0 buffer of shape [B, H].
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        dev = device if device is not None else next(self.parameters()).device
        self.h0 = torch.zeros(int(batch_size), self.hidden_dim, device=dev, dtype=self.h0.dtype)
        return self.h0

    def _lambda_fn(self, h: Tensor, t: Tensor) -> Tensor:
        """Compute per-mark intensities λ(h,t) with shape [..., K]."""
        if isinstance(self.intensity_head, IntensityHead):
            lam = self.intensity_head(h)  # [..., 1]
            # Expand to K=1 output
            return lam
        else:
            # Concatenate K scalar heads
            outs = [head(h) for head in self.intensity_head]  # K tensors [...,1]
            return torch.cat(outs, dim=-1)  # [..., K]

    def forward(self, events: Any, T_end: float, reduction: str = "mean") -> Tensor:
        """Compute the continuous-time NLL loss for training.

        This is designed to be called inside a training loop. It wires the
        module's subcomponents into sr_ciden.losses.nll_continuous.

        Parameters
        ----------
        events:
            Event container accepted by nll_continuous: either (times, marks) or
            a list of such tuples for batched sequences.
        T_end:
            Horizon (float) for the integral upper limit.
        reduction:
            Reduction over batch: one of {'mean','sum','none'}. Default 'mean'.

        Returns
        -------
        Tensor
            Loss tensor per reduction.
        """
        loss = nll_continuous(
            f_h=self.ode_core,
            lambda_fn=self._lambda_fn,
            h0=self.h0,
            events=events,
            T_end=float(T_end),
            jump_update=self.jump_net,  # may be None
            K=self.num_marks,
            config=self.config,
            reduction=reduction,
        )
        return loss

    def get_intensity_fn(self, h_init: Optional[Tensor] = None) -> Callable[[float], Tensor]:
        """Return an inference-only callable intensity_function(t) -> Tensor[K].

        The returned function:
        - Maintains an internal hidden state h, initialized from h_init if provided,
          else from this module's current h0 (broadcast to [1, H] if needed).
        - Integrates the ODE from the last queried time to the new time t on each call,
          updating the internal state.
        - Computes and returns the current marked intensities as a 1D tensor [K].
        - Is fully wrapped in torch.no_grad() to avoid constructing autograd graphs.

        Constraints
        - This helper is intended for single-sequence inference (batch size 1). If
          h_init has leading dimension > 1, a ValueError is raised.
        - Times passed to the returned function must be nondecreasing.

        Parameters
        ----------
        h_init:
            Optional initial hidden state of shape [H] or [1, H]. If None, uses self.h0.

        Returns
        -------
        intensity_function:
            Callable taking a Python float time and returning a 1D tensor [K] of
            nonnegative intensities.
        """
        dev = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Prepare initial hidden state
        if h_init is None:
            h0_local = self.h0
        else:
            h0_local = h_init

        if h0_local.ndim == 1:
            h_state = h0_local.reshape(1, -1)
        elif h0_local.ndim == 2 and h0_local.shape[0] == 1:
            h_state = h0_local
        else:
            raise ValueError("get_intensity_fn supports only a single sequence; h_init must be [H] or [1, H].")

        # Ensure proper device/dtype and detach from any graph
        h_state = h_state.detach().to(device=dev, dtype=dtype)
        last_t = torch.zeros((), device=dev, dtype=dtype)

        def intensity_function(t: float) -> Tensor:
            nonlocal h_state, last_t
            with torch.no_grad():
                t_curr = torch.as_tensor(float(t), device=dev, dtype=dtype).reshape(())
                # Allow tiny backward perturbations from adaptive solvers; do not integrate backward.
                eps_time = torch.as_tensor(1e-12, device=dev, dtype=dtype)
                if bool(((t_curr + eps_time) < last_t).item()):
                    # Don't integrate backward; evaluate at current state/time.
                    t_curr = last_t
                # Integrate only if time advanced
                if bool((t_curr > last_t).item()):
                    t_eval = torch.stack([last_t, t_curr], dim=0)  # [2]
                    h_traj = odeint_wrapper(self.ode_core, h_state, t_eval, config=self.config)  # [2, 1, H]
                    h_state = h_traj[-1]  # [1, H]
                    last_t = t_curr

                lam = self._lambda_fn(h_state, last_t)  # [1, K]
                return lam[0].detach()  # [K]

        return intensity_function


class SRReadoutTorch(nn.Module):
    """Discrete-time Bernoulli readout wrapper for inference.

    This convenience module samples spikes from instantaneous intensities on a
    Δt grid using p = 1 - exp(-λ * dt). The operation is non-differentiable and
    is explicitly detached from autograd.

    This is intended for inference only and complements continuous-time readout
    utilities in sr_ciden.readout (e.g., sample_ogata). It does not integrate
    the ODE; you must supply instantaneous intensities at the desired grid.

    Parameters
    ----------
    seed:
        Optional integer seed. When provided, sampling uses a local torch.Generator
        for deterministic behavior independent of global RNG state.

    Forward
    -------
    lam_t: Tensor[B, K]
        Instantaneous nonnegative intensities per batch and mark.
    dt: float
        Time step size (seconds or arbitrary units).

    Returns
    -------
    Tensor[B, K]
        Binary spike indicators sampled from Bernoulli(p), detached from the graph.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__()
        self.seed: Optional[int] = int(seed) if seed is not None else None

    def forward(self, lam_t: Tensor, dt: float) -> Tensor:
        """Sample binary spikes given instantaneous intensities on a Δt grid."""
        if lam_t.ndim != 2:
            raise ValueError(f"lam_t must have shape [B, K]; got {tuple(lam_t.shape)}")
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        with torch.no_grad():
            p = 1.0 - torch.exp(-lam_t * float(dt))
            p = torch.clamp(p, min=0.0, max=1.0 - 1e-12)

            if self.seed is not None:
                gen = torch.Generator(device=lam_t.device)
                gen.manual_seed(self.seed)
                u = torch.rand(lam_t.shape, device=lam_t.device, generator=gen, dtype=lam_t.dtype)
            else:
                u = torch.rand_like(p)

            spikes = (u < p).to(dtype=lam_t.dtype)
            return spikes.detach()
