"""ODE solver wrappers with optional adjoint support and a safe fixed-step fallback.

This module provides a minimal and consistent integration interface for continuous-time
dynamics used in C-IDEN. It prefers torchdiffeq (if available) and can optionally use
its adjoint method for memory-efficient training. When torchdiffeq is not available,
it falls back to deterministic, pure-PyTorch fixed-step solvers (RK4 or Euler).

Key points
- Training–inference decoupling: These solvers only integrate continuous ODE dynamics
  and optionally accumulate integrals. They do NOT sample spikes or perform SR readout.
- Backends:
  - torchdiffeq (if installed): high-accuracy adaptive solvers with optional adjoint.
  - Fallback: deterministic fixed-step RK4 (default) or Euler using only PyTorch ops.

Examples (not executed)

    import torch
    from sr_ciden.solvers import ODESolverConfig, odeint_wrapper, make_augmented_field, odeint_augmented

    # Basic ODE integration -----------------------------------------------------
    D = 4
    def f(h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Simple linear decay
        return -0.5 * h

    y0 = torch.zeros(3, D)                 # [..., D]
    t_eval = torch.linspace(0.0, 1.0, 11)  # 11 evaluation times

    states = odeint_wrapper(f, y0, t_eval)    # [T, ..., D]

    # Adjoint with torchdiffeq if available
    cfg = ODESolverConfig(use_adjoint=True, method="dopri5", rtol=1e-5, atol=1e-7)
    states_adj = odeint_wrapper(f, y0, t_eval, cfg)  # requires torchdiffeq installed

    # Augmented integration for accumulating integrals -------------------------
    # Suppose lambda_fn returns a per-dimension nonnegative intensity (K dims).
    K = 2
    def lambda_fn(h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Example: positive values via clamp
        return torch.clamp(h[..., :K], min=0) + 1e-6

    h_traj, I_traj = odeint_augmented(f, lambda_fn, y0, K=K, t_eval=t_eval)
    # h_traj: [T, ..., D], I_traj: [T, ..., K]

No top-level code executes on import.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

Tensor = torch.Tensor


def has_torchdiffeq() -> bool:
    """Return True if torchdiffeq is importable at runtime, else False.

    Notes
    - This performs a light import check via importlib and a guarded import.
    - The rest of the module does not require torchdiffeq unless you call
      functions that explicitly use it (e.g., odeint_wrapper with the backend).

    Example
    -------
    if has_torchdiffeq():
        # Safe to use adjoint / torchdiffeq solvers
        ...
    """
    try:
        import importlib.util

        spec = importlib.util.find_spec("torchdiffeq")
        if spec is None:
            return False
        # Ensure the module actually imports
        import torchdiffeq  # noqa: F401

        return True
    except Exception:
        return False


@dataclass
class ODESolverConfig:
    """Configuration for ODE integration.

    Parameters
    ----------
    use_adjoint:
        If True and torchdiffeq is available, use odeint_adjoint instead of odeint.
        Ignored by the fallback solvers.
    method:
        Method string for torchdiffeq (e.g., "dopri5", "rk4", "euler", "bosh3", ...).
        Ignored by the fallback, which is controlled by `fallback`.
    rtol:
        Relative tolerance for adaptive solvers in torchdiffeq.
    atol:
        Absolute tolerance for adaptive solvers in torchdiffeq.
    max_steps:
        Safety cap on the maximum number of internal steps for torchdiffeq.
        Passed via options={"max_num_steps": max_steps}.
    dt:
        Fixed step size for the deterministic PyTorch fallback.
    fallback:
        Which fallback integrator to use when torchdiffeq is unavailable.
        One of {"rk4", "euler"}.

    Usage note
    ----------
    For deterministic experiments without torchdiffeq, set `dt` explicitly
    and adjust it for accuracy/stability. RK4 generally provides a good trade-off.
    """

    use_adjoint: bool = True
    method: str = "dopri5"
    rtol: float = 1e-5
    atol: float = 1e-7
    max_steps: int = 100000
    dt: float = 1e-3
    fallback: str = "rk4"


def _check_time_grid(t_eval: Tensor) -> None:
    if t_eval.ndim != 1:
        raise ValueError(f"t_eval must be 1D, got shape {tuple(t_eval.shape)}")
    if t_eval.numel() == 0:
        raise ValueError("t_eval must have at least one time point")
    if t_eval.numel() > 1:
        diffs = t_eval[1:] - t_eval[:-1]
        if torch.any(diffs < 0):
            raise ValueError("t_eval must be monotonically nondecreasing")


def _as_time_scalar(x: Tensor, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Ensure a 0-dim tensor on the desired device/dtype."""
    if x.ndim == 0:
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype).reshape(())


def euler_solve(
    f: Callable[[Tensor, Tensor], Tensor],
    y0: Tensor,
    t_eval: Tensor,
    dt: float,
) -> Tensor:
    """Deterministic fixed-step forward Euler solver (pure PyTorch).

    Integrates from t_eval[0] to t_eval[-1] and returns states evaluated exactly
    at each t_eval[i]. For each interval [t_i, t_{i+1}], the solver takes
    n = ceil((t_{i+1}-t_i)/dt) internal steps with step size h = (t_{i+1}-t_i)/n
    to land precisely on t_{i+1}, avoiding drift.

    Parameters
    ----------
    f:
        Callable computing dh/dt = f(h, t) with signature (h, t) -> Tensor
        matching the shape of h.
    y0:
        Initial state tensor of shape [..., D]. The last dimension is treated as
        the state dimension D; leading dimensions are preserved.
    t_eval:
        1D tensor of evaluation times (monotone nondecreasing).
    dt:
        Nominal internal step size used to partition each interval.

    Returns
    -------
    states:
        Tensor of shape [len(t_eval), ..., D].

    Example
    -------
    states = euler_solve(f, y0, t_eval, dt=1e-3)
    """
    _check_time_grid(t_eval)
    device = y0.device
    dtype = y0.dtype
    t_eval = t_eval.to(device=device, dtype=dtype)

    T = t_eval.numel()
    states = []
    y = y0.clone()
    t_curr = _as_time_scalar(t_eval[0], device, dtype)

    # Record initial
    states.append(y)

    for i in range(1, T):
        t_next = _as_time_scalar(t_eval[i], device, dtype)
        delta = t_next - t_curr
        if bool((delta < 0).item()):
            raise ValueError("t_eval must be nondecreasing")

        if bool((delta == 0).item()):
            # No advance in time; duplicate the state
            states.append(y)
            continue

        dt_t = torch.as_tensor(float(dt), device=device, dtype=dtype)
        n_steps = int(torch.ceil(delta / dt_t).item())
        n_steps = max(1, n_steps)
        h = delta / n_steps  # 0-dim tensor

        for _ in range(n_steps):
            k1 = f(y, t_curr)
            if k1.shape != y.shape:
                raise RuntimeError(f"f(y,t) shape {tuple(k1.shape)} does not match y shape {tuple(y.shape)}")
            y = y + h * k1
            t_curr = t_curr + h

        # Now exactly at t_next
        states.append(y)

    return torch.stack(states, dim=0)


def rk4_solve(
    f: Callable[[Tensor, Tensor], Tensor],
    y0: Tensor,
    t_eval: Tensor,
    dt: float,
) -> Tensor:
    """Deterministic fixed-step RK4 solver (pure PyTorch).

    For each interval [t_i, t_{i+1}] in the requested evaluation grid, this routine
    performs n = ceil((t_{i+1}-t_i)/dt) internal sub-steps with size
    h = (t_{i+1}-t_i)/n to hit t_{i+1} exactly.

    Parameters
    ----------
    f:
        Callable computing dh/dt = f(h, t) with signature (h, t) -> Tensor
        matching the shape of h.
    y0:
        Initial state tensor of shape [..., D].
    t_eval:
        1D tensor of evaluation times (monotone nondecreasing).
    dt:
        Nominal internal step size.

    Returns
    -------
    states:
        Tensor of shape [len(t_eval), ..., D].

    Example
    -------
    states = rk4_solve(f, y0, t_eval, dt=1e-3)
    """
    _check_time_grid(t_eval)
    device = y0.device
    dtype = y0.dtype
    t_eval = t_eval.to(device=device, dtype=dtype)

    T = t_eval.numel()
    states = []
    y = y0.clone()
    t_curr = _as_time_scalar(t_eval[0], device, dtype)

    states.append(y)

    for i in range(1, T):
        t_next = _as_time_scalar(t_eval[i], device, dtype)
        delta = t_next - t_curr
        if bool((delta < 0).item()):
            raise ValueError("t_eval must be nondecreasing")

        if bool((delta == 0).item()):
            states.append(y)
            continue

        dt_t = torch.as_tensor(float(dt), device=device, dtype=dtype)
        n_steps = int(torch.ceil(delta / dt_t).item())
        n_steps = max(1, n_steps)
        h = delta / n_steps

        for _ in range(n_steps):
            k1 = f(y, t_curr)
            k2 = f(y + 0.5 * h * k1, t_curr + 0.5 * h)
            k3 = f(y + 0.5 * h * k2, t_curr + 0.5 * h)
            k4 = f(y + h * k3, t_curr + h)

            if not (k1.shape == k2.shape == k3.shape == k4.shape == y.shape):
                raise RuntimeError("f(y,t) returns shape inconsistent with y for RK4")

            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t_curr = t_curr + h

        states.append(y)

    return torch.stack(states, dim=0)


def odeint_wrapper(
    f: Callable[[Tensor, Tensor], Tensor],
    y0: Tensor,
    t_eval: Tensor,
    config: Optional[ODESolverConfig] = None,
) -> Tensor:
    """Integrate an ODE y' = f(y, t) with a uniform interface and safe fallback.

    Behavior
    - If torchdiffeq is available:
        - Chooses odeint_adjoint if config.use_adjoint else odeint
        - Passes (method, rtol, atol, options={"max_num_steps": config.max_steps})
    - Else:
        - Uses a fixed-step fallback (RK4 or Euler) that integrates deterministically
          in pure PyTorch from t_eval[0] to t_eval[-1] and returns states at each
          requested time.

    Parameters
    ----------
    f:
        Callable computing dh/dt = f(h, t) with signature (h, t) -> Tensor
        matching y0's shape.
    y0:
        Initial state tensor of shape [..., D].
    t_eval:
        1D tensor of evaluation times (monotone nondecreasing). Returned trajectory
        has leading dimension len(t_eval).
    config:
        Optional ODESolverConfig. Defaults are chosen for practical stability.

    Returns
    -------
    states:
        Tensor of shape [len(t_eval), ..., D].

    Notes
    -----
    - This wrapper does not sample discrete events. It strictly integrates continuous
      dynamics and returns states at the requested times.
    - The function `f` must be deterministic and side-effect free for reproducibility.

    Example
    -------
    y_traj = odeint_wrapper(f, y0, t_eval)  # uses torchdiffeq if present, else fallback

    cfg = ODESolverConfig(use_adjoint=True, method="dopri5")
    y_traj = odeint_wrapper(f, y0, t_eval, cfg)
    """
    _check_time_grid(t_eval)
    device = y0.device
    dtype = y0.dtype
    t_eval = t_eval.to(device=device, dtype=dtype)
    cfg = config or ODESolverConfig()

    if has_torchdiffeq():
        # Import locally to avoid hard dependency at import time
        from torchdiffeq import odeint as _odeint  # type: ignore
        from torchdiffeq import odeint_adjoint as _odeint_adj  # type: ignore

        solver = _odeint_adj if cfg.use_adjoint else _odeint

        # torchdiffeq expects signature f(t, y); our f is f(y, t)
        def _backend_fn(t: Tensor, y: Tensor) -> Tensor:
            return f(y, t)

        options = {"max_num_steps": int(cfg.max_steps)}
        # Prepare kwargs and ensure adjoint_params is set when using adjoint.
        solver_kwargs = {
            "rtol": float(cfg.rtol),
            "atol": float(cfg.atol),
            "method": str(cfg.method),
            "options": options,
        }
        if cfg.use_adjoint:
            # If f is an nn.Module, pass its parameters; otherwise pass an empty tuple.
            # This satisfies torchdiffeq's requirement when func is not an nn.Module.
            if isinstance(f, nn.Module):
                solver_kwargs["adjoint_params"] = tuple(f.parameters())
            else:
                solver_kwargs["adjoint_params"] = ()

        y_traj = solver(_backend_fn, y0, t_eval, **solver_kwargs)
        return y_traj

    # Fallback path (pure PyTorch)
    if cfg.fallback.lower() == "euler":
        return euler_solve(f, y0, t_eval, dt=float(cfg.dt))
    elif cfg.fallback.lower() == "rk4":
        return rk4_solve(f, y0, t_eval, dt=float(cfg.dt))
    else:
        raise ValueError("config.fallback must be either 'rk4' or 'euler'")


def make_augmented_field(
    f_h: Callable[[Tensor, Tensor], Tensor],
    lambda_fn: Callable[[Tensor, Tensor], Tensor],
    K: int,
) -> Callable[[Tensor, Tensor], Tensor]:
    """Build an augmented ODE field to accumulate integrals of intensities.

    This helper creates a vector field F over z = concat(h, I), where:
        - h has dimension D (state)
        - I has dimension K (integral accumulators)
      and returns:
        F(z, t) = concat( f_h(h, t), lambda_fn(h, t) )

    The intended use is to accumulate exact integrals of nonnegative intensities
    commonly required in temporal point process (TPP) likelihoods, i.e.,
        I(t) = ∫ λ*(t) dt
    possibly across K marked dimensions. The user is responsible for ensuring
    positivity of lambda_fn if desired.

    Parameters
    ----------
    f_h:
        The base ODE field dh/dt = f_h(h, t), signature (h, t) -> Tensor[..., D].
    lambda_fn:
        Function computing intensities λ*(t) from h(t), signature (h, t) -> Tensor[..., K].
    K:
        Number of marked intensity dimensions to accumulate.

    Returns
    -------
    F:
        Callable over (z, t) where z[..., :D] is h and z[..., D:] is I. The output
        has the same shape as z, with last dimension D+K.

    Example
    -------
    F = make_augmented_field(f_h, lambda_fn, K=2)
    # Then integrate z' = F(z, t) to get both h(t) and accumulated I(t).
    """
    if K <= 0:
        raise ValueError("K must be a positive integer")

    def F(z: Tensor, t: Tensor) -> Tensor:
        if z.ndim < 1:
            raise ValueError("z must have at least 1 dimension with last dim = D+K")
        D_plus_K = z.shape[-1]
        D = D_plus_K - K
        if D <= 0:
            raise ValueError("Last dimension of z must be at least K larger than 0")

        h = z[..., :D]
        # I = z[..., D:]  # not needed in the derivative computation
        dh = f_h(h, t)
        lam = lambda_fn(h, t)
        if dh.shape != h.shape:
            raise RuntimeError("f_h(h,t) must return same shape as h")
        if lam.shape[:-1] != h.shape[:-1] or lam.shape[-1] != K:
            raise RuntimeError(f"lambda_fn(h,t) must return shape {h.shape[:-1] + (K,)}")

        return torch.cat([dh, lam], dim=-1)

    return F


def odeint_augmented(
    f_h: Callable[[Tensor, Tensor], Tensor],
    lambda_fn: Callable[[Tensor, Tensor], Tensor],
    h0: Tensor,
    K: int,
    t_eval: Tensor,
    config: Optional[ODESolverConfig] = None,
) -> Tuple[Tensor, Tensor]:
    """Integrate the augmented system to return both h(t) and integral accumulators.

    Constructs the augmented state z = concat(h, I) with I(0) = 0 in R^K, and integrates:
        z' = [ f_h(h, t), lambda_fn(h, t) ]
    over the requested time grid.

    Parameters
    ----------
    f_h:
        Base ODE field, (h, t) -> Tensor[..., D].
    lambda_fn:
        Intensity function, (h, t) -> Tensor[..., K].
    h0:
        Initial hidden state tensor [..., D].
    K:
        Number of integral channels to accumulate.
    t_eval:
        1D tensor of evaluation times (monotone nondecreasing).
    config:
        Optional ODESolverConfig for integration backend and tolerances.

    Returns
    -------
    h_traj:
        Tensor of shape [len(t_eval), ..., D].
    I_traj:
        Tensor of shape [len(t_eval), ..., K].

    Example
    -------
    h_traj, I_traj = odeint_augmented(f_h, lambda_fn, h0, K=2, t_eval=t_eval)
    """
    if K <= 0:
        raise ValueError("K must be a positive integer")
    device = h0.device
    dtype = h0.dtype

    # If h0 is 1D [D], lift to [1, D] for nn.Module compatibility during integration.
    expanded_batch = False
    if h0.ndim == 1:
        h0_in = h0.unsqueeze(0)
        expanded_batch = True
    else:
        h0_in = h0

    zeros_I0 = torch.zeros(*h0_in.shape[:-1], K, device=device, dtype=dtype)
    z0 = torch.cat([h0_in, zeros_I0], dim=-1)

    F = make_augmented_field(f_h, lambda_fn, K)
    z_traj = odeint_wrapper(F, z0, t_eval, config=config)

    D = h0_in.shape[-1]
    h_traj = z_traj[..., :D]
    I_traj = z_traj[..., D:]

    # If we lifted the batch dim, squeeze it back so callers see [T, D] / [T, K].
    if expanded_batch:
        h_traj = h_traj.squeeze(-2)
        I_traj = I_traj.squeeze(-2)

    return h_traj, I_traj


__all__ = [
    "ODESolverConfig",
    "has_torchdiffeq",
    "odeint_wrapper",
    "rk4_solve",
    "euler_solve",
    "make_augmented_field",
    "odeint_augmented",
]
