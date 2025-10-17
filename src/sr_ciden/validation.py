from __future__ import annotations

"""Goodness-of-fit validation utilities based on the Time-Rescaling Theorem.

This module implements tools to assess whether a fitted temporal point process
(TPP) model is consistent with observed event data. The Time-Rescaling Theorem
states that if a model's conditional intensity λ_k(t) over K marks is correct,
then the transformed event times

    τ_i = ∫_0^{t_i} ∑_{k=1}^K λ_k(u) du

are the arrival times of a unit-rate Poisson process. Consequently, the inter-event
intervals Δτ_i = τ_i - τ_{i-1} (with τ_0 := 0) are i.i.d. Exp(1). This property
enables a simple distributional check via a one-sample Kolmogorov–Smirnov (K-S) test
against the standard exponential distribution.

Scope
- This module provides analysis/validation utilities only; it does not train or sample.
- Integration is performed using the unified ODE interface in sr_ciden.solvers.

Examples (not executed)
-----------------------
    import torch
    from sr_ciden.validation import time_rescaling_test

    # Synthetic example with a constant total intensity of 2.0
    def intensity_fn(t: torch.Tensor) -> torch.Tensor:
        # K=1 in this toy example; any K>=1 is supported
        return torch.tensor([2.0], dtype=t.dtype, device=t.device)

    times = torch.tensor([0.1, 0.3, 0.9], dtype=torch.get_default_dtype())
    marks = torch.zeros_like(times, dtype=torch.long)  # not used here
    result = time_rescaling_test(intensity_fn, (times, marks))
    # Inspect result["p_value"]: a high value suggests good fit under the null.
"""

from typing import Callable, Optional, Tuple, Dict

import torch
from torch import Tensor
from .solvers import ODESolverConfig, odeint_wrapper


def time_rescale(
    intensity_fn: Callable[[Tensor], Tensor],
    events: Tuple[Tensor, Tensor],
    config: Optional[ODESolverConfig] = None,
) -> Tensor:
    """Compute rescaled event times τ_i via cumulative total intensity.

    Given an intensity function over K marks, λ(t) ∈ R^K with components λ_k(t),
    this computes

        τ_i = ∫_0^{t_i} ∑_{k=1}^K λ_k(u) du

    for each observed event time t_i. The integral is evaluated by numerically
    integrating the scalar ODE dI/dt = ∑_k λ_k(t) with initial condition I(0)=0.
    Integration uses the unified solver wrapper in sr_ciden.solvers.

    Contract for intensity_fn
    - Callable taking a scalar time t: Tensor[] -> Tensor[K]
      (0-d time tensor to a 1D tensor of K nonnegative intensities).
    - The returned tensor is expected to be broadcast-free and device/dtype
      compatible with t; this function will handle minor conversions if needed.

    Parameters
    ----------
    intensity_fn:
        Callable t -> Tensor[K] providing per-mark intensities at time t.
    events:
        Tuple (times, marks) where:
          - times: 1D tensor [N] of event times (monotone nondecreasing, t >= 0).
          - marks: 1D tensor [N] of integer marks. The values are not used by this
            routine but are included for API uniformity.
    config:
        Optional ODESolverConfig to control the integration backend and tolerances.

    Returns
    -------
    τ:
        1D tensor [N] of cumulative rescaled times evaluated at each t_i.

    Examples (not executed)
    -----------------------
    >>> import torch
    >>> from sr_ciden.validation import time_rescale
    >>> def lam(t):  # K=2 constant intensities
    ...     return torch.tensor([1.0, 1.5], dtype=t.dtype, device=t.device)
    >>> times = torch.tensor([0.2, 0.5])
    >>> marks = torch.zeros_like(times, dtype=torch.long)
    >>> tau = time_rescale(lam, (times, marks))
    >>> # Here, total intensity is 2.5, so tau ≈ 2.5 * times
    """
    times, _marks = events

    if times.ndim != 1:
        raise ValueError(f"events[0] (times) must be 1D, got shape {tuple(times.shape)}")
    if times.numel() == 0:
        return times.clone()  # empty input -> empty output

    device = times.device
    dtype = times.dtype

    # Build evaluation grid including t=0 to get I(0)=0 as the first entry.
    t0 = torch.zeros(1, device=device, dtype=dtype)
    t_eval = torch.cat([t0, times], dim=0)

    # Scalar integral state I(t) with I(0)=0
    I0 = torch.zeros(1, device=device, dtype=dtype)

    def dI_dt(_I: Tensor, t: Tensor) -> Tensor:
        # Sum across mark intensities at time t and return as shape [1]
        lam_t = intensity_fn(t)
        if lam_t.ndim == 0:
            total = lam_t
        else:
            total = lam_t.sum()
        return total.to(device=device, dtype=dtype).reshape(1)

    I_traj = odeint_wrapper(dI_dt, I0, t_eval, config=config)  # [T, 1]
    # Drop the I(0) entry; retain values at observed times only
    tau = I_traj[1:, 0]
    return tau


def ks_test(rescaled_inter_event_times: Tensor) -> Dict[str, float]:
    """One-sample Kolmogorov–Smirnov test against Exp(1) for Δτ samples.

    Under the null hypothesis (model is correctly specified), the rescaled
    inter-event intervals Δτ are i.i.d. Exp(1). This function applies
    scipy.stats.kstest to compare the provided sample against the standard
    exponential distribution.

    Parameters
    ----------
    rescaled_inter_event_times:
        1D tensor of inter-event intervals Δτ (i.e., torch.diff(τ) with τ_0=0).

    Returns
    -------
    result:
        Dict with keys:
          - "D": K-S statistic (float)
          - "p_value": p-value under the null (float)

    Examples (not executed)
    -----------------------
    >>> import torch
    >>> from sr_ciden.validation import ks_test
    >>> x = torch.distributions.Exponential(rate=1.0).sample((100,))
    >>> ks_test(x)
    {'D': ..., 'p_value': ...}
    """
    from scipy.stats import kstest as _kstest

    x = rescaled_inter_event_times
    if x.ndim != 1:
        raise ValueError(f"rescaled_inter_event_times must be 1D, got shape {tuple(x.shape)}")
    if x.numel() == 0:
        # With no data, KS is undefined; return degenerate values.
        return {"D": float("nan"), "p_value": float("nan")}

    sample = x.detach().to(dtype=torch.float64).cpu().numpy()  # (N,)
    # Compare to standard exponential (loc=0, scale=1)
    res = _kstest(sample, "expon")
    return {"D": float(res.statistic), "p_value": float(res.pvalue)}


def time_rescaling_test(
    intensity_fn: Callable[[Tensor], Tensor],
    events: Tuple[Tensor, Tensor],
    config: Optional[ODESolverConfig] = None,
) -> Dict[str, float]:
    """End-to-end goodness-of-fit test via time rescaling and K-S statistic.

    This orchestrates the standard TPP validation pipeline:
      1) Compute cumulative rescaled times τ_i = ∫_0^{t_i} ∑_k λ_k(u) du.
      2) Form inter-event intervals Δτ_i with Δτ_1 = τ_1 - 0 via torch.diff
         and a zero prepend.
      3) Apply a one-sample K-S test against the Exp(1) distribution.

    Interpretation
    - Null hypothesis H0: The model is correctly specified so that Δτ_i are i.i.d. Exp(1).
    - A large p-value (e.g., > 0.05) indicates no evidence to reject H0 (good fit).
    - A very small p-value suggests model misspecification.

    Parameters
    ----------
    intensity_fn:
        Callable t -> Tensor[K] returning per-mark intensities at time t.
    events:
        Tuple (times, marks), where times is 1D [N]. Marks are not used here.
    config:
        Optional ODESolverConfig for numerical integration.

    Returns
    -------
    result:
        Dictionary with keys "D" and "p_value" from the K-S test.

    Examples (not executed)
    -----------------------
    >>> import torch
    >>> from sr_ciden.validation import time_rescaling_test
    >>> def lam(t):  # total intensity ~ 1.0
    ...     return torch.tensor([1.0], dtype=t.dtype, device=t.device)
    >>> times = torch.linspace(0.1, 1.0, 10)
    >>> marks = torch.zeros_like(times, dtype=torch.long)
    >>> out = time_rescaling_test(lam, (times, marks))
    >>> out  # {'D': ..., 'p_value': ...}
    """
    tau = time_rescale(intensity_fn, events, config=config)  # [N]
    d0 = torch.zeros(1, device=tau.device, dtype=tau.dtype)
    inter = torch.diff(tau, prepend=d0)  # [N]
    return ks_test(inter)


__all__ = ["time_rescale", "ks_test", "time_rescaling_test"]
