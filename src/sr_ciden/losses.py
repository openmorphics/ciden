"""Loss functions for exact-likelihood training and related objectives.

This module provides:
- nll_continuous: exact continuous-time negative log-likelihood (NLL) for marked
  temporal point processes (TPPs) using augmented ODE integration (accumulates the
  integral of intensities exactly via an augmented state).
- nll_discrete: a discrete-time Δt fallback NLL useful when intensities are
  precomputed on a grid.
- nll: a thin public wrapper that dispatches to nll_continuous when given a
  dict-like/object container with ODE components, otherwise raises a helpful error.

No top-level code executes on import (strict training–inference decoupling).
"""
from __future__ import annotations

from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from .solvers import (
    ODESolverConfig,
    odeint_wrapper,       # imported for API completeness; not directly used here
    make_augmented_field, # imported for API completeness; not directly used here
    odeint_augmented,
)


# ---------------------------------------------------------------------------
# Utilities (kept minimal and self-contained for this module)
# ---------------------------------------------------------------------------

def _safe_log(x: Tensor, eps: float) -> Tensor:
    """Numerically stable log with lower clamp.

    Parameters
    ----------
    x:
        Input tensor.
    eps:
        Small positive clamp to avoid log(0).

    Returns
    -------
    Tensor
        log(clamp_min(x, eps)), same shape as x.
    """
    return torch.log(x.clamp_min(eps))


def _ensure_batch_h0(h0: Tensor, B: int) -> Tensor:
    """Broadcast or validate h0 to shape [B, D].

    Parameters
    ----------
    h0:
        Initial hidden state of shape [D] or [B, D].
    B:
        Target batch size.

    Returns
    -------
    Tensor
        Tensor of shape [B, D].

    Raises
    ------
    ValueError
        If h0 has incompatible shape.
    """
    if h0.ndim == 1:
        return h0.unsqueeze(0).expand(B, -1)
    if h0.ndim == 2:
        if h0.shape[0] == B:
            return h0
        if h0.shape[0] == 1:
            return h0.expand(B, -1)
        raise ValueError(f"h0 has shape {tuple(h0.shape)} incompatible with batch size {B}")
    raise ValueError(f"h0 must be 1D or 2D, got shape {tuple(h0.shape)}")


def _normalize_events(
    events: Union[
        Tuple[Tensor, Tensor],
        Sequence[Tuple[Tensor, Tensor]],
    ]
) -> List[Tuple[Tensor, Tensor]]:
    """Normalize event containers to a per-batch list of (times, marks) tuples.

    Supported inputs
    ----------------
    - Single sequence (no batch): (times, marks)
      * times: 1D float Tensor [N_i], event times (monotone nondecreasing)
      * marks: 1D long Tensor [N_i], event marks in [0, K-1]
    - Per-batch sequences: list of (times_b, marks_b) for b in [0..B-1]

    Returns
    -------
    List[Tuple[Tensor, Tensor]]
        A list with length B, each element a tuple (times_b, marks_b), both 1D.

    Raises
    ------
    TypeError
        If the provided container format is unsupported.
    ValueError
        If shapes are inconsistent or times are not nondecreasing per sequence.
    """
    def _is_single_pair(obj: Any) -> bool:
        return (
            isinstance(obj, tuple)
            and len(obj) >= 2
            and isinstance(obj[0], Tensor)
            and isinstance(obj[1], Tensor)
        )

    if _is_single_pair(events):
        seqs = [events]  # type: ignore[list-item]
    elif isinstance(events, Sequence):
        seqs = []
        for i, item in enumerate(events):  # type: ignore[assignment]
            if not _is_single_pair(item):
                raise TypeError(
                    f"events[{i}] is not a (times, marks) tuple of Tensors; got type {type(item)}"
                )
            seqs.append(item)
    else:
        raise TypeError(
            "events must be either a (times, marks) tuple or a sequence of such tuples"
        )

    out: List[Tuple[Tensor, Tensor]] = []
    for (times, marks) in seqs:
        t = times.reshape(-1)
        m = marks.reshape(-1).to(dtype=torch.long)
        if t.numel() != m.numel():
            raise ValueError(f"times and marks must have equal length; got {t.numel()} vs {m.numel()}")

        if t.numel() > 1:
            diffs = t[1:] - t[:-1]
            if torch.any(diffs < 0):
                raise ValueError("Event times must be monotonically nondecreasing per sequence")

        out.append((t, m))
    return out


# ---------------------------------------------------------------------------
# Public losses
# ---------------------------------------------------------------------------

def nll_continuous(
    f_h: Callable[[Tensor, Tensor], Tensor],
    lambda_fn: Callable[[Tensor, Tensor], Tensor],
    h0: Tensor,
    events: Union[Tuple[Tensor, Tensor], Sequence[Tuple[Tensor, Tensor]]],
    T_end: float,
    *,
    jump_update: Optional[Callable[[Optional[Tensor], Tensor], Tensor]] = None,
    K: Optional[int] = None,
    config: Optional[ODESolverConfig] = None,
    reduction: str = "mean",
    eps: float = 1e-8,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Exact continuous-time marked-TPP NLL via augmented ODE integration.

    Computes the negative log-likelihood:
        NLL = -∑_i log λ_{m_i}(t_i) + ∫_0^T ∑_{k=1}^K λ_k(u) du

    using piecewise augmented ODE integration between observed event times. The
    augmented field integrates both the hidden state h(t) and integral channels I_k(t)
    such that, on each segment [t_prev, t_next], the increase ΔI_k equals the exact
    integral ∫_{t_prev}^{t_next} λ_k(u) du.

    Behavior
    --------
    - Piecewise integrate over segments 0 → t_1 → … → t_{N} → T_end:
      * For each segment [t_prev, t_next], integrate the augmented system:
            z' = concat( f_h(h, t), lambda_fn(h, t) )
        with I initialized to 0 on the segment, obtaining h(t_next) (pre-jump state)
        and ΔI = I(t_next) (segment integrals).
      * Accumulate integral term by summing ΔI over the K channels.
    - At each event time t_i:
      * Evaluate λ(h(t_i^-), t_i) to compute the hit term -log λ_{m_i}(t_i) with eps clamp.
      * If jump_update is provided, apply it to produce h(t_i^+):
            Δh = jump_update(x_i, h_minus)
        where x_i is passed as None (no per-event payload supported here).
    - Supports per-batch processing by looping over sequences.

    Parameters
    ----------
    f_h:
        Continuous hidden-state field dh/dt = f_h(h, t). Signature (h, t) -> Tensor[..., D].
    lambda_fn:
        Intensity function λ(h, t) returning K nonnegative channels. Signature (h, t) -> Tensor[..., K].
    h0:
        Initial hidden state of shape [D] or [B, D]. If [D], it is broadcast to the
        number of event sequences in `events`.
    events:
        Event container in one of two forms:
        (a) Single sequence:
            (times: 1D Tensor[Ni], marks: 1D LongTensor[Ni])
        (b) List/sequence of per-batch tuples:
            [(times_b: 1D Tensor[Ni_b], marks_b: 1D LongTensor[Ni_b]), ...] of length B
        Times must be monotonically nondecreasing per sequence. Events outside [0, T_end]
        are ignored for likelihood accumulation.
    T_end:
        Horizon T for the integral upper limit ∫_0^T and last segment end-time.
    jump_update:
        Optional callable producing an instantaneous state jump at events:
            jump_update(x_i, h_minus) -> Δh
        If provided, Δh is added to the pre-jump state h(t_i^-) to obtain h(t_i^+).
        x_i is passed as None (no payloads supported by this function).
    K:
        Number of marked intensity channels. If None, it's inferred from
        lambda_fn(h0, t=0) by reading the last dimension.
    config:
        Optional ODESolverConfig controlling integration backend and tolerances.
        Uses torchdiffeq with optional adjoint if available; otherwise falls back
        to a deterministic fixed-step solver via the wrappers.
    reduction:
        Reduction over batch: one of {"none", "mean", "sum"}. Default "mean".
    eps:
        Small positive value used to avoid log(0) in hit terms.
    device:
        Optional torch.device to move inputs and intermediate computations onto.
        Defaults to h0.device.

    Returns
    -------
    Tensor
        - If reduction="none": shape [B], per-sequence NLLs.
        - Else: scalar tensor.

    Notes
    -----
    - Strict training–inference decoupling: this function only computes likelihood terms.
      It does not sample events or perform readout.
    - All ops are differentiable with respect to parameters occurring in f_h, lambda_fn,
      and jump_update.

    Examples (not executed)
    -----------------------
    Minimal setup with two marked intensities:
        D, K = 4, 2

        def f_h(h, t):
            # simple contractive dynamics
            return -0.1 * h

        def lambda_fn(h, t):
            # example nonnegative intensities
            raw = h[..., :K]
            return raw.clamp_min(0.0) + 1e-6

        h0 = torch.zeros(D)
        times = torch.tensor([0.2, 0.8])
        marks = torch.tensor([0, 1], dtype=torch.long)

        # Optional solver config (uses adjoint if torchdiffeq is installed)
        cfg = ODESolverConfig(use_adjoint=True, method="dopri5", rtol=1e-5, atol=1e-7)

        loss = nll_continuous(
            f_h=f_h,
            lambda_fn=lambda_fn,
            h0=h0,
            events=(times, marks),
            T_end=1.0,
            K=K,
            config=cfg,
        )
    """
    # Normalize events and establish batch
    event_list = _normalize_events(events)
    B = len(event_list)

    # Device / dtype setup
    dev = device if device is not None else h0.device
    h0 = h0.to(dev)
    dtype = h0.dtype

    # Infer K if needed
    if K is None:
        # Use a representative h for inference (single item is sufficient)
        h_sample = h0[0] if h0.ndim == 2 else h0
        t0 = torch.zeros((), device=dev, dtype=dtype)
        lam0 = lambda_fn(h_sample, t0)
        if lam0.ndim == 0:
            raise ValueError("lambda_fn must return a tensor with last dimension K >= 1")
        K = int(lam0.shape[-1])
        if K <= 0:
            raise ValueError("Inferred K must be positive")

    # Prepare per-batch initial states
    h0B = _ensure_batch_h0(h0, B)  # [B, D]
    T_end_t = torch.as_tensor(float(T_end), device=dev, dtype=dtype)

    losses: List[Tensor] = []

    for b in range(B):
        times_b, marks_b = event_list[b]
        times_b = times_b.to(device=dev, dtype=dtype)
        marks_b = marks_b.to(device=dev, dtype=torch.long)

        # Keep only events within [0, T_end]; preserve original order
        if times_b.numel() > 0:
            mask_win = (times_b >= 0) & (times_b <= T_end_t)
            times_b = times_b[mask_win]
            marks_b = marks_b[mask_win]

        # Initialize state and accumulators
        h_curr = h0B[b]  # [D]
        t_prev = torch.zeros((), device=dev, dtype=dtype)

        # Accumulate integral over K channels and the hit terms
        integral_sum = torch.zeros((), device=dev, dtype=dtype)
        hit_sum = torch.zeros((), device=dev, dtype=dtype)

        # Iterate through events
        for i in range(times_b.numel()):
            t_next = times_b[i]

            # Integrate augmented system over [t_prev, t_next]
            t_eval = torch.stack([t_prev, t_next], dim=0)  # [2]
            h_traj, I_traj = odeint_augmented(
                f_h=f_h,
                lambda_fn=lambda_fn,
                h0=h_curr,
                K=int(K),
                t_eval=t_eval,
                config=config,
            )
            h_end = h_traj[-1]       # [D], pre-jump state h(t_i^-)
            I_end = I_traj[-1]       # [K], ΔI over this segment
            integral_sum = integral_sum + I_end.sum()

            # Hit term at the event
            lam_vec = lambda_fn(h_end, t_next)  # [K]
            if lam_vec.shape[-1] != int(K):
                raise RuntimeError(f"lambda_fn returned shape {tuple(lam_vec.shape)}, expected last dim {K}")
            mark_i = int(marks_b[i].item())
            if not (0 <= mark_i < int(K)):
                raise ValueError(f"Event mark {mark_i} out of range [0, {K-1}]")
            hit_sum = hit_sum - _safe_log(lam_vec[mark_i], eps)

            # Optional jump at event: h(t_i^+) = h(t_i^-) + Δh
            if jump_update is not None:
                try:
                    delta_h = jump_update(None, h_end)
                except TypeError:
                    # Allow simplified signature jump_update(h_minus) -> Δh
                    delta_h = jump_update(h_end)  # type: ignore[misc]
                h_curr = h_end + delta_h
            else:
                h_curr = h_end

            t_prev = t_next

        # Final segment up to T_end
        if bool((T_end_t - t_prev) >= 0):
            t_eval_final = torch.stack([t_prev, T_end_t], dim=0)
            h_traj_f, I_traj_f = odeint_augmented(
                f_h=f_h,
                lambda_fn=lambda_fn,
                h0=h_curr,
                K=int(K),
                t_eval=t_eval_final,
                config=config,
            )
            I_end_final = I_traj_f[-1]  # [K]
            integral_sum = integral_sum + I_end_final.sum()

        # Total NLL for sequence b
        losses.append(hit_sum + integral_sum)

    loss_vec = torch.stack(losses, dim=0)  # [B]

    if reduction == "none":
        return loss_vec
    if reduction == "mean":
        return loss_vec.mean()
    if reduction == "sum":
        return loss_vec.sum()
    raise ValueError("reduction must be one of {'none','mean','sum'}")


def nll_discrete(
    lam: Tensor,
    spikes: Tensor,
    dt: float,
    *,
    mask: Optional[Tensor] = None,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> Tensor:
    """Discrete-time Δt fallback NLL for marked TPPs.

    Computes:
        loss = -∑_{t,b,k} s[t,b,k] * log(lam[t,b,k] + eps) + ∑_{t,b,k} lam[t,b,k] * dt

    with optional masking of padded time steps.

    Parameters
    ----------
    lam:
        Intensity tensor of shape [T, B, K], elementwise nonnegative.
    spikes:
        Spike counts (or 0/1 indicators) tensor of shape [T, B, K].
    dt:
        Time step size (float).
    mask:
        Optional validity mask of shape [T, B]. Steps with mask==0 do not contribute
        to the loss. If None, all steps are considered valid.
    reduction:
        Reduction over batch: one of {"none", "mean", "sum"}. Default "mean".
    eps:
        Small positive clamp to avoid log(0) for the hit term.

    Returns
    -------
    Tensor
        - If reduction="none": shape [B], per-batch discrete NLLs.
        - Else: scalar tensor.

    Example (not executed)
    ----------------------
        T, B, K = 100, 8, 3
        lam = torch.rand(T, B, K).clamp_min(1e-6)
        spikes = torch.poisson(lam * 0.01)  # synthetic counts
        dt = 0.01
        mask = torch.ones(T, B)

        loss = nll_discrete(lam, spikes, dt, mask=mask, reduction="mean")
    """
    if lam.ndim != 3 or spikes.ndim != 3:
        raise ValueError(f"lam and spikes must be 3D [T,B,K]; got {lam.shape} and {spikes.shape}")
    if lam.shape != spikes.shape:
        raise ValueError(f"lam and spikes must have identical shape; got {lam.shape} vs {spikes.shape}")

    T, B, K = lam.shape
    dev = lam.device
    dtype = lam.dtype

    if mask is None:
        mask_ = torch.ones(T, B, 1, device=dev, dtype=dtype)
    else:
        if mask.shape != (T, B):
            raise ValueError(f"mask must have shape [T,B]; got {tuple(mask.shape)}")
        mask_ = mask.to(device=dev, dtype=dtype).unsqueeze(-1)  # [T,B,1]

    hit_term = -(spikes * _safe_log(lam, eps)) * mask_                 # [T,B,K]
    integral_term = (lam * float(dt)) * mask_                          # [T,B,K]
    total = hit_term + integral_term                                   # [T,B,K]

    per_batch = total.sum(dim=(0, 2))  # [B], sum over T and K

    if reduction == "none":
        return per_batch
    if reduction == "mean":
        return per_batch.mean()
    if reduction == "sum":
        return per_batch.sum()
    raise ValueError("reduction must be one of {'none','mean','sum'}")


def nll(
    intensity_fn: Any,
    events: Union[Tuple[Tensor, Tensor], Sequence[Tuple[Tensor, Tensor]]],
    horizon: float,
    **kwargs: Any,
) -> Tensor:
    """Public NLL wrapper.

    Dispatch behavior
    -----------------
    - If `intensity_fn` is a dict-like or object that provides attributes/keys
      {"f_h", "lambda_fn", "h0"} (and optionally "jump_update", "K", "config"),
      this function dispatches to nll_continuous using those components plus any
      explicit keyword overrides.
    - Otherwise, this function raises NotImplementedError with guidance to call
      nll_continuous (for ODE-based training) or nll_discrete (for pre-gridded
      intensities) directly.

    Parameters
    ----------
    intensity_fn:
        Dict-like or object with required entries:
            - f_h: callable (h, t) -> dh/dt
            - lambda_fn: callable (h, t) -> intensities [..., K]
            - h0: Tensor [D] or [B, D]
        Optional entries:
            - jump_update: callable(x_i, h_minus) -> Δh
            - K: int
            - config: ODESolverConfig
    events:
        Event container as in nll_continuous.
    horizon:
        End time T_end (float) as in nll_continuous.
    kwargs:
        Additional keyword arguments forwarded to nll_continuous, overriding any
        same-named values from `intensity_fn`.

    Returns
    -------
    Tensor
        The NLL per the reduction configuration of nll_continuous.

    Example (not executed)
    ----------------------
        container = {
            "f_h": f_h,
            "lambda_fn": lambda_fn,
            "h0": h0,
            "K": K,
            "config": ODESolverConfig(use_adjoint=True),
        }
        loss = nll(container, events=(times, marks), horizon=1.0, reduction="mean")
    """
    def _get(obj: Any, name: str) -> Any:
        if isinstance(obj, Mapping):
            return obj.get(name, None)
        return getattr(obj, name, None)

    f_h = _get(intensity_fn, "f_h")
    lam_fn = _get(intensity_fn, "lambda_fn")
    h0 = _get(intensity_fn, "h0")

    if f_h is not None and lam_fn is not None and h0 is not None:
        # Optional defaults from container, overridden by kwargs if provided
        base_kwargs: dict = {}
        jup = _get(intensity_fn, "jump_update")
        if jup is not None and "jump_update" not in kwargs:
            base_kwargs["jump_update"] = jup
        K_opt = _get(intensity_fn, "K")
        if K_opt is not None and "K" not in kwargs:
            base_kwargs["K"] = K_opt
        cfg = _get(intensity_fn, "config")
        if cfg is not None and "config" not in kwargs:
            base_kwargs["config"] = cfg

        return nll_continuous(
            f_h=f_h,
            lambda_fn=lam_fn,
            h0=h0,
            events=events,
            T_end=horizon,
            **base_kwargs,
            **kwargs,
        )

    raise NotImplementedError(
        "Unsupported intensity_fn container. "
        "Provide a dict/object with keys/attrs {'f_h','lambda_fn','h0'} to use the ODE-based "
        "path via nll_continuous(...), or call nll_discrete(...) when working with pre-gridded intensities."
    )


__all__ = [
    "nll_continuous",
    "nll_discrete",
    "nll",
]
