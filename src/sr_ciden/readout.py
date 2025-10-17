"""Sparse Regression (SR) readout via thinning and related inference utilities.

This module implements inference-only spike readout routines for marked
inhomogeneous Poisson processes:
- sample_ogata: Continuous-time sampling via Ogata's thinning algorithm
- sample_bernoulli: Discrete-time fallback using per-bin Bernoulli sampling
- sample_ogata_with_stats: Same spikes as sample_ogata plus efficiency stats

Training–inference decoupling
-----------------------------
These routines are not part of the training graph. They must not participate
in autograd. To enforce this, all calls to the provided intensity_fn(t)
are wrapped in torch.no_grad() and their results are detached before use.

Intensity function contract
---------------------------
intensity_fn(t: float) -> torch.Tensor
- Returns a 1D tensor of shape [K] with nonnegative intensities λ_k(t) ≥ 0.
- May close over model state; this function itself must not backpropagate.
- The returned tensor may live on any device (CPU or GPU). Sampling RNG
  is always done via a local CPU torch.Generator; times and marks are plain
  Python scalars.

Determinism and RNG
-------------------
- A local CPU torch.Generator is constructed and seeded from the seed argument.
- No global RNG state is used.
- sample_ogata_with_stats produces exactly the same spike sequence as
  sample_ogata for the same seed and inputs.

Numerical stability
-------------------
- Ogata acceptance probabilities are clamped to [0, 1].
- If sum(λ_k(t)) exceeds lam_max by a tolerance (1e-6), acceptance is clamped
  to 1.0 and a "bound violation" is recorded in stats (see sample_ogata_with_stats).
- Discrete fallback uses p = 1 - exp(-λ_k dt) and clamps p to [0, 1 - 1e-12].

Usage example (non-executable):
-------------------------------
    import torch
    # Suppose `model` holds state on GPU and exposes a rate(t, state) -> [K] tensor:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = {...}  # arbitrary model state

    def intensity_fn(t: float) -> torch.Tensor:
        # Inference-only: do not construct autograd graphs
        with torch.no_grad():
            lam = model.rate(t, state)  # shape [K], any device
            return lam.clamp_min(0.0)

    # Continuous-time sampling with an envelope bound lam_max
    spikes = sample_ogata(intensity_fn, T=1.0, lam_max=50.0, seed=42)

    # Discrete-time sampling with dt = 1 ms
    spikes_dt = sample_bernoulli(intensity_fn, T=1.0, dt=0.001, seed=42)
"""

from typing import Callable, Dict, List, Tuple
import math

import torch

__all__ = ["sample_ogata", "sample_bernoulli", "sample_ogata_with_stats"]


def _as_1d_nonneg_cpu(x: torch.Tensor) -> torch.Tensor:
    """Detach, move to CPU, flatten to 1D, and clamp negative entries to zero."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("intensity_fn(t) must return a torch.Tensor")
    x = x.detach().to("cpu")
    if x.ndim == 0:
        x = x.reshape(1)
    elif x.ndim > 1:
        x = x.reshape(-1)
    return x.clamp_min(0.0)


def _ogata_impl(
    intensity_fn: Callable[[float], torch.Tensor],
    T: float,
    lam_max: float,
    gen: torch.Generator,
    collect_stats: bool = False,
) -> Tuple[List[Tuple[float, int]], Dict[str, int]]:
    """Internal single-pass Ogata sampler used by public APIs.

    Returns the spikes list and, if collect_stats is True, a dict of counters.
    """
    if lam_max <= 0:
        raise ValueError("lam_max must be positive.")
    if T <= 0:
        return [], {"num_proposals": 0, "num_accepts": 0, "bound_violations": 0}

    spikes: List[Tuple[float, int]] = []
    t: float = 0.0
    tol = 1e-6

    num_proposals = 0
    num_accepts = 0
    bound_violations = 0

    # Loop: propose event times and thin
    while True:
        # Exponential proposal Δ ~ Exp(lam_max)
        u = float(torch.rand((), generator=gen))
        # Avoid log(0)
        if u <= 0.0:
            u = 1e-12
        dt = -math.log(u) / lam_max
        t += dt
        if t >= T:
            break

        num_proposals += 1
        with torch.no_grad():
            lam_vec = intensity_fn(t)
        lam = _as_1d_nonneg_cpu(lam_vec)
        if lam.numel() == 0:
            # No dimensions -> cannot produce an event at this time
            continue

        s = float(lam.sum().item())
        # Compute acceptance probability
        p = 0.0
        if s > 0.0:
            ratio = s / lam_max
            if ratio > 1.0 + tol:
                # Hard violation of the bounding rate; accept with prob 1
                p = 1.0
                bound_violations += 1
            else:
                # Clamp to 1.0 for tiny overshoots, otherwise use s/lam_max
                p = ratio if ratio <= 1.0 else 1.0
        else:
            p = 0.0

        u_accept = float(torch.rand((), generator=gen))
        if u_accept < p:
            # Accept: sample mark m ~ Categorical(lam / s) using the same generator
            if s <= 0.0:
                # Should not happen when accepted, but guard anyway
                continue
            # torch.multinomial uses unnormalized weights; OK to pass lam directly
            m_idx = int(torch.multinomial(lam, num_samples=1, generator=gen).item())
            spikes.append((float(t), m_idx))
            num_accepts += 1

    stats = {"num_proposals": num_proposals, "num_accepts": num_accepts, "bound_violations": bound_violations}
    return spikes, stats


def sample_ogata(
    intensity_fn: Callable[[float], torch.Tensor],
    T: float,
    lam_max: float,
    seed: int = 0,
) -> List[Tuple[float, int]]:
    """Sample spikes from a marked inhomogeneous Poisson process via Ogata's thinning.

    Parameters
    ----------
    intensity_fn : Callable[[float], torch.Tensor]
        Function returning a 1D tensor [K] of nonnegative intensities at time t.
        Must be inference-only (no autograd/grad graph creation).
    T : float
        End time (exclusive). If T <= 0, returns an empty list.
    lam_max : float
        Global envelope bound such that sum(intensity_fn(t)) <= lam_max for all t in [0, T].
    seed : int
        RNG seed for a local CPU torch.Generator.

    Returns
    -------
    List[Tuple[float, int]]
        A time-ordered list of (t, k) spikes where t is a Python float and k is an int mark in [0, K).
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    spikes, _ = _ogata_impl(intensity_fn, T, lam_max, gen, collect_stats=False)
    return spikes


def sample_bernoulli(
    intensity_fn: Callable[[float], torch.Tensor],
    T: float,
    dt: float,
    seed: int = 0,
) -> List[Tuple[float, int]]:
    """Discrete-time Bernoulli fallback for SR readout.

    For t in [0, T) with step dt, samples independent Bernoulli outcomes per mark k
    with probability p_k(t) = 1 - exp(-λ_k(t) * dt). When an outcome is 1, emits
    a spike (t, k). Multiple marks can fire within the same bin.

    Parameters
    ----------
    intensity_fn : Callable[[float], torch.Tensor]
        Function returning a 1D tensor [K] of nonnegative intensities at time t.
    T : float
        End time (exclusive). Must be positive.
    dt : float
        Time step size. Must be positive.
    seed : int
        RNG seed for a local CPU torch.Generator.

    Returns
    -------
    List[Tuple[float, int]]
        A time-ordered list of (t, k) spikes.
    """
    if dt <= 0 or T <= 0:
        raise ValueError("dt and T must be positive.")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    spikes: List[Tuple[float, int]] = []
    t = 0.0
    one_minus_eps = 1.0 - 1e-12

    while t < T:
        with torch.no_grad():
            lam_vec = intensity_fn(t)
        lam = _as_1d_nonneg_cpu(lam_vec)
        if lam.numel() > 0:
            # p = 1 - exp(-λ dt)
            p = 1.0 - torch.exp(-lam * float(dt))
            p = torch.clamp(p, min=0.0, max=one_minus_eps)
            u = torch.rand(p.shape, generator=gen)
            mask = u < p
            if mask.any():
                idxs = torch.nonzero(mask, as_tuple=False).flatten().tolist()
                # Emit in ascending mark order for determinism
                for k in idxs:
                    spikes.append((float(t), int(k)))

        t += dt

    return spikes


def sample_ogata_with_stats(
    intensity_fn: Callable[[float], torch.Tensor],
    T: float,
    lam_max: float,
    seed: int = 0,
) -> Tuple[List[Tuple[float, int]], Dict[str, float]]:
    """Ogata thinning with efficiency reporting.

    Produces the exact same spike sequence as sample_ogata for the same inputs
    by performing a single sampling pass that also records counters.

    Returns
    -------
    (spikes, stats) where:
      - spikes: List[(float_time, int_mark)]
      - stats: Dict with
          {
            "num_proposals": int,
            "num_accepts": int,
            "acceptance_ratio": float in [0, 1],
            "bound_violations": int,
          }
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    spikes, raw_stats = _ogata_impl(intensity_fn, T, lam_max, gen, collect_stats=True)
    num_props = int(raw_stats.get("num_proposals", 0))
    num_acc = int(raw_stats.get("num_accepts", 0))
    bviol = int(raw_stats.get("bound_violations", 0))
    acc_ratio = float(num_acc / num_props) if num_props > 0 else 0.0
    stats: Dict[str, float] = {
        "num_proposals": num_props,
        "num_accepts": num_acc,
        "acceptance_ratio": acc_ratio,
        "bound_violations": bviol,
    }
    return spikes, stats
