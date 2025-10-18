"""
Discrete-time Bernoulli binning readout wrapper.

Delegates to [sr_ciden.readout.sample_bernoulli](src/sr_ciden/readout.py:183).
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import torch
from sr_ciden.readout import sample_bernoulli as _sample_bernoulli_impl


def sample_bernoulli(
    intensity_fn: Callable[[float], torch.Tensor],
    T: float,
    dt: float,
    seed: int = 0,
) -> List[Tuple[float, int]]:
    """
    Discrete fallback readout using p(t) = 1 - exp(-λ(t)·dt).

    Parameters:
      intensity_fn: t -> Tensor[K] nonnegative intensities
      T: horizon (exclusive)
      dt: grid step
      seed: local RNG seed
    """
    return _sample_bernoulli_impl(intensity_fn, T=T, dt=dt, seed=seed)
