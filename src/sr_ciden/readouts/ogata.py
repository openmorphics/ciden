"""
Ogata thinning readout wrappers.

These functions delegate to the canonical implementations in [sr_ciden.readout](src/sr_ciden/readout.py:1)
to maintain a single source of truth, while exposing a modular readouts API.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import torch

# Delegate to original implementations for now
from sr_ciden.readout import (
    sample_ogata as _sample_ogata_impl,
    sample_ogata_with_stats as _sample_ogata_with_stats_impl,
)


def sample_ogata(
    intensity_fn: Callable[[float], torch.Tensor],
    T: float,
    lam_max: float,
    seed: int = 0,
) -> List[Tuple[float, int]]:
    """
    Continuous-time sampling via Ogata's thinning.

    Parameters:
      intensity_fn: t -> Tensor[K] nonnegative intensities
      T: horizon (exclusive)
      lam_max: global bound on total rate
      seed: local RNG seed

    Returns:
      List[(t, k)] ordered spikes
    """
    return _sample_ogata_impl(intensity_fn, T=T, lam_max=lam_max, seed=seed)


def sample_ogata_with_stats(
    intensity_fn: Callable[[float], torch.Tensor],
    T: float,
    lam_max: float,
    seed: int = 0,
) -> Tuple[List[Tuple[float, int]], Dict[str, float]]:
    """
    Ogata thinning with efficiency stats (acceptance ratio, bound violations).

    Returns:
      (spikes, stats) where stats keys include:
        - num_proposals
        - num_accepts
        - acceptance_ratio in [0,1]
        - bound_violations
    """
    return _sample_ogata_with_stats_impl(intensity_fn, T=T, lam_max=lam_max, seed=seed)
