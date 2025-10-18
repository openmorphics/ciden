"""
Inversion-based readouts (placeholder).

For simple parametric λ(t) where the integrated intensity Λ(t) has a closed-form inverse,
one can sample event times by drawing u ~ Uniform(0,1) and solving Λ(t) = -log(1-u).

This module provides a placeholder API to be filled with concrete closed-form cases.
"""
from __future__ import annotations

from typing import Callable, List


def sample_inversion(
    inv_cum_intensity: Callable[[float], float],
    num_events: int,
    seed: int = 0,
) -> List[float]:
    """
    Placeholder inversion sampler.

    Args:
      inv_cum_intensity: inverse Λ^{-1}(x) mapping from cumulative intensity to time
      num_events: number of events to draw
      seed: RNG seed

    Returns:
      List of event times (floats), strictly increasing.

    Raises:
      NotImplementedError: until concrete families are implemented.
    """
    raise NotImplementedError(
        "Inversion-based samplers are not yet implemented. "
        "Provide concrete inv_cum_intensity cases (e.g., piecewise-constant λ) in camera-ready."
    )
