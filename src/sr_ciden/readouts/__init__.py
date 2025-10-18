"""
Readouts subpackage: interchangeable readout implementations and common metrics API.

This namespace provides a stable surface for:
- Ogata thinning (continuous-time)
- Bernoulli binning (discrete-time)
- Inversion-based samplers (for simple λ(t); placeholder)

Backwards compatibility:
- These wrappers delegate to canonical implementations in [sr_ciden.readout](src/sr_ciden/readout.py:1).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import torch

# Re-export wrappers
from .ogata import sample_ogata, sample_ogata_with_stats
from .bernoulli import sample_bernoulli

__all__ = [
    "sample_ogata",
    "sample_ogata_with_stats",
    "sample_bernoulli",
]
