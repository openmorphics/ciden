"""Backend adapter namespace for sr_ciden.

This package exposes framework-specific adapters that wrap the functional core
into conventional module constructs for ease of use in training and inference.

Currently available:
- PyTorch adapters: CIDENContinuous (training), SRReadoutTorch (discrete inference)
"""
from .torch import CIDENContinuous, SRReadoutTorch

__all__ = ["CIDENContinuous", "SRReadoutTorch"]
