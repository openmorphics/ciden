"""Phase 2 JAX/diffrax adapter placeholder for sr_ciden.

This module defines the public API surface expected for a future JAX-based backend
built on:
- JAX for array transformations and JIT/autodiff
- Flax (flax.linen.Module) for parameterized modules and training loops
- Diffrax for ODE/SDE integration

Important:
- This is a stub only. All classes and functions raise NotImplementedError.
- Do NOT import jax, flax, or diffrax here; they are not Phase 1 dependencies.
- The intent is to allow downstream code to import symbols and feature-flag behavior
  without introducing runtime dependencies prior to Phase 2.

Exports (stubs):
- CIDENJax
- SRReadoutJax
- nll_jax
- sample_ogata_jax
- sample_bernoulli_jax
"""

__all__ = [
    "CIDENJax",
    "SRReadoutJax",
    "nll_jax",
    "sample_ogata_jax",
    "sample_bernoulli_jax",
]


class CIDENJax:
    """Stub of a Flax Module for C-IDEN training.

    When implemented in Phase 2, this class will subclass flax.linen.Module and
    encapsulate the continuous-time dynamics, optional jump updates, and intensity
    readout for marked point processes. It will be used in standard JAX/Flax
    training loops.

    Methods (to be implemented in Phase 2)
    - setup(self): create submodules and initialize configuration
    - __call__(self, ...): compute training losses or outputs consistent with Flax API

    Notes
    - This stub intentionally raises NotImplementedError in every method.
    - The final API will follow Flax conventions and JAX's functional patterns.
    """

    def setup(self) -> None:
        """Create child modules and initialize configuration (future implementation)."""
        raise NotImplementedError("Phase 2: JAX/diffrax backend is not yet implemented.")

    def __call__(self, *args, **kwargs):
        """Forward call for training/inference (future implementation)."""
        raise NotImplementedError("Phase 2: JAX/diffrax backend is not yet implemented.")


class SRReadoutJax:
    """Stub for JAX-based discrete-time inference readout.

    This class will provide utilities for sampling spikes from instantaneous
    intensities on a Δt grid (Bernoulli approximation) and may complement continuous-
    time samplers (e.g., Ogata thinning) in a JAX/Flax workflow.

    Notes
    - This is an inference utility; it will not participate in gradient computation.
    - The actual implementation will live in Phase 2 with JAX-compatible random APIs.
    """

    def __call__(self, *args, **kwargs):
        """Sample or transform intensities in discrete time (future implementation)."""
        raise NotImplementedError("Phase 2: JAX/diffrax backend is not yet implemented.")


def nll_jax(*args, **kwargs):
    """Stub for a JAX functional negative log-likelihood.

    Intended design (to be finalized in Phase 2)
    - Functional signature following a (params, state) style.
    - Mirrors the PyTorch nll_continuous but adapted to JAX/Flax conventions.
    - Pure function that wires dynamics, readout, jump updates, and solver config.

    Returns
    -------
    loss : scalar-like
        Continuous-time negative log-likelihood (future implementation).
    """
    raise NotImplementedError("Phase 2: JAX/diffrax backend is not yet implemented.")


def sample_ogata_jax(*args, **kwargs):
    """Stub for Ogata-thinning event sampling in JAX.

    Intended design (to be finalized in Phase 2)
    - Functional API accepting (params, state) and an intensity function λ(t).
    - Uses JAX PRNG keys for reproducible sampling.

    Returns
    -------
    samples : structure
        Sampled event times/marks per process (future implementation).
    """
    raise NotImplementedError("Phase 2: JAX/diffrax backend is not yet implemented.")


def sample_bernoulli_jax(*args, **kwargs):
    """Stub for discrete-time Bernoulli spike sampling in JAX.

    Intended design (to be finalized in Phase 2)
    - p = 1 - exp(-λ * dt) computed in a JAX-compatible manner.
    - Accepts PRNG keys for randomness control.

    Returns
    -------
    spikes : array-like
        Binary spike indicators on a regular grid (future implementation).
    """
    raise NotImplementedError("Phase 2: JAX/diffrax backend is not yet implemented.")
