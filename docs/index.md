# SR-CIDEN

Decoupled Continuous-Time Neuromorphic Library (C-IDEN with SR readout). SR-CIDEN cleanly separates training (exact continuous-time likelihoods) from inference/readout (forward-only intensity functions), enabling interchangeable readouts and robust diagnostics without entangling gradient paths.

- Training: continuous-time negative log-likelihood via intensity integration and event log-intensities.
- Inference: forward-only `intensity_fn(t[, m])` with no gradients.
- Readouts: Ogata thinning (continuous-time), Bernoulli binning (discrete), and planned inversion samplers for simple λ(t).
- Diagnostics: time-rescaling goodness-of-fit (K–S), thinning acceptance analysis, latency/throughput benchmarks.

## Quick links

- Concepts and architecture: see Architecture overview.
- API overview: see the API surface and call contracts.
- Minimal example: run `python examples/minimal_training_loop.py` from the repository root to exercise a short end-to-end flow (training entry point + inference-only `intensity_fn` + KS diagnostic).

## Installation

- From PyPI (planned):
  - `pip install sr-ciden`
- Editable development install with docs and dev tooling:
  - `pip install -e .[dev,docs]`

Python 3.10–3.12 are supported. The default CI baseline uses CPU PyTorch; CUDA builds are tested locally and documented in the reproducibility kit.

## Core ideas

- Decoupling:
  - Training computes exact continuous-time likelihoods with ODE-evolved hidden states and differentiable jump updates.
  - Inference exposes a side-effect-free `intensity_fn` for readouts; readouts never participate in backprop.
- Readouts:
  - Ogata’s thinning for continuous-time spike sampling.
  - Discrete fallback via Bernoulli on a grid with `p(t) = 1 - exp(-λ(t)·dt)`.
  - Planned inversion samplers for simple parametric λ(t).
- Diagnostics:
  - Time-rescaling theorem to test goodness-of-fit on synthetic and real spike trains.
  - Thinning acceptance vs. λ_max strategies to quantify efficiency.
  - Training stability and throughput benchmarks with optional profiling.

## Getting started

1) Install and verify:
   - `pip install -e .[dev,docs]`
   - `pytest -q`
   - `mkdocs build`

2) Minimal example:
   - `python examples/minimal_training_loop.py`
   - Prints training loss, constructs the inference-only `intensity_fn`, runs time-rescaling, and reports the K–S p-value.

3) Benchmarks:
   - `python benchmarks/benchmark_memory_and_speed.py --bench all`
   - Additional standardized benchmarks live under `benchmarks/` (see repository README for details).

## Documentation structure

- Concepts
  - Architecture: training vs. inference pathways, data contracts, and module decomposition.
- API
  - Public functions and adapters; autodoc sourced from `src/sr_ciden/**`.
- Academy
  - A structured learning path covering background, theory, integration guides, and applications.

## Reproducibility and datasets

- A reproducibility kit provides pinned environments, deterministic toggles, and experiment configs under `configs/`.
- Datasets (e.g., SHD/SSC/DVS Gesture/NCARS) are fetched with checksums and cached under a root path configured via the `SR_CIDEN_DATA` environment variable.

## Citation

If you use SR-CIDEN in your work, please cite:

- Title: SR-CIDEN: Decoupled Continuous-Time Neuromorphic Modeling with Sparse Readout
- Authors: The SR-CIDEN Authors
- Year: 2025

A machine-readable citation is provided in `CITATION.cff`. A DOI will be added upon Zenodo deposition.

## License

Apache License 2.0. See `LICENSE` in the repository.