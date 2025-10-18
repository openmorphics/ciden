# sr-ciden — Decoupled Continuous-Time Neuromorphic Library (C-IDEN with SR readout)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-brightgreen)](https://openmorphics.github.io/ciden/) [![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

sr-ciden is a research-oriented library for continuous-time neuromorphic modeling. It cleanly decouples training (exact likelihood over continuous time) from inference/readout (e.g., Sparse Regression via thinning), so you can swap readouts and diagnostic tools without entangling training dynamics.

- Train with exact continuous-time likelihoods
- Inference via forward-only intensity functions and readout utilities
- PyTorch-first, with optional torchdiffeq integration

## Installation

- From PyPI:
  ```bash
  pip install sr-ciden
  ```
- Editable/development install (with dev tools):
  ```bash
  pip install -e .[dev]
  ```
- Optional torchdiffeq support:
  ```bash
  pip install sr-ciden[torchdiffeq]
  ```

## Core Concepts

sr-ciden separates model training from inference/readout:
- Training computes a continuous-time negative log-likelihood (or related objectives) by integrating intensities between events and summing log-intensities at events.
- Inference uses a forward-only intensity_fn that is side-effect free and never participates in backpropagation. Readout utilities (e.g., Ogata thinning, discrete Bernoulli) operate purely on intensity_fn.

This design:
- Simplifies experimentation with new readout heads
- Keeps gradients confined to training-time computations
- Makes validation and diagnostics (e.g., time-rescaling tests) straightforward

## Quickstart

The following end-to-end script:
- Generates synthetic Poisson events
- Instantiates CIDENContinuous
- Runs a tiny training loop (if the model exposes trainable params)
- Obtains the inference-only intensity_fn
- Runs a time-rescaling goodness-of-fit test and prints the K–S p-value

```python
import torch
import sr_ciden
from sr_ciden.adapters import CIDENContinuous
from sr_ciden.solvers import ODESolverConfig
from sr_ciden.validation import time_rescaling_test


def generate_poisson_events(rate: float = 2.0, T: float = 1.0, seed: int = 0):
    """Homogeneous Poisson process via exponential inter-arrivals."""
    g = torch.Generator().manual_seed(seed)
    times = []
    t = 0.0
    while True:
        u = torch.rand((), generator=g).item()
        # Exponential inter-arrival with parameter 'rate'
        w = -float(torch.log(torch.tensor(u))) / rate
        t += w
        if t >= T:
            break
        times.append(t)
    times_t = (
        torch.tensor(times, dtype=torch.get_default_dtype()) if times
        else torch.empty(0, dtype=torch.get_default_dtype())
    )
    marks_t = torch.zeros_like(times_t, dtype=torch.long)
    return times_t, marks_t


def main():
    torch.manual_seed(0)

    # 1) Synthetic data
    events = generate_poisson_events(rate=2.0, T=1.0, seed=0)

    # 2) Model
    model = CIDENContinuous(
        hidden_dim=8,
        num_marks=1,
        solver_config=ODESolverConfig(dt=1e-3, use_adjoint=False),
    )
    print(f"sr_ciden version: {sr_ciden.__version__}")

    # 3) Basic training loop (decoupled from readout)
    maybe_params = [p for p in getattr(model, "parameters", lambda: [])() if p.requires_grad]
    opt = torch.optim.AdamW(maybe_params, lr=1e-2) if len(maybe_params) > 0 else None

    for step in range(3):
        model.train()
        model.reset_state(batch_size=1)
        loss = model(events, T_end=1.0)
        print(f"step={step} loss={float(loss):.4f}")
        if opt is not None and loss.requires_grad:
            opt.zero_grad()
            loss.backward()
            opt.step()

    # 4) Inference-only intensity_fn (no gradients)
    model.eval()
    intensity_fn = model.get_intensity_fn()

    # 5) Goodness-of-fit via Time-Rescaling Theorem
    result = time_rescaling_test(intensity_fn, events)
    print(f"K-S p-value: {result['p_value']:.4f}")


if __name__ == "__main__":
    main()
```

## Running Examples and Benchmarks

- Minimal training + inference example:
  ```bash
  python examples/minimal_training_loop.py
  ```
  See the source at [examples/minimal_training_loop.py](examples/minimal_training_loop.py).

- Hawkes process notebook (for a richer, end-to-end workflow):
  - Open [examples/notebook_hawkes_fit.ipynb](examples/notebook_hawkes_fit.ipynb) in Jupyter.

- Benchmarks:
  ```bash
  python benchmarks/benchmark_memory_and_speed.py
  ```
  See [benchmarks/benchmark_memory_and_speed.py](benchmarks/benchmark_memory_and_speed.py).

## Reproducibility and one-shot artifacts

- Reproduce all paper artifacts (examples + benchmarks + manifest):
  ```bash
  bash scripts/reproduce_paper.sh
  ```
  This writes metrics to [results/README.md](results/README.md:1) locations and figures/CSVs to [artifacts/figures/](artifacts/figures/:1). The script also captures the runtime environment to [artifacts/env.json](artifacts/env.json:1) and records the commit SHA.

- Dataset cache root:
  - Set SR_CIDEN_DATA to control where datasets are cached. Defaults to ~/.cache/sr_ciden.
    ```bash
    export SR_CIDEN_DATA=/path/to/datasets
    ```
  - See fetcher skeletons in [src/sr_ciden/data/fetchers.py](src/sr_ciden/data/fetchers.py:1). Replace placeholder URLs and checksums before camera-ready.

- Environments:
  - Conda: [environment.yml](environment.yml:1)
  - Pip (locked): [requirements-lock.txt](requirements-lock.txt:1) (regenerate via pip-tools)

## API

For the intended API surface and callable contracts, see [docs/API.md](docs/API.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE).
