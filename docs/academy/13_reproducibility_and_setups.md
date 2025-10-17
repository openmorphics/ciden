# 13 Reproducibility and Setups

Purpose
- Establish rigorous, audit-ready practices for deterministic training/evaluation of C‑IDEN and SR readout.
- Provide environment guidance (conda/pip), seeding protocols, data versioning, and notebook execution tips.
- Clarify optional framework extras to be introduced later without modifying the core project configuration.

Core sr_ciden anchors
- Exact NLL and augmented ODEs: [`Python.nll_continuous()`](src/sr_ciden/losses.py:152), [`Python.odeint_augmented()`](src/sr_ciden/solvers.py:455), [`Python.odeint_wrapper()`](src/sr_ciden/solvers.py:298), [`Python.ODESolverConfig()`](src/sr_ciden/solvers.py:86).
- Dynamics and stable generator: [`Python.LinearNonlinearODE()`](src/sr_ciden/dynamics.py:191), [`Python.StableMatrixParam()`](src/sr_ciden/dynamics.py:126), piecewise integrator [`Python.integrate()`](src/sr_ciden/dynamics.py:395).
- SR readout and validation: [`Python.sample_ogata()`](src/sr_ciden/readout.py:152), [`Python.sample_ogata_with_stats()`](src/sr_ciden/readout.py:242), [`Python.sample_bernoulli()`](src/sr_ciden/readout.py:183), [`Python.time_rescaling_test()`](src/sr_ciden/validation.py:169).

--------------------------------------------------------------------------------

## A. Environment management

A.1 Recommended tools
- Use conda/mamba for environment isolation and pinned solver stacks (CUDA, cuDNN).
- For lighter-weight setups, use `venv`/`pip` with pinned versions (requirements lock file).

A.2 Baseline conda environment (illustrative)
```
name: sr-ciden
channels: [conda-forge, pytorch, nvidia]
dependencies:
  - python=3.10
  - pytorch=*=*cuda*        # or cpu builds for determinism in CI
  - torchvision
  - cudatoolkit              # if GPU desired; else omit
  - numpy
  - scipy
  - matplotlib
  - pip
  - pip:
    - torchdiffeq==0.2.3    # optional; see determinism notes
    - jupyterlab
    - black
    - isort
```
Notes
- torchdiffeq is optional; sr_ciden has deterministic fixed-step fallbacks via [`Python.rk4_solve()`](src/sr_ciden/solvers.py:221) and [`Python.euler_solve()`](src/sr_ciden/solvers.py:144).
- For CI and unit tests, prefer CPU builds to avoid nondeterministic GPU kernels.

A.3 Optional extras (installed later; do not modify core pyproject here)
- snnTorch (PyTorch-based SNNs) for interop notebooks.
- Brian2 and Brian2GeNN for event-driven and GPU-accelerated simulations.
- NEST and Nengo (and Nengo Loihi) for large-scale simulation and neuromorphic deployment prototyping.
- SpykeTorch for rank-order/time-to-first-spike pipelines.
Install with separate environment files (to be added later) to keep core minimal.

--------------------------------------------------------------------------------

## B. Seeding and determinism

B.1 Global seed setup (Python, NumPy, PyTorch)
```
import os, random, numpy as np, torch

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Deterministic flags (PyTorch)
torch.use_deterministic_algorithms(True)
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = False
```
Notes
- Determinism may reduce throughput; disable only for exploratory runs (record that choice).

B.2 ODE solver determinism
- Adaptive solvers via torchdiffeq can be sensitive to tolerances (rtol, atol) and may take different internal branches across hardware/versions.
- For strict reproducibility (CI, papers), prefer fixed-step wrappers with tuned `dt`:
  - Use [`Python.odeint_wrapper()`](src/sr_ciden/solvers.py:298) with fallback "rk4" and set `dt` via [`Python.ODESolverConfig()`](src/sr_ciden/solvers.py:86).
  - For exact NLL segments, [`Python.odeint_augmented()`](src/sr_ciden/solvers.py:455) inherits this determinism.

B.3 SR readout determinism
- `sample_ogata`/`sample_ogata_with_stats` use a local CPU `torch.Generator` seeded per call. Outputs are independent of the global RNG state:
  - See [`Python.sample_ogata()`](src/sr_ciden/readout.py:152) and [`Python.sample_ogata_with_stats()`](src/sr_ciden/readout.py:242).
- Discrete-time fallback `sample_bernoulli` uses local seeded RNG and clamps probabilities to avoid numeric edge cases (see [`Python.sample_bernoulli()`](src/sr_ciden/readout.py:183)).

--------------------------------------------------------------------------------

## C. Data versioning and artifacts

C.1 Datasets
- Use git-lfs or DVC to version large datasets. Record:
  - Dataset name, version/hash, license, and preprocessing script commit.
  - Train/val/test split seeds and indices (persisted to JSON).

C.2 Artifact logging
- Persist a JSON manifest per run with:
  - Environment (Python, PyTorch, CUDA/cuDNN, torchdiffeq presence).
  - Seeds and reproducibility flags.
  - [`Python.ODESolverConfig()`](src/sr_ciden/solvers.py:86) fields (backend/method/rtol/atol/dt/max_steps).
  - SR readout lam_max policy and statistics (acceptance ratio, bound violations).
  - Metrics (NLL per event/time, K–S p-value/statistic, task accuracy/latency).
Template provided in 12 Evaluation Metrics.

--------------------------------------------------------------------------------

## D. Notebook execution guidance

D.1 Hygiene
- Make notebooks top-to-bottom runnable; avoid hidden state.
- Parameterize all file paths (e.g., via environment variables or a config cell).
- At the top of each notebook:
  - Set seeds and deterministic flags (Section B.1).
  - Print and log versions of Python, PyTorch, CUDA, torchdiffeq.

D.2 Visualization and diagnostics
- Plot intensity λ_k(t), cumulative integral I_k(t), and spike rasters to spot miscalibration early.
- Use time-rescaling diagnostics via [`Python.time_rescaling_test()`](src/sr_ciden/validation.py:169) with PIT/QQ visualizations.

D.3 Reproducibility cells
- Save run manifests (JSON) and figures with unique, seed-tagged filenames.
- Optionally export a requirements lock file (`pip freeze`) and conda `env export --no-builds`.

--------------------------------------------------------------------------------

## E. CI and regression testing

E.1 Provided tests
- Unit tests cover:
  - Time-rescaling correctness: [tests/test_time_rescaling.py](tests/test_time_rescaling.py)
  - Thinning efficiency and stats: [tests/test_thinning_efficiency.py](tests/test_thinning_efficiency.py)
  - Gradient correctness for NLL: [tests/test_nll_gradients.py](tests/test_nll_gradients.py)
  - Shape discipline for core flows: [tests/test_shapes.py](tests/test_shapes.py)
- Enable these with CPU builds and fixed-step RK4 to ensure stable behavior across runners.

E.2 Suggested CI gates
- Lint/docs checks, unit tests, and a smoke benchmark for ODE speed/memory: [benchmarks/benchmark_memory_and_speed.py](benchmarks/benchmark_memory_and_speed.py).
- Cache torch and dataset artifacts between runs when feasible.

--------------------------------------------------------------------------------

## F. Optional framework setups (to be added later)

- Separate env files for each framework with pinned versions (examples):
  - snnTorch: `pip install snntorch`
  - Brian2 / Brian2GeNN: `pip install brian2 brian2genn`
  - NEST: follow platform-specific install guides; often `conda install -c conda-forge nest-simulator`
  - Nengo / Nengo Loihi: `pip install nengo nengo-loihi`
  - SpykeTorch: follow project instructions; may require specific PyTorch versions
- Interop notebooks (08–11) will declare their extras explicitly and remain optional.

--------------------------------------------------------------------------------

## G. Reproducibility checklist

- [ ] Fixed seeds (Python/NumPy/Torch) and deterministic flags set.
- [ ] Solver backend and tolerances (or `dt`) recorded via [`Python.ODESolverConfig()`](src/sr_ciden/solvers.py:86).
- [ ] SR readout lam_max policy chosen; acceptance ratio and bound violations measured via [`Python.sample_ogata_with_stats()`](src/sr_ciden/readout.py:242).
- [ ] Dataset version/hash and splits persisted.
- [ ] Notebooks re-run top-to-bottom; manifests and figures saved.
- [ ] CI green on unit tests: time-rescaling, thinning efficiency, NLL gradients, shapes.

--------------------------------------------------------------------------------

## H. Known sources of nondeterminism and mitigations

- Adaptive ODE solvers (torchdiffeq): prefer fixed-step RK4 in papers/CI; if adaptive is required, fix (rtol, atol), disable stochastic data order, and lock package versions.
- GPU kernels: some ops may be nondeterministic on specific hardware; prefer CPU for CI or use deterministic kernels and flags.
- Parallel dataloaders: set `num_workers=0` or fix worker seeds.

--------------------------------------------------------------------------------

## I. Minimal “manifest” example (JSON)

```
{
  "env": {
    "python": "3.10.13",
    "pytorch": "2.3.0+cpu",
    "cuda": null,
    "torchdiffeq": "0.2.3"
  },
  "seeds": {
    "python": 42,
    "numpy": 42,
    "torch": 42
  },
  "determinism": {
    "torch.use_deterministic_algorithms": true,
    "cudnn.deterministic": true,
    "cudnn.benchmark": false
  },
  "solver": {
    "backend": "rk4",
    "dt": 0.001,
    "rtol": null,
    "atol": null,
    "max_steps": null
  },
  "sr_readout": {
    "lam_max_policy": "global",
    "lam_max_value": 50.0,
    "acceptance_ratio": 0.37,
    "bound_violations": 0
  },
  "metrics": {
    "nll_per_event": 1.23,
    "nll_per_time": 12.3,
    "ks_p_value": 0.41,
    "accuracy": 0.89
  },
  "data": {
    "dataset": "SHD",
    "split_seed": 123,
    "hash": "abc123..."
  }
}
```

For full citations mentioned in this document, see [docs/academy/REFERENCES.md](docs/academy/REFERENCES.md).

Where this fits in the curriculum
- Week 8 emphasizes reproducibility and setups (this document) alongside Evaluation Metrics (12). Integration notebooks will inherit these practices.