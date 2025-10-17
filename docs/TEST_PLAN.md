# Test Plan - sr-ciden

Scope: enumerate unit tests (to be added later) for continuous-time training and SR readout inference, ensuring strict training–inference decoupling, determinism, and implementation readiness. No code in this document; test files are planned for a future PR.

Planned test files:
- tests/test_shapes.py
- tests/test_nll_gradients.py
- tests/test_time_rescaling.py
- tests/test_thinning_efficiency.py
- [tests/test_import_and_version.py](tests/test_import_and_version.py) (already present)

## Conventions

- Batch size B, hidden dimension H, marks K, horizon T, grid steps N.
- Default dtype float32.
- Devices: cpu as baseline; cuda for parity checks where feasible.
- Seeded RNG for any sampling; fixed tolerances for ODE solvers.
- Reference APIs (spec only):
  - [integrate(f, state, events)](src/sr_ciden/dynamics.py:1)
  - [nll(intensity_fn, events, horizon)](src/sr_ciden/losses.py:1)
  - [sample_ogata(intensity_fn, T, lam_max, seed)](src/sr_ciden/readout.py:1)
  - [sample_bernoulli(intensity_fn, T, dt, seed)](src/sr_ciden/readout.py:1)
  - [time_rescaling_test(intensity_fn, events)](src/sr_ciden/validation.py:1)

## tests/test_shapes.py

Goal: verify input/output shapes, broadcasting, ragged batch handling, and device/dtype propagation for core functions.

Assertions:
- integrate
  - Given B ragged sequences of variable-length events and initial state [B, H], returns:
    - trajectory.final_state with shape [B, H]
    - trajectory.integral with shape [B]
    - sample_times either None or a ragged structure aligned per sequence
  - Device and dtype match inputs; raises on device mismatch.
  - Rejects non-monotonic event times and t > T (ValueError).
- nll
  - With unmarked events, returns scalar loss (reduction="mean"/"sum") or [B] (reduction="none").
  - With marked events, validates mark/index alignment; raises on shape mismatch.
  - discrete_fallback path produces same shape guarantees.
- Readout
  - sample_ogata returns a list of (t, m) with 0 < t <= T, m in {0..K-1}; stats contains accepted, proposed, and acceptance_ratio in [0,1].
  - sample_bernoulli returns grid-aligned times and optional marks; no gradients.
- Validation
  - time_rescaling_test returns dict keys {transformed, ks_stat, p_value}; transformed is list-like per sequence.

Edge cases:
- Empty sequence (no events) with positive horizon.
- Single-event sequence at boundary t = T (allowed or rejected per spec; must be consistent).
- Very small horizon (T ~ 1e-6) and tiny dt for discrete fallback.
- Large B with mixed sequence lengths to check batching overhead.

## tests/test_nll_gradients.py

Goal: gradient sanity for exact NLL and stability toggles.

Setup:
- Construct a simple linear ODE with known stable parameterization A = -(alpha I + L L^T) and softplus intensity head lambda*(t) = softplus(W_o h + b_o).
- Events: small synthetic batches with known horizons.

Assertions:
- Finite-diff vs autograd
  - Compute NLL and compare autograd gradients w.r.t. a small subset of parameters against central finite differences within tolerance (e.g., 1e-3 absolute or relative).
- Non-negativity and finiteness
  - Integral terms are non-negative; logs are finite given eps clamps; loss is finite for valid inputs.
- Stability toggle
  - With stability enabled (default), ODE solutions remain bounded for long horizon sanity checks.
  - When disabled (for ablation), detect and warn on explosions but ensure well-defined error handling (e.g., NaN guard raising).
- Adjoint vs direct
  - When adjoint is available, gradients match direct backprop within tolerance on short sequences, acknowledging solver path differences.

## tests/test_time_rescaling.py

Goal: validate Time-Rescaling Theorem utilities.

Setup:
- Simulate ground truth events from a well-specified process (e.g., Hawkes or homogeneous Poisson) with known intensity function.
- Provide intensity_fn to [time_rescaling_test(intensity_fn, events)](src/sr_ciden/validation.py:1).

Assertions:
- Rescaled inter-arrivals are approximately Exp(1) (or Uniform(0,1) after CDF mapping).
- K–S test returns p-value above threshold (e.g., p > 0.05) for well-specified model.
- For misspecified intensity (stress test), p-value trends lower, detecting model mismatch.
- Per-mark breakdown (if K > 1) is returned and consistent with overall metrics.

Determinism:
- No RNG required; deterministic given inputs.

## tests/test_thinning_efficiency.py

Goal: ensure Ogata thinning efficiency trends with the tightness of the bound lam_max.

Setup:
- Choose a time-varying intensity where a computable upper bound lam_max is adjustable by a slack factor c ≥ 1 (e.g., lam_max = c * sup_t lam(t)).
- Run [sample_ogata(intensity_fn, T, lam_max, seed)](src/sr_ciden/readout.py:1) for a grid of c values (e.g., {1.0, 1.2, 2.0, 4.0}) at fixed T and seeds.

Assertions:
- acceptance_ratio decreases monotonically as c increases (allow small numerical tolerance).
- Proposed and accepted counts are logged; accepted ≤ proposed always.
- With c = 1.0 (tight bound), acceptance_ratio is highest and consistent across repeated runs with same seed.

Determinism:
- Identical seeds yield identical spike trains and stats across CPU and GPU (when available).

## tests/test_import_and_version.py (existing)

Goal: ensure package importability and version string presence.

Assertions:
- Import sr_ciden succeeds.
- __version__ string exists and is non-empty (format not enforced at this stage).

Reference:
- Already present at [tests/test_import_and_version.py](tests/test_import_and_version.py).

## Determinism and reproducibility policy

- All stochastic tests must pass a fixed integer seed through to the implementation and, where applicable, to framework-native generators (e.g., torch.Generator).
- Document seed, device, solver, rtol/atol or dt, lam_max, and acceptance_ratio in test logs for reproducibility.
- CPU-only baseline runs: every stochastic test must pass deterministically on cpu. GPU runs, when used, must match cpu outputs bitwise or within stated tolerances (solver path differences accepted where documented).

## Tolerances and thresholds

- Gradient checks: atol ≤ 1e-3 and rtol ≤ 1e-3 for finite-diff vs autograd on small problems.
- K–S p-value threshold: default 0.05 unless stricter criteria are justified.
- Monotonic trends (acceptance_ratio vs c): allow small non-monotonicity within ε = 0.01 due to stochasticity; otherwise assert monotonic non-increasing sequence.

## Benchmark outlines (later; not CI-blocking)

- Adjoint memory scaling
  - Measure peak memory and runtime vs horizon length and event density for adjoint vs direct backprop.
  - Report break-even regime where adjoint becomes beneficial.
- Thinning efficiency curves
  - acceptance_ratio and throughput vs lam_max slack factor c.
  - Compare naive global bound vs adaptive local bounds if added later.
- Throughput
  - Events/second for sampling (Ogata and Bernoulli), and training steps/second for NLL on representative workloads.

## Coverage criteria

- Shape and error-path coverage for all public APIs:
  - [integrate(f, state, events)](src/sr_ciden/dynamics.py:1)
  - [nll(intensity_fn, events, horizon)](src/sr_ciden/losses.py:1)
  - [sample_ogata(intensity_fn, T, lam_max, seed)](src/sr_ciden/readout.py:1)
  - [sample_bernoulli(intensity_fn, T, dt, seed)](src/sr_ciden/readout.py:1)
  - [time_rescaling_test(intensity_fn, events)](src/sr_ciden/validation.py:1)
- Determinism checks for all stochastic components.
- Stability checks for ODE parameterizations and jump updates.

## References

- Architecture and design contract: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- API details: [docs/API.md](docs/API.md)