# 01 Prerequisites

This document enumerates the mathematical and computing foundations assumed by the SNN Academy track and maps each topic to where it appears in the C‑IDEN and SR readout pipeline. It is intended for researchers, graduate students, and practitioners preparing to reproduce results and extend the sr_ciden codebase.

Scope
- Mathematical: calculus and ODEs; linear algebra; probability for temporal point processes (TPPs) with basic measure-theoretic intuition; optimization for training continuous-time models.
- Computing: Python and PyTorch fundamentals; Jupyter workflows; reproducibility, seeding, determinism; environment and data versioning.
- Implementation mapping: where these concepts land in code during training and inference.

See also: sr_ciden core modules
- Dynamics and stable generator parameterization via [`Python.StableMatrixParam()`](src/sr_ciden/dynamics.py:126), [`Python.LinearNonlinearODE()`](src/sr_ciden/dynamics.py:191).
- ODE integration and adjoint-capable wrapper [`Python.odeint_wrapper()`](src/sr_ciden/solvers.py:298), augmented integrals [`Python.odeint_augmented()`](src/sr_ciden/solvers.py:455).
- Exact NLL for marked TPPs [`Python.nll_continuous()`](src/sr_ciden/losses.py:152).
- SR readout samplers [`Python.sample_ogata()`](src/sr_ciden/readout.py:152), [`Python.sample_bernoulli()`](src/sr_ciden/readout.py:183).
- Time-rescaling validation [`Python.time_rescaling_test()`](src/sr_ciden/validation.py:169).

## A. Mathematical prerequisites

A.1 Calculus and Ordinary Differential Equations (ODEs)
- Initial value problems: dh/dt = f(h, t) with h(0) = h0; existence, uniqueness under local Lipschitz continuity (Picard–Lindelöf).
- Stability notions: Lyapunov and contractivity; intuition for exponential stability and the effect of negative-definite generators.
- Numerical integration: fixed-step Euler and RK4; error vs stability trade-offs; adaptive solvers conceptually. These are invoked through [`Python.odeint_wrapper()`](src/sr_ciden/solvers.py:298) with pure-PyTorch fallbacks [`Python.rk4_solve()`](src/sr_ciden/solvers.py:221) and [`Python.euler_solve()`](src/sr_ciden/solvers.py:144).
- Neural ODEs: parameterizing f(h, t; θ) as a neural field; memory–accuracy trade-offs via the adjoint method (Chen et al., 2018).

A.2 Linear Algebra
- Norms, matrix exponentials, spectral radius; positive semidefiniteness.
- Stable generator parameterization: A = −(α I + L Lᵀ) with α > 0 ensures a negative-definite/semidefinite drift that promotes contractive flows of h(t). See [`Python.StableMatrixParam()`](src/sr_ciden/dynamics.py:126).
- Affine blocks and readouts: W_h h + b, per-mark scalar heads; understanding tensor shapes and broadcasting in implementations (see [`Python.LinearNonlinearODE()`](src/sr_ciden/dynamics.py:191), [`Python.IntensityHead()`](src/sr_ciden/dynamics.py:346)).

A.3 Probability for Temporal Point Processes (TPPs)
- Conditional intensity λ*(t): λ*(t) dt ≈ P[event in [t, t+dt) | history]. For marked processes with K channels, λ(t) ∈ ℝ^K with total intensity ∑_k λ_k(t).
- Examples: inhomogeneous Poisson; mutually exciting Hawkes processes (Hawkes, 1971).
- Log-likelihood for marked TPPs on [0, T]: −∑_i log λ_{m_i}(t_i) + ∫_0^T ∑_k λ_k(u) du. This objective is implemented via augmented ODE integration in [`Python.nll_continuous()`](src/sr_ciden/losses.py:152) using [`Python.odeint_augmented()`](src/sr_ciden/solvers.py:455).
- Ogata thinning for simulation (Lewis and Shedler, 1979; Ogata, 1981) and discrete-time Bernoulli approximations: see [`Python.sample_ogata()`](src/sr_ciden/readout.py:152) and [`Python.sample_bernoulli()`](src/sr_ciden/readout.py:183).

A.4 Measure-theoretic intuition (lightweight)
- Filtrations and predictability: λ*(t) is adapted to history up to t.
- Radon–Nikodym viewpoint: intensity as a density of the compensator with respect to Lebesgue measure for Poisson-like processes.
- Time-Rescaling Theorem: when the model is correct, τ_i = ∫_0^{t_i} ∑_k λ_k(u) du are arrival times of a unit-rate Poisson process; inter-arrival Δτ_i are i.i.d. Exp(1). Validation uses K–S tests on Δτ (Brown et al., 2002) via [`Python.time_rescaling_test()`](src/sr_ciden/validation.py:169).

A.5 Optimization
- First-order methods: SGD, momentum, Adam; line-search intuition.
- Differentiating through integrators: continuous adjoint for memory efficiency (Chen et al., 2018). sr_ciden delegates to torchdiffeq when present through [`Python.odeint_wrapper()`](src/sr_ciden/solvers.py:298).
- Practical aspects: gradient clipping for stability; regularization and spectral penalties (if using unconstrained A); learning-rate schedules; mixed precision caveats when integrating ODEs.

## B. Computing prerequisites

B.1 Python and PyTorch fundamentals
- Tensors: dtype and device control; broadcasting; in-place vs out-of-place semantics.
- Autograd: computation graphs; torch.no_grad() for inference-only calls (critical in SR readout, see [`src/sr_ciden/readout.py`](src/sr_ciden/readout.py)).
- Modules: nn.Module parameter registration; buffers vs parameters; saving/loading state dicts.
- Shape discipline in C‑IDEN: h(t) with leading batch dimension; K mark channels returned by the intensity head(s); see [`Python.CIDENContinuous()`](src/sr_ciden/adapters/torch.py:36) and [`Python.CIDENContinuous.get_intensity_fn()`](src/sr_ciden/adapters/torch.py:191).

B.2 Jupyter workflows
- Notebook hygiene: deterministic cell order, re-runnable top-to-bottom; data path parametrization; version checkpoints.
- Visualization of trajectories, intensities, residuals for time-rescaling diagnostics.

B.3 Reproducibility and determinism
- Seeding hierarchy: Python random, NumPy, torch (CPU/GPU), torch.backends flags.
- Deterministic inference-only samplers use local generators seeded per call (see [`Python.sample_ogata_with_stats()`](src/sr_ciden/readout.py:242)); training determinism depends on backend and solver choices.
- File and environment recording: capture pyproject.lock or a requirements export; persist git commit and experiment config.

B.4 Environment and data management
- Environments: conda/mamba or virtualenv; pinned versions for torch, torchdiffeq, scipy, numpy, matplotlib; optional frameworks will be added as extras later.
- Data versioning: store raw vs processed; hash checks; dataset cards with license/attribution.
- CI considerations: smoke tests and unit tests around likelihood, thinning efficiency, and time rescaling exist under `tests/` and help catch regressions.

Readiness self-check
- You can derive and explain the TPP log-likelihood and the Time-Rescaling Theorem at a high level (Brown et al., 2002).
- You can implement RK4 and Euler steps and understand stability implications.
- You understand the stable generator A = −(αI + L Lᵀ) rationale and can relate it to contractive flows.
- You can write PyTorch modules with clear tensor shapes and know how to place computations on CPU/GPU.
- You can run a deterministic inference pass with a local RNG and interpret acceptance ratios from thinning.
- You can version-lock an environment and re-run a notebook top-to-bottom without hidden state.

Suggested preparatory reading (inline references)
- Hawkes processes and self-excitation: Hawkes (1971).
- Thinning methods: Lewis and Shedler (1979); Ogata (1981).
- Time-Rescaling and K–S: Brown et al. (2002).
- Neural ODEs and adjoint: Chen et al. (2018).
- STDP and SNN learning: Bi and Poo (1998); Song, Miller and Abbott (2000).
- Surrogate gradients: Neftci, Mostafa and Zenke (2019).
- Simulator/framework references: Brian2 (Stimberg et al., 2019), NEST (Gewaltig and Diesmann, 2007), Nengo (Bekolay et al., 2014), SpykeTorch (Mozafari et al., 2019), snnTorch (Eshraghian et al., 2021).

Where this is used in the curriculum
- Weeks 1–2 of the Overview rely on the above material before diving into C‑IDEN theory and SR readout.
- The forthcoming notebooks will embed short primers and link back to this page.

For full citations, see [docs/academy/REFERENCES.md](docs/academy/REFERENCES.md).