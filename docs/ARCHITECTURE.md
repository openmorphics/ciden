# Architecture - sr-ciden

Purpose: define the design contract and module responsibilities for training and inference of continuous-time conditional intensity models with SR readout.

## Two-stage contract

- Training: learn a continuous-time conditional intensity using a Neural ODE core with differentiable jump updates. Optimize exact temporal point process likelihood with adjoint gradients when available.
- Inference: produce spike trains via SR readout using Ogata thinning in continuous-time, with a discrete-Δt Bernoulli fallback. The readout is never in the backprop graph.

### Training details
- Core dynamics define h(t) with stable parameterization A = -(alpha I + L L^T). Jump updates U(x_i, h^-) applied at events.
- Objective: maximize TPP likelihood; we minimize negative log-likelihood with an augmented ODE accumulating integral of lambda*(t).
- Gradients: prefer adjoint via torchdiffeq; fallback to direct backprop through solver wrapper.

### Inference details
- Ogata thinning for marked processes; reports acceptance ratio; deterministic seeding support.
- Discrete fallback uses Bernoulli with p(t) = 1 - exp(-lambda*dt).
- Readout never backpropagates; training–inference are strictly decoupled.

## Module decomposition

- [src/sr_ciden/dynamics.py](src/sr_ciden/dynamics.py): Neural ODE core; stable A = -(alpha I + L L^T); differentiable jump update U(x_i, h^-).
- [src/sr_ciden/solvers.py](src/sr_ciden/solvers.py): ODE wrappers; autodetect torchdiffeq; adjoint option; safe fixed-step RK4/Euler fallback.
- [src/sr_ciden/losses.py](src/sr_ciden/losses.py): Continuous-time NLL with augmented ODE; discrete fallback NLL.
- [src/sr_ciden/readout.py](src/sr_ciden/readout.py): Ogata thinning sampler for marked processes; Bernoulli fallback; acceptance ratio; deterministic seeding.
- [src/sr_ciden/validation.py](src/sr_ciden/validation.py): Time-Rescaling Theorem utilities and Kolmogorov–Smirnov test harness.
- [src/sr_ciden/adapters/torch.py](src/sr_ciden/adapters/torch.py): PyTorch nn.Module wrappers for Phase 1.
- [src/sr_ciden/adapters/jax.py](src/sr_ciden/adapters/jax.py): Phase 2 API stubs targeting diffrax.

## Data and object model

Library-agnostic conceptual structures. These are contracts, not code.

- EventStream: sequence of events as (t, m) pairs where t are strictly increasing times and m is an optional mark in {0..K-1}. Batches are lists of variable-length sequences with a shared horizon T per sequence.
- Trajectory: piecewise ODE segments between events; records solver step times for integral accumulation and any dense evaluations needed by losses.
- SpikeTrain: list of (t, m) emitted by SR readout.

Batching strategy:
- Per-sequence variable-length events; minimize padding. Batch container carries masks if packing is required by downstream frameworks.
- CPU/GPU parity; operations vectorize across batch where possible without enforcing fixed length.

## Design decisions and rationale

- Stability: reparameterize A with alpha > 0 and L; default on, with a toggle allowed for ablations.
- Intensity: lambda*(t) = g(W_o h(t) + b_o) with g in {softplus, exp}. Default softplus for stability and bounded gradients.
- Jump updates: U(x_i, h^-) differentiable; optional exogenous features x(t) or per-event x_i are supported.
- Adjoint gradients: prefer torchdiffeq; fallback wrapper provides direct backprop with controlled memory.
- Training–inference decoupling: strict separation; readout is not part of the training graph.
- Randomness: deterministic via explicit seeds and torch.Generator; CPU/GPU parity; document PRNG streams in logs.

## Pseudocode

### Augmented ODE for integral accumulation

Goal: integrate lambda*(t) over [t0, T] while evolving h(t) and applying jump updates at event times.

```
Inputs: initial state h0, event times {t_i}, optional marks {m_i}, horizon T
State: z = [h, s] where s accumulates integral, s(0) = 0
For each interval [t_i, t_{i+1}) with t_0 = 0 and t_{N+1} = T:
  define ODE:
    dh/dt = f_theta(t, h, context)
    sdot = lambda_star(h)   // lambda_star(t) = g(W_o h + b_o)
  integrate z from t_i to t_{i+1} with chosen solver
  at t_{i+1} if i < N:
    h <- U(x_i_plus_1, h_minus)   // differentiable jump update
return final h(T), accumulated s(T), optional dense samples
```

Notes:
- The solver may request dense output times for loss integration diagnostics.
- The adjoint formulation treats s as part of the augmented state for stable gradients.

### Ogata thinning for marked processes

```
Inputs: intensity_fn, horizon T, global upper bound lam_max, RNG seed
Initialize: t = 0, S = empty list, accepted = 0, proposed = 0
while t < T:
  draw u1 ~ Uniform(0,1) using seeded RNG
  w = -log(u1) / lam_max
  t_candidate = t + w
  if t_candidate >= T: break
  lam_t, mark_probs = intensity_fn(t_candidate)   // no backprop
  proposed += 1
  draw u2 ~ Uniform(0,1)
  if u2 * lam_max <= lam_t:
    sample mark m ~ Categorical(mark_probs) if marked, else m = 0
    append (t_candidate, m) to S
    accepted += 1
  t = t_candidate
acceptance_ratio = accepted / max(1, proposed)
return SpikeTrain S, stats {accepted, proposed, acceptance_ratio}
```

Guidance:
- Choose lam_max as a safe bound; too loose lowers acceptance_ratio and increases compute.
- Deterministic seeding yields reproducible S and stats on CPU and GPU.

### Discrete Δt Bernoulli fallback

```
Inputs: intensity_fn, horizon T, step dt, RNG seed
Initialize: t = 0, S = empty list
while t < T:
  lam = intensity_fn(t)    // no backprop
  p = 1 - exp(-lam * dt)   // small dt => Poisson approximation
  draw b ~ Bernoulli(p)
  if b == 1:
    append (t, optional mark) to S
  t = t + dt
return SpikeTrain S
```

## Reproducibility and logging
- All stochastic procedures accept seeds and use framework Generators when available.
- Report solver type, tolerances, dt, lam_max, acceptance_ratio, device, and seed in experiment logs.