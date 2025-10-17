# 07 SR Readout and Thinning

Purpose
- Formalize SR (sparse regression) readout as a sampling stage decoupled from training.
- Detail Ogata’s thinning for continuous-time marked inhomogeneous Poisson processes and a discrete-time Bernoulli fallback.
- Provide acceptance-ratio analysis, practical lam_max selection, dynamic bounds, and framework-agnostic pseudocode.

Implementation anchors in sr_ciden
- Continuous-time thinning sampler: [`Python.sample_ogata()`](src/sr_ciden/readout.py:152).
- Same spikes plus efficiency stats (acceptance ratio, bound violations): [`Python.sample_ogata_with_stats()`](src/sr_ciden/readout.py:242).
- Discrete-time fallback sampler: [`Python.sample_bernoulli()`](src/sr_ciden/readout.py:183).
- Inference-only intensity function constructed from a trained model: [`Python.CIDENContinuous.get_intensity_fn()`](src/sr_ciden/adapters/torch.py:191).
- Goodness-of-fit via time rescaling and K–S testing: [`Python.time_rescaling_test()`](src/sr_ciden/validation.py:169).

Inline references
- Thinning: Lewis and Shedler (1979); Ogata (1981).
- Time-Rescaling: Brown et al. (2002).

--------------------------------------------------------------------------------

## 1. Problem setup

- Given a trained intensity model λ(h(t), t) ∈ ℝ^K (nonnegative per mark), SR readout samples spikes from the corresponding marked inhomogeneous Poisson process on [0, T].
- Two implementations:
  - Continuous-time (Ogata’s thinning): exact in distribution under a valid envelope lam_max ≥ sup_{t∈[0,T]} ∑_k λ_k(t).
  - Discrete-time Bernoulli fallback: approximate sampling on a grid with step dt (‘small’ such that per-bin event probabilities remain ≪ 1).

Strict training–inference decoupling
- All sr_ciden readout functions run under torch.no_grad(); they must not participate in autograd. See the explicit guards in [`src/sr_ciden/readout.py`](src/sr_ciden/readout.py).

--------------------------------------------------------------------------------

## 2. Ogata’s thinning for marked processes

Superposition trick
- Let S(t) = ∑_k λ_k(t). If S(t) ≤ lam_max on [0, T], a dominating homogeneous Poisson process with rate lam_max can propose event times. Each proposal at time t is accepted with probability p(t) = S(t) / lam_max (clamped to 1). If accepted, the mark is drawn from Categorical(λ(t) / S(t)).

Pseudocode (framework-agnostic; mirrors [`Python.sample_ogata()`](src/sr_ciden/readout.py:152))
```
# Inputs:
#   intensity_fn(t) -> Tensor[K] (nonnegative), inference-only
#   T > 0, lam_max > 0, RNG seed
# Output: time-ordered list of (t, k) events
spikes = []
t = 0.0
while True:
    # exponential inter-arrival from dominating Poisson
    u = rand_uniform()
    dt = -log(max(u, 1e-12)) / lam_max
    t = t + dt
    if t >= T:
        break

    lam_vec = clamp_nonneg(intensity_fn(t))  # shape [K]
    S = sum(lam_vec)
    if S <= 0:
        continue

    # Acceptance
    ratio = S / lam_max
    p = min(ratio, 1.0)
    if rand_uniform() < p:
        # Mark thinning (unnormalized weights permitted)
        k = categorical_sample(weights=lam_vec)
        spikes.append((t, k))
```

Properties
- Correctness: If lam_max ≥ sup S(t), accepted events form the target process (Ogata, 1981).
- Efficiency: The expected acceptance ratio ρ satisfies ρ = E[S(t)] / lam_max if proposals “see” typical S(t). Larger lam_max reduces acceptance and increases proposals.

--------------------------------------------------------------------------------

## 3. Discrete-time Bernoulli fallback

Sampling rule
- On bins [t, t+dt), emit independent outcomes per k with
  p_k(t) = 1 − exp(−λ_k(t) dt), producing 0/1 spikes possibly across multiple k. Suitable when a grid is mandated (e.g., framework constraints) or for fast approximations.

Pseudocode (mirrors [`Python.sample_bernoulli()`](src/sr_ciden/readout.py:183))
```
spikes = []
for t in {0, dt, 2dt, ... , T-dt}:
    lam = clamp_nonneg(intensity_fn(t))  # [K]
    p = 1 - exp(-lam * dt)               # elementwise
    u = rand_vector(K)
    for k where u[k] < p[k]:
        spikes.append((t, k))
```

Notes
- Accuracy improves as dt → 0 but cost increases linearly in T/dt. Use the continuous-time method when possible.

--------------------------------------------------------------------------------

## 4. Acceptance ratios, bounds, and heuristics

Acceptance ratio ρ
- For homogeneous total intensity S(t) ≡ s_0, ρ = s_0 / lam_max.
- For inhomogeneous S(t), if proposals are independent of S(t) (dominated by lam_max), ρ ≈ (1/T) ∫_0^T S(u) du / lam_max when S has modest variation. More generally, empirical ρ is reported by [`Python.sample_ogata_with_stats()`](src/sr_ciden/readout.py:242) via "acceptance_ratio".

Bound violations
- If S(t) exceeds lam_max in practice (due to model drift or underestimation), a robust implementation clamps acceptance p to 1.0 and records a violation counter (see stats["bound_violations"] in [`Python.sample_ogata_with_stats()`](src/sr_ciden/readout.py:242)). Frequent violations indicate lam_max is too small.

Setting lam_max: practical heuristics
- Static headroom: lam_max = c · max_t Ŝ(t) for an estimate Ŝ from a coarse sweep, with c ∈ [1.1, 2.0] depending on desired safety margin.
- Quantile-based: lam_max = Q_{0.995}(Ŝ(t)) · c to suppress rare peaks; monitor bound violations to validate safety.
- Analytical bounds: if g and h are bounded and ρ, W_o norms are known, compute a crude Lipschitz/supremum bound for S(t). In practice, empirical sweeps are simpler.

Dynamic bounds and piecewise envelopes
- Segment [0, T] into windows; maintain lam_max,w per window using a pre-pass or an adaptive running maximum. Thinning remains correct if each proposal in a window uses a valid local bound.
- Lipschitz envelope (when available): if S is L-Lipschitz and at time t₀ we have S(t₀), then for t in [t₀, t₀+δ], S(t) ≤ S(t₀) + L δ. Use this to grow the bound between re-estimates.
- Rejection cap: if empirical ρ falls below a threshold (e.g., 5%), raise lam_max or refine windows.

Expected cost
- Proposals scale with lam_max T; accepted events scale with ∫_0^T S(u) du. Choose lam_max so that ρ is comfortably high (20–50% is often a good target for research; production can push higher).

--------------------------------------------------------------------------------

## 5. Continuous-time vs discrete-time decisions

- Use continuous-time thinning when:
  - Validation requires exact TPP semantics and precise event timing.
  - You can secure a reliable lam_max with acceptable acceptance ratio.
- Use discrete-time fallback when:
  - The downstream simulator enforces a grid.
  - You require synchronized updates with other time-stepped subsystems.

--------------------------------------------------------------------------------

## 6. End-to-end workflow with C‑IDEN

- Train with exact likelihood:
  - Loss: [`Python.nll_continuous()`](src/sr_ciden/losses.py:152) integrates augmented ODEs [`Python.odeint_augmented()`](src/sr_ciden/solvers.py:455) under [`Python.odeint_wrapper()`](src/sr_ciden/solvers.py:298).
- Build inference intensity:
  - intensity_fn = [`Python.CIDENContinuous.get_intensity_fn()`](src/sr_ciden/adapters/torch.py:191).
- Sample spikes:
  - Continuous-time: [`Python.sample_ogata()`](src/sr_ciden/readout.py:152) or [`Python.sample_ogata_with_stats()`](src/sr_ciden/readout.py:242).
  - Discrete fallback: [`Python.sample_bernoulli()`](src/sr_ciden/readout.py:183).
- Validate:
  - Time-rescaling and K–S: [`Python.time_rescaling_test()`](src/sr_ciden/validation.py:169).

--------------------------------------------------------------------------------

## 7. Pseudocode: dynamic-bounded thinning

```
# Pre-pass: coarse grid to estimate S_hat(t) and define window bounds
windows = partition([0, T], length=W)
lam_bounds = []
for w in windows:
    S_hat_max = max(S_hat(t) for t in w)        # cheap estimate on coarse grid
    lam_bounds.append(c * S_hat_max + eps)

# Sampling pass
spikes = []
t = 0.0
for w, lam_w in zip(windows, lam_bounds):
    t = max(t, w.start)
    while t < w.end:
        dt = -log(max(rand(), 1e-12)) / lam_w
        t = t + dt
        if t >= w.end:
            break
        lam_vec = clamp_nonneg(intensity_fn(t))
        S = sum(lam_vec)
        p = min(S / lam_w, 1.0)
        if rand() < p:
            k = categorical_sample(lam_vec)
            spikes.append((t, k))
```

This maintains correctness provided each lam_w dominates S(t) on its window. Efficiency improves when window-local bounds are tighter than a single global lam_max.

--------------------------------------------------------------------------------

## 8. Suggested reading

- Lewis, P. A. W., & Shedler, G. S. (1979). Simulation of nonhomogeneous Poisson processes by thinning.
- Ogata, Y. (1981). On Lewis–Shedler simulation method for point processes.
- Brown, E. N., Barbieri, R., Ventura, V., Kass, R. E., & Frank, L. M. (2002). The time-rescaling theorem and its applications.

For full citations, see [docs/academy/REFERENCES.md](docs/academy/REFERENCES.md).

Where this fits in the curriculum
- Week 5 focuses on SR readout and thinning (this document), after the probabilistic foundations in 06 and before framework interop (08–09).