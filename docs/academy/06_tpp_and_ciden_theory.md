# 06 Temporal Point Processes and C‑IDEN Theory

Purpose
- Present a rigorous treatment of temporal point processes (TPPs) with marked events and conditional intensities.
- Derive the exact negative log-likelihood (NLL) used in C‑IDEN and explain its evaluation via augmented ODE integration.
- Detail the C‑IDEN latent dynamics: stable parameterization of the drift, nonlinear field, and differentiable jump updates.
- Explain the Time‑Rescaling Theorem and K–S testing protocol for goodness‑of‑fit.
- Provide clear pseudocode that mirrors the sr_ciden implementation.

Implementation anchors in sr_ciden
- Latent field and stable generator: [`Python.LinearNonlinearODE()`](src/sr_ciden/dynamics.py:191), [`Python.StableMatrixParam()`](src/sr_ciden/dynamics.py:126), jumps via [`Python.JumpMLP()`](src/sr_ciden/dynamics.py:282), fixed‑step piecewise integrator [`Python.integrate()`](src/sr_ciden/dynamics.py:395).
- ODE wrappers and augmented integrals: [`Python.odeint_wrapper()`](src/sr_ciden/solvers.py:298), [`Python.make_augmented_field()`](src/sr_ciden/solvers.py:391), [`Python.odeint_augmented()`](src/sr_ciden/solvers.py:455).
- Exact continuous‑time NLL: [`Python.nll_continuous()`](src/sr_ciden/losses.py:152).
- Time rescaling and K–S testing: [`Python.time_rescale()`](src/sr_ciden/validation.py:44), [`Python.time_rescaling_test()`](src/sr_ciden/validation.py:169).

Inline references
- Hawkes processes: Hawkes (1971).
- Thinning and Poisson simulation: Lewis and Shedler (1979); Ogata (1981).
- Time‑Rescaling Theorem: Brown et al. (2002).
- Neural ODE and adjoint: Chen et al. (2018).

--------------------------------------------------------------------------------

## 1. Temporal point processes: conditional intensity and marks

Event representation
- Let 0 < t₁ < t₂ < … < t_N ≤ T be event times on [0, T]. For a marked process with K marks m_i ∈ {1,…,K}, define the counting processes N_k(t) for each mark k, and the total count N(t) = ∑_k N_k(t).

Conditional intensity
- The conditional intensity for mark k given the history 𝓗_t is
  λ_k*(t) = lim_{Δ→0⁺} P{event of mark k in [t, t+Δ) | 𝓗_t} / Δ
- Stack into λ*(t) ∈ ℝ^K with total intensity S(t) = ∑_k λ_k*(t).

Marked likelihood on [0, T]
- Under standard regularity (predictable λ*, locally integrable S), the log‑likelihood is
  log L = ∑_{i=1}^N log λ_{m_i}*(t_i) − ∫_0^T S(u) du
- The negative log‑likelihood (NLL) is
  NLL = −∑_{i=1}^N log λ_{m_i}*(t_i) + ∫_0^T ∑_{k=1}^K λ_k*(u) du

Special cases
- Inhomogeneous Poisson: λ_k*(t) depends only on t (no self‑excitation).
- Hawkes (Hawkes, 1971): λ_k*(t) = μ_k + ∑_j ∑_{n} h_{kj}(t − t_j^n), with causal kernels h_{kj}.

--------------------------------------------------------------------------------

## 2. C‑IDEN latent dynamics and intensity readout

Latent hidden state
- C‑IDEN models a continuous hidden state h(t) ∈ ℝ^D via a neural ODE with optional jumps at event times. The intensity is a differentiable readout λ(h(t), t) ∈ ℝ^K with λ ≥ 0 coordinatewise.

Vector field and stability
- The field has the form
  dh/dt = f(h, t) = A h + ρ(W_h h + b [+ W_x x(t)])
  with ρ a pointwise nonlinearity.
- Stability parameterization (contractive drift):
  A = −(α I + L Lᵀ), with α > 0 enforced by a softplus mapping. This yields negative definiteness or semidefiniteness of A and encourages contracting trajectories toward a stable manifold.
  See [`Python.StableMatrixParam()`](src/sr_ciden/dynamics.py:126).

External input and jumps
- Optional continuous input x(t) drives W_x x(t) inside f (use [`Python.LinearNonlinearODE.set_input_provider()`](src/sr_ciden/dynamics.py:241)).
- At event times t_i, apply a differentiable jump update U(x_i, h⁻): h⁺ = h⁻ + U(x_i, h⁻), see [`Python.JumpMLP()`](src/sr_ciden/dynamics.py:282). This models instantaneous state updates (e.g., spike‑triggered adaptation, stimulus‑locked features).

Nonnegative intensity
- Intensity head g maps state to K nonnegative channels:
  λ(h, t) = g(W_o h + b_o) + ε, with g = softplus or exp and ε > 0 for numerical safety. See [`Python.IntensityHead()`](src/sr_ciden/dynamics.py:346).

--------------------------------------------------------------------------------

## 3. Exact NLL and augmented ODE integration

Piecewise segments between events
- The NLL requires evaluating the integral ∫ S(u) du. C‑IDEN computes this exactly (up to solver accuracy) by augmenting the ODE to accumulate intensities between events and summing segment contributions.

Augmented state
- Define z(t) = concat(h(t), I(t)) ∈ ℝ^{D+K}, with I_k(t) = ∫ λ_k(h(u), u) du.
- The augmented field is
  d/dt [h, I] = [ f(h, t), λ(h, t) ]
- On each segment [t_prev, t_next], initialize I to zero at t_prev and integrate to t_next; the increment ΔI = I(t_next) is the required integral over that segment.
- sr_ciden builds this field with [`Python.make_augmented_field()`](src/sr_ciden/solvers.py:391) and integrates with [`Python.odeint_augmented()`](src/sr_ciden/solvers.py:455).

Hit terms and jumps
- At each observed event time t_i with mark m_i, evaluate λ(h(t_i⁻), t_i) and add −log λ_{m_i}(t_i) to the NLL.
- Apply the optional jump update to obtain h(t_i⁺) before continuing to the next segment.

Adjoint and solvers
- When torchdiffeq is available, [`Python.odeint_wrapper()`](src/sr_ciden/solvers.py:298) uses adaptive solvers with optional adjoint (Chen et al., 2018) for memory efficiency. Otherwise, deterministic fixed‑step RK4 or Euler fallbacks ensure reproducible trajectories.

Implementation reference
- The full NLL orchestration is in [`Python.nll_continuous()`](src/sr_ciden/losses.py:152).

--------------------------------------------------------------------------------

## 4. Time‑Rescaling Theorem and K–S testing

Statement
- If a marked TPP model is correctly specified, the rescaled event times
  τ_i = ∫_0^{t_i} ∑_{k=1}^K λ_k*(u) du
  form the arrival times of a unit‑rate Poisson process. Then the inter‑event intervals Δτ_i = τ_i − τ_{i−1} are i.i.d. Exp(1).

Practical test
- Compute τ_i via numerical integration of dI/dt = ∑_k λ_k(t) with I(0)=0; obtain Δτ_i; apply a one‑sample K–S test against Exp(1).
- Implementation: [`Python.time_rescale()`](src/sr_ciden/validation.py:44) and [`Python.time_rescaling_test()`](src/sr_ciden/validation.py:169).

Interpretation
- A large p‑value (e.g., > 0.05) indicates no evidence to reject the null (good fit).
- A small p‑value suggests model misspecification (e.g., missing history dependence or incorrect intensity scale).

--------------------------------------------------------------------------------

## 5. Pseudocode: exact NLL via augmented segments

The following mirrors [`Python.nll_continuous()`](src/sr_ciden/losses.py:152) using [`Python.odeint_augmented()`](src/sr_ciden/solvers.py:455):

```
# Inputs:
#   f_h(h, t): hidden ODE field
#   lambda_fn(h, t): K nonnegative intensities
#   h0: initial hidden state
#   events: (times[Ni], marks[Ni]) sorted, 0 < t1 < ... < tN <= T
#   jump_update: optional U(x_i, h_minus) -> delta_h
#   T_end: horizon
#
# Output: scalar NLL

def continuous_time_nll(f_h, lambda_fn, h0, events, T_end, jump_update=None):
    times, marks = events
    t_prev = 0.0
    h_curr = h0
    nll = 0.0

    # Helper: integrate augmented system over [t_prev, t_next]
    def integrate_segment(h_start, t_prev, t_next):
        # Build evaluation grid [t_prev, t_next]
        t_eval = [t_prev, t_next]
        # Integrate augmented dynamics to get h(t_next^-), I(t_next^-)
        h_traj, I_traj = odeint_augmented(f_h, lambda_fn, h_start, K, t_eval)
        h_end = h_traj[-1]
        I_end = I_traj[-1]    # segment integrals for each mark
        return h_end, I_end

    # Iterate over observed events
    for i in range(len(times)):
        t_next = times[i]
        # Segment integral
        h_end, I_end = integrate_segment(h_curr, t_prev, t_next)
        nll += sum(I_end)  # add integral term over K marks

        # Hit term at event time
        lam_vec = lambda_fn(h_end, t_next)  # shape [K]
        k = marks[i]
        nll += -log(lam_vec[k] + eps)

        # Optional jump
        if jump_update is not None:
            delta_h = jump_update(x_i=None, h_minus=h_end)
            h_curr = h_end + delta_h
        else:
            h_curr = h_end

        t_prev = t_next

    # Final segment up to T_end
    h_end, I_end = integrate_segment(h_curr, t_prev, T_end)
    nll += sum(I_end)
    return nll
```

Notes
- Differentiability: the entire computation is differentiable w.r.t. the parameters in f_h, lambda_fn, and jump_update.
- Numerical accuracy: choose tolerances (adaptive) or dt (fixed‑step) to balance speed‑accuracy trade‑offs. Deterministic fallbacks are helpful for testing and reproducibility.

--------------------------------------------------------------------------------

## 6. Relation to Hawkes processes and classical models

- Linear Hawkes corresponds to choosing λ(h(t), t) as a linear convolution of past spikes with causal kernels plus a baseline. C‑IDEN generalizes this by maintaining a latent ODE state whose dynamics can represent multi‑timescale memory, nonlinear interactions, and input‑driven modulation.
- With a stable A and simple ρ, C‑IDEN can emulate decaying traces and linear‑nonlinear Poisson models. With jumps U at events, it captures instantaneous after‑effects.

--------------------------------------------------------------------------------

## 7. Practical guidance

- Stable generator A
  - Prefer A = −(α I + L Lᵀ) to ensure contractive linear drift; if using a free A, consider adding a spectral penalty externally.

- Intensity head choice
  - softplus is numerically stable and avoids overflow; exp may need pre‑exp clamping (see [`Python.IntensityHead()`](src/sr_ciden/dynamics.py:346)).

- Solver selection
  - For research notebooks, start with adaptive solvers and adjoint if installed; for CI or deterministic baselines, use RK4 at a tuned dt (see [`Python.odeint_wrapper()`](src/sr_ciden/solvers.py:298)).

- Validation
  - Always report Time‑Rescaling K–S p‑values alongside task metrics. Plot PIT histograms or QQ plots of Δτ against Exp(1) as secondary checks.

--------------------------------------------------------------------------------

## 8. Suggested reading

- Hawkes, A. G. (1971). Spectra of some self‑exciting and mutually exciting point processes.
- Lewis, P. A. W., & Shedler, G. S. (1979). Simulation of nonhomogeneous Poisson processes by thinning.
- Ogata, Y. (1981). On Lewis‑Shedler simulation method for point processes.
- Brown, E. N., Barbieri, R., Ventura, V., Kass, R. E., & Frank, L. M. (2002). The time‑rescaling theorem and its applications.
- Chen, T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural Ordinary Differential Equations.

For full citations, see [docs/academy/REFERENCES.md](docs/academy/REFERENCES.md).

Where this fits in the curriculum
- Week 4 centers on this document to establish probabilistic training foundations before moving to SR readout (07), frameworks (08–09), and applications (10–11).