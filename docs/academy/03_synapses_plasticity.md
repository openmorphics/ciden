# 03 Synapses and Plasticity

Purpose
- Characterize synaptic dynamics using current-based and conductance-based formulations with common kernels.
- Present plasticity mechanisms including pair-based STDP, triplet extensions, and reward-modulated STDP (R-STDP).
- Provide framework-agnostic pseudocode that maps to event-driven updates and to sr_ciden latent jump and continuous-time integration.

Implementation anchors in sr_ciden
- Differentiable jump updates U(x_i, h⁻) implemented by [`Python.JumpMLP()`](src/sr_ciden/dynamics.py:282) and applied during piecewise integration with [`Python.integrate()`](src/sr_ciden/dynamics.py:395).
- Intensity readout converting hidden state to nonnegative rates via [`Python.IntensityHead()`](src/sr_ciden/dynamics.py:346), trained by exact marked TPP likelihood [`Python.nll_continuous()`](src/sr_ciden/losses.py:152) and augmented ODEs [`Python.odeint_augmented()`](src/sr_ciden/solvers.py:455).
- Continuous hidden dynamics via [`Python.LinearNonlinearODE()`](src/sr_ciden/dynamics.py:191) with stable generator [`Python.StableMatrixParam()`](src/sr_ciden/dynamics.py:126).

Inline references
- STDP: Bi and Poo (1998); Song, Miller and Abbott (2000).
- Time rescaling validation: Brown et al. (2002).
- Hawkes processes and self-excitation: Hawkes (1971) (conceptual relation to synaptic coupling).

--------------------------------------------------------------------------------

## 1. Synaptic dynamics

Synapses transform presynaptic spikes into postsynaptic currents or conductances through temporal kernels. Two common modeling choices:

### 1.1 Current-based synapses
- Postsynaptic current I_syn(t) is added to the membrane equation (e.g., LIF) as an external drive independent of membrane voltage V:
  I_syn(t) = ∑_j w_j ∑_n κ(t − t_j^n)
  where w_j are synaptic weights and κ is a postsynaptic current (PSC) kernel.

- Effect on LIF (current-based form):
  dV/dt = −(V − E_L)/τ_m + (1/C)[I_syn(t) + I_ext(t)]

Trade-offs
- Pro: Simpler numerically; linear superposition of currents.
- Con: Less biophysically faithful for fast membrane–synapse interactions.

### 1.2 Conductance-based synapses
- Synaptic input changes conductance g_syn(t), which interacts multiplicatively with membrane voltage relative to a reversal potential E_syn:
  I_syn(t) = g_syn(t) [E_syn − V(t)],
  g_syn(t) = ∑_j w_j ∑_n κ(t − t_j^n)

- Effect on LIF (conductance-based form):
  C dV/dt = −g_L(V − E_L) − g_syn(t)(V − E_syn) + I_ext(t)

Trade-offs
- Pro: Captures shunting inhibition and voltage dependence.
- Con: Nonlinear coupling may demand smaller time steps or semi-implicit updates.

### 1.3 Canonical kernels

- Exponential kernel (PSC):
  κ(t) = H(t) exp(−t/τ_s)
  with H the Heaviside step.

- Alpha kernel (rise and decay):
  κ(t) = H(t) (t/τ_s) exp(1 − t/τ_s)
  (equivalent to a difference of exponentials with matched peak time).

- Double exponential:
  κ(t) = H(t) (exp(−t/τ_d) − exp(−t/τ_r)) / (τ_d − τ_r), with τ_d > τ_r

Continuous-time state augmentation
- One may represent filtered presynaptic traces r_j(t) via ODEs, e.g.,
  dr_j/dt = −r_j/τ_s + ∑_n δ(t − t_j^n),
  so I_syn(t) = ∑_j w_j r_j(t) (current-based), or g_syn(t) = ∑_j w_j r_j(t) (conductance-based).

--------------------------------------------------------------------------------

## 2. Plasticity mechanisms

Plasticity changes synaptic weights in response to spike timing and modulatory signals.

### 2.1 Pair-based STDP (Bi and Poo, 1998; Song, Miller and Abbott, 2000)

Timing rule
- Δt = t_post − t_pre
- Long-term potentiation (LTP) for pre before post: Δw = A⁺ exp(−Δt/τ⁺) if Δt > 0
- Long-term depression (LTD) for post before pre: Δw = −A⁻ exp(Δt/τ⁻) if Δt < 0

Weight bounding
- Keep w in [w_min, w_max], often via clamping or soft constraints.

Trace-based implementation
- Maintain pre- and post-synaptic traces x_pre, x_post with exponential decay between spikes:
  dx_pre/dt = −x_pre/τ⁺, dx_post/dt = −x_post/τ⁻
- On a pre-spike: x_pre ← x_pre + 1; induce LTD via post-trace: Δw ← Δw − A⁻ x_post
- On a post-spike: x_post ← x_post + 1; induce LTP via pre-trace: Δw ← Δw + A⁺ x_pre

Pseudocode: pair-based STDP with traces
```
# Parameters: A_plus, A_minus, tau_plus, tau_minus, dt
# State per synapse: w, x_pre, x_post
# Events: pre spikes at t_pre[k], post spikes at t_post[m]

for t in time_grid(0, T, dt):
    # Decay traces
    x_pre  *= exp(-dt / tau_plus)
    x_post *= exp(-dt / tau_minus)

    # Handle presynaptic spikes at time t
    if is_pre_spike(t):
        x_pre += 1.0
        # LTD: depression proportional to post trace
        w += -A_minus * x_post

    # Handle postsynaptic spikes at time t
    if is_post_spike(t):
        x_post += 1.0
        # LTP: potentiation proportional to pre trace
        w += A_plus * x_pre

    w = clamp(w, w_min, w_max)
```

Triplet and rate-based refinements
- Triplet STDP introduces additional dependencies on two pre- or two post-traces, improving rate-consistency and stabilizing learning in high-rate regimes.

### 2.2 Reward-modulated STDP (R-STDP)

Three-factor learning
- Combine local eligibility traces with a delayed neuromodulatory reward r(t):
  Δw ∝ ∫ e(t) r(t) dt
- Eligibility trace e(t) typically couples pre/post spike timing (as in STDP) but is accumulated and modulated by reward signals that may arrive after the spikes.

Eligibility dynamics (example)
- e(t) decays with τ_e and is incremented on pre- or post-spikes:
  de/dt = −e/τ_e + f_pre_post(t)
- When a reward r(t) arrives, apply Δw ← η e(t) r(t); optionally low-pass r(t) into a dopaminergic trace.

Pseudocode: R-STDP with delayed reward
```
# Parameters: eta, tau_e, A_plus, A_minus, tau_plus, tau_minus
# State: w, x_pre, x_post, e
for t in time_grid(0, T, dt):
    # Decay traces
    x_pre  *= exp(-dt / tau_plus)
    x_post *= exp(-dt / tau_minus)
    e      *= exp(-dt / tau_e)

    if is_pre_spike(t):
        x_pre += 1.0
        # Update eligibility with LTD part
        e += -A_minus * x_post

    if is_post_spike(t):
        x_post += 1.0
        # Update eligibility with LTP part
        e += A_plus * x_pre

    if reward_available(t):  # scalar reward r(t)
        w += eta * e * r(t)
        w = clamp(w, w_min, w_max)
```

Remarks
- R-STDP enables delayed credit assignment through eligibility traces; stability requires appropriate bounding and possibly weight-dependent learning rates.

--------------------------------------------------------------------------------

## 3. Mapping synaptic concepts to C‑IDEN and SR-CIDEN

C‑IDEN does not explicitly integrate biophysical synapses during training; instead it learns a continuous hidden state h(t) optimized via exact TPP likelihood. However, synaptic ideas map naturally:

- Event-driven updates as jumps
  - Instantaneous post-event state shifts mimic synaptic jumps or spike-triggered adaptation: use [`Python.JumpMLP()`](src/sr_ciden/dynamics.py:282) as U(x_i, h⁻) within [`Python.integrate()`](src/sr_ciden/dynamics.py:395) or during likelihood segments in [`Python.nll_continuous()`](src/sr_ciden/losses.py:152) by providing `jump_update`.

- Latent traces as ODE channels
  - Exponential synaptic traces correspond to stable linear channels within h(t), realized by the stable generator [`Python.StableMatrixParam()`](src/sr_ciden/dynamics.py:126) and a nonlinear block in [`Python.LinearNonlinearODE()`](src/sr_ciden/dynamics.py:191).

- From membrane to intensity
  - Instead of thresholding a membrane potential, C‑IDEN uses a nonnegative intensity head λ(h, t) (K marks), with positivity enforced by [`Python.IntensityHead()`](src/sr_ciden/dynamics.py:346). Spikes are then sampled post-training via SR readout samplers: [`Python.sample_ogata()`](src/sr_ciden/readout.py:152) and [`Python.sample_bernoulli()`](src/sr_ciden/readout.py:183).

- Plasticity vs gradient training
  - Pair-based STDP and R-STDP are local rules. In C‑IDEN, parameters are optimized by gradient-based likelihood training using augmented integrals and, optionally, adjoint methods (Chen et al., 2018) through [`Python.odeint_wrapper()`](src/sr_ciden/solvers.py:298) and [`Python.odeint_augmented()`](src/sr_ciden/solvers.py:455). Local plasticity can be emulated within h(t) by designing U and f(h, t) to realize trace-like dynamics, while global optimization tunes parameters θ.

--------------------------------------------------------------------------------

## 4. Framework-agnostic synapse simulation templates

Current-based synapse with exponential kernel (discrete-time)
```
# For each presynaptic neuron j, maintain r_j
# r_j[t+1] = r_j[t] * exp(-dt / tau_s) + sum_{spikes at t} 1
# I_syn[t] = sum_j w_j * r_j[t]
```

Conductance-based synapse with double-exponential kernel (state-space form)
```
# Two filters r_fast, r_slow with tau_r < tau_d
r_fast[t+1] = r_fast[t] * exp(-dt / tau_r) + sum_{spikes at t} 1
r_slow[t+1] = r_slow[t] * exp(-dt / tau_d) + sum_{spikes at t} 1
g_syn[t]    = sum_j w_j * (r_slow[t] - r_fast[t]) / (tau_d - tau_r)
I_syn[t]    = g_syn[t] * (E_syn - V[t])
```

Trace-based STDP (pair-based) integrated into the loop
```
# Update x_pre, x_post as in Section 2.1; apply w updates on events
```

--------------------------------------------------------------------------------

## 5. Numerical and practical considerations

- Kernel discretization
  - For stability, prefer recursive filters (exact decay factors exp(−dt/τ)) over naive numerical integration.
  - For high firing rates, consider fixed-point or mixed-precision to control rounding error.

- Weight constraints and homeostasis
  - Implement soft bounds (sigmoid-mapped parameters) or additive decay toward a target range to avoid runaway potentiation or depression.

- Event scheduling
  - Event-driven backends (e.g., Brian2, NEST) can schedule synaptic updates precisely on spike times; time-stepped frameworks should interpolate spike timing within the step if needed.

- Relation to Hawkes-like coupling
  - In multivariate TPPs, λ_k(t) = μ_k + ∑_j ∑_n h_{kj}(t − t_j^n) parallels synaptic kernels; C‑IDEN generalizes this by learning λ(h(t), t) through latent ODEs rather than fixed convolutional coupling.

--------------------------------------------------------------------------------

## 6. Suggested reading

- Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons.
- Song, S., Miller, K. D., & Abbott, L. F. (2000). Competitive Hebbian learning through spike-timing-dependent synaptic plasticity.
- Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting point processes.
- Brown, E. N., Barbieri, R., Ventura, V., Kass, R. E., & Frank, L. M. (2002). The time-rescaling theorem and its applications.
- Neural ODE training and adjoint: Chen, T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018).

For full citations, see [docs/academy/REFERENCES.md](docs/academy/REFERENCES.md).

Where this fits in the curriculum
- Week 2 focuses on synaptic dynamics and plasticity (this document). The following modules connect encodings (04) and learning rules (05) to C‑IDEN training (06) and SR readout (07).