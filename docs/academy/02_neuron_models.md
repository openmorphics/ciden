# 02 Neuron Models: LIF, QIF, and Izhikevich

Purpose
- Present canonical spiking neuron models used in SNN education and research: Leaky Integrate-and-Fire (LIF), Quadratic Integrate-and-Fire (QIF), and Izhikevich.
- Define membrane equations, resets, and refractoriness; show time-stepped simulation pseudocode suitable for later translation into multiple frameworks.
- Map the conceptual state dynamics to the continuous-time hidden state h(t) and intensity readout used by C‑IDEN.

Implementation anchors in sr_ciden
- Continuous-time latent dynamics as a Neural ODE are generic and can subsume canonical neuron fields when desired. See [`Python.LinearNonlinearODE()`](src/sr_ciden/dynamics.py:191), stable generator [`Python.StableMatrixParam()`](src/sr_ciden/dynamics.py:126), and piecewise integration with events [`Python.integrate()`](src/sr_ciden/dynamics.py:395).
- ODE solver backends and augmented integrals: [`Python.odeint_wrapper()`](src/sr_ciden/solvers.py:298), [`Python.odeint_augmented()`](src/sr_ciden/solvers.py:455).
- Marked intensity readout and exact-likelihood training: [`Python.IntensityHead()`](src/sr_ciden/dynamics.py:346), [`Python.nll_continuous()`](src/sr_ciden/losses.py:152).
- PyTorch adapter for training loops and inference helper: [`Python.CIDENContinuous()`](src/sr_ciden/adapters/torch.py:36), [`Python.CIDENContinuous.get_intensity_fn()`](src/sr_ciden/adapters/torch.py:191).

Notation
- V: membrane potential (mV), I_syn(t): synaptic/input current (nA or normalized units), C: capacitance, g_L: leak conductance, E_L: leak reversal, τ_m = C/g_L: membrane time constant.
- Δt: fixed time step for discrete simulation; in continuous-time formulations we use t ∈ [0, T].
- Threshold crossing emits a spike; a reset and an absolute refractory period follow.

Inline references
- Canonical IF neuron and biophysics: Dayan and Abbott (2001).
- QIF canonical form and normal form arguments (used in reduction of Type I excitability): Ermentrout and Terman (2010).
- Izhikevich reduced model for diverse firing patterns: Izhikevich (2003).

--------------------------------------------------------------------------------

## 1. Leaky Integrate-and-Fire (LIF)

Biophysical membrane equation (current-based)
- C dV/dt = −g_L (V − E_L) + I_syn(t) + I_ext(t)
- Divide by C to obtain a standard form:
  dV/dt = −(V − E_L)/τ_m + (1/C) [I_syn(t) + I_ext(t)]
- Alternatively: using resistance R_m = 1/g_L,
  τ_m dV/dt = −(V − E_L) + R_m [I_syn(t) + I_ext(t)]

Spike, reset, refractory
- If V(t⁻) ≥ V_th, emit spike at t and set V(t⁺) ← V_r (often below E_L).
- Absolute refractory: hold V fixed (or clamped) for τ_ref after a spike.
- Relative refractory (optional): increase effective threshold or reduce membrane gain for a short window post-spike.

Discrete-time Euler update (Δt small)
- V[t+1] = V[t] + Δt [ −(V[t] − E_L)/τ_m + (1/C) I_total[t] ]
- With absolute refractory: if t is within refractory window, skip update and keep V = V_r.

Pseudocode: LIF with current-based synapses
```
# Parameters: E_L, V_th, V_r, tau_m, C, dt, t_end, tau_ref
# Inputs: I_syn[t], I_ext[t] sampled on grid
V = V_init
t = 0
t_ready = 0   # next time when neuron leaves refractory
spikes = []
while t < t_end:
    if t < t_ready:
        # absolute refractory
        V = V_r
    else:
        I_total = I_syn[t] + I_ext[t]
        dV = (-(V - E_L)/tau_m + (1.0/C)*I_total) * dt
        V = V + dV

        if V >= V_th:
            spikes.append(t)
            V = V_r
            t_ready = t + tau_ref

    t = t + dt
```

Remarks
- Stability: Euler requires sufficiently small Δt relative to τ_m. RK2 or RK4 reduces numerical error.
- Conductance-based synapses: replace current term by g_syn(t)(E_syn − V); see 03 Synapses and Plasticity.

--------------------------------------------------------------------------------

## 2. Quadratic Integrate-and-Fire (QIF)

Canonical form (dimensionless)
- τ dV/dt = V² + I(t)
- Spike when V → +∞; reset to V → −∞. In practice, use large finite thresholds: V ≥ V_max triggers a spike and reset to V_reset ≪ 0.
- QIF captures Type I excitability near a saddle-node on invariant circle bifurcation (normal form). The quadratic nonlinearity produces divergent slope near spike emission.

Regularized practical form
- τ dV/dt = V² + η + I(t) with offset η controlling intrinsic excitability.

Pseudocode: QIF with finite thresholds
```
# Parameters: tau, V_max (e.g., +100), V_reset (e.g., -100), dt, t_end
# Inputs: I[t]
V = V_init
spikes = []
t = 0
while t < t_end:
    dV = ((V*V) + I[t]) / tau * dt
    V = V + dV

    if V >= V_max:
        spikes.append(t)
        V = V_reset

    t = t + dt
```

Remarks
- Step size sensitivity: near spiking, slope diverges; small dt or semi-implicit methods help.
- Phase transformation (theta neuron) provides an exact spike mapping; many event-driven formulations derive from this.

--------------------------------------------------------------------------------

## 3. Izhikevich Model

Two-variable dynamical system
- dv/dt = 0.04 v² + 5 v + 140 − u + I(t)
- du/dt = a (b v − u)
- Spike/reset rule: if v ≥ 30 mV then emit spike; set v ← c and u ← u + d

Parameters
- a, b, c, d shape the firing pattern (regular spiking, fast spiking, bursting, chattering, etc.). See Izhikevich (2003) for canonical parameter sets.

Pseudocode: Izhikevich update
```
# Parameters: a, b, c, d; dt; t_end
# Inputs: I[t]
v = v_init
u = b * v
spikes = []
t = 0
while t < t_end:
    # Using Euler or 2-step "half-step" recommended by original article for numerical robustness
    dv = (0.04*v*v + 5*v + 140 - u + I[t]) * dt
    du = (a*(b*v - u)) * dt
    v = v + dv
    u = u + du

    if v >= 30.0:
        spikes.append(t)
        v = c
        u = u + d

    t = t + dt
```

Remarks
- The polynomial dv/dt can be stiff under large I; reduce dt to stabilize.
- A 2-step (Heun-like) or RK2 update improves accuracy versus plain Euler.

--------------------------------------------------------------------------------

## 4. Reset dynamics and refractoriness

Absolute refractory
- A fixed duration τ_ref following a spike during which the neuron cannot spike and V is clamped at V_r (or the dynamics are paused).

Relative refractory (optional)
- Transient adaptation implemented as a dynamic threshold V_th(t) or an adaptation current (e.g., variable u in Izhikevich).

Event emission semantics
- For time-stepped simulation, emit a spike when V crosses threshold; some frameworks interpolate the spike time within the step using linear or quadratic reconstruction to reduce temporal jitter.

--------------------------------------------------------------------------------

## 5. Time-stepped simulation loops: general template

This template factors synaptic updates from membrane updates. It supports multiple neuron models by plugging in a per-neuron f(V, u, t) and an update rule.

```
# Precompute or stream synaptic input I_syn[t] per neuron
init_state(neuron_idx) -> (V, aux)   # aux may include u, refractory counters

for t in time_grid(0, T, dt):
    for n in neurons:
        V, aux = state[n]

        # Refractory handling
        if aux.refractory_until > t:
            V = aux.V_reset
            continue

        # Compute membrane derivative f; depends on model
        dstate = f(neuron=n, V=V, aux=aux, I=I_syn[n, t], t=t)
        state[n] = integrate(state[n], dstate, dt)  # Euler, RK2, or RK4

        # Threshold and reset
        if state[n].V >= V_th[n]:
            emit_spike(n, t)
            state[n].V = V_reset[n]
            aux.refractory_until = t + tau_ref[n]
            apply_post_spike_updates(aux)  # e.g., u += d for Izhikevich
```

--------------------------------------------------------------------------------

## 6. Mapping canonical neurons to C‑IDEN latent dynamics

C‑IDEN uses a continuous-time hidden state h(t) with an intensity readout λ(h(t), t) (K marked channels). This differs from neuron-centric membrane dynamics in two important ways:

- Hidden-state abstraction
  - h(t) need not correspond to a single membrane potential; it can embed multi-timescale latent features that summarize past events and inputs. The field dh/dt = f(h, t) is implemented by [`Python.LinearNonlinearODE()`](src/sr_ciden/dynamics.py:191) and can mirror LIF-like linear drift plus nonlinearity (e.g., tanh/softplus), or more expressive neural fields.

- Intensity readout
  - The spike generation in C‑IDEN is not by threshold crossing of V, but by a marked Poisson intensity head λ_k(h(t)) ≥ 0, implemented by [`Python.IntensityHead()`](src/sr_ciden/dynamics.py:346). During training, we maximize the likelihood of observed events against this intensity via [`Python.nll_continuous()`](src/sr_ciden/losses.py:152).

- Event effects and jumps
  - Instantaneous event-driven state updates U(x_i, h⁻) can mimic synaptic jumps or after-spike resets within the latent space, implemented by [`Python.JumpMLP()`](src/sr_ciden/dynamics.py:282) and applied piecewise in [`Python.integrate()`](src/sr_ciden/dynamics.py:395).

- Inference decoupling
  - After training, spike synthesis is performed via SR readout samplers on the learned intensity using Ogata’s thinning or a discrete Bernoulli grid (see [`Python.sample_ogata()`](src/sr_ciden/readout.py:152), [`Python.sample_bernoulli()`](src/sr_ciden/readout.py:183)). This preserves reproducibility and separates gradient-bearing computations from sampling steps.

Consequently, one can:
- Emulate LIF-like behavior by choosing f(h, t) = A h + ρ(W_h h + b) with stable A = −(αI + L Lᵀ) (drift to baseline) and λ = g(W_o h + b_o) to modulate event rate.
- Emulate threshold-reset by making U(x_i, h⁻) a learned function that drives h toward a reset manifold following events.
- Recover biophysical interpretability by selecting bases in h that correspond to membrane proxies and synaptic traces, if desired.

--------------------------------------------------------------------------------

## 7. Numerical considerations

- Time step choice
  - LIF is usually well-behaved with dt ≪ τ_m; QIF near spike requires smaller dt. Izhikevich typically uses dt = 1 ms with a corrective half-step; RK2/Heun or RK4 reduce discretization error.

- Stability and drift control
  - For latent ODEs in C‑IDEN, the stable generator A = −(αI + L Lᵀ) (see [`Python.StableMatrixParam()`](src/sr_ciden/dynamics.py:126)) ensures contractive linear part and helps avoid hidden-state blow-up. If unconstrained A is used, a spectral penalty is recommended externally.

- Determinism and reproducibility
  - Continuous-time training relies on ODE integration; when [`Python.odeint_wrapper()`](src/sr_ciden/solvers.py:298) uses adaptive solvers (torchdiffeq), tolerances (rtol, atol) affect trajectories and gradients. Fixed-step fallbacks [`Python.rk4_solve()`](src/sr_ciden/solvers.py:221) and [`Python.euler_solve()`](src/sr_ciden/solvers.py:144) provide deterministic baselines for testing.

--------------------------------------------------------------------------------

## 8. Suggested reading

- Dayan, P., & Abbott, L. F. (2001). Theoretical Neuroscience.
- Ermentrout, G. B., & Terman, D. H. (2010). Mathematical Foundations of Neuroscience.
- Izhikevich, E. M. (2003). Simple model of spiking neurons.
- For SNN learning and plasticity, see Bi & Poo (1998) and Song, Miller & Abbott (2000) (covered in 03 and 05).
- For continuous-time training and validation within C‑IDEN, see Chen et al. (2018) and Brown et al. (2002) (covered in 06).

For full citations, see [docs/academy/REFERENCES.md](docs/academy/REFERENCES.md).

Where this fits in the curriculum
- Week 1 focuses on neuron models (this document) and prerequisites (01).
- 03 extends to synapses and plasticity; 04 covers temporal encodings; 05 introduces surrogate gradients for spiking systems.
- 06 connects these dynamics to temporal point processes and the C‑IDEN likelihood.