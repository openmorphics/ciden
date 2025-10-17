# 05 Spike Learning Rules: Surrogate Gradients and Unsupervised Plasticity

Purpose
- Summarize supervised training with surrogate gradients for spiking neurons and contrast it with C‑IDEN’s exact-likelihood training of continuous-time intensities.
- Present unsupervised/local learning rules (STDP variants and three-factor R‑STDP) and discuss local vs global error signals.
- Provide framework-agnostic pseudocode templates.

Implementation anchors in sr_ciden
- C‑IDEN training maximizes the exact marked-TPP likelihood with a differentiable intensity readout; no surrogate gradient for spike non-differentiability is needed in the training objective. See [`Python.nll_continuous()`](src/sr_ciden/losses.py:152), augmented integration [`Python.odeint_augmented()`](src/sr_ciden/solvers.py:455), and solver wrapper [`Python.odeint_wrapper()`](src/sr_ciden/solvers.py:298).
- The Neural ODE hidden field and stable generator are fully differentiable: [`Python.LinearNonlinearODE()`](src/sr_ciden/dynamics.py:191), [`Python.StableMatrixParam()`](src/sr_ciden/dynamics.py:126), intensity head [`Python.IntensityHead()`](src/sr_ciden/dynamics.py:346).
- SR readout sampling is decoupled from training and executed with inference-only samplers [`Python.sample_ogata()`](src/sr_ciden/readout.py:152) and [`Python.sample_bernoulli()`](src/sr_ciden/readout.py:183). In a PyTorch workflow, see [`Python.CIDENContinuous.forward()`](src/sr_ciden/adapters/torch.py:157) and the inference helper [`Python.CIDENContinuous.get_intensity_fn()`](src/sr_ciden/adapters/torch.py:191).

Inline references
- Surrogate gradients overview: Neftci, Mostafa, and Zenke (2019).
- STDP: Bi and Poo (1998); Song, Miller, and Abbott (2000).

--------------------------------------------------------------------------------

## 1. Surrogate gradients for spiking neurons

Problem setup
- A deterministic spiking nonlinearity is often modeled as a hard threshold s = H(u − θ) (Heaviside). Its derivative is zero almost everywhere and undefined at θ, preventing gradient backpropagation through spike generation.

Surrogate idea
- During backprop, replace ∂H/∂u with a smooth surrogate ϕ′(u − θ) while keeping the forward pass unchanged (hard threshold). Common surrogates include:
  - Sigmoid derivative: ϕ′(x) = β σ(βx) (1 − σ(βx))
  - Fast sigmoid derivative: ϕ′(x) = 1 / (1 + |βx|)²
  - Piecewise linear clip: ϕ′(x) = 1 if |x| ≤ 1, else 0, scaled by β

Forward (deterministic spikes)
- u[t+1] = f(u[t], I[t], …)     neuron state update (e.g., LIF)
- s[t] = H(u[t] − θ)

Backward (surrogate derivative)
- Replace ∂s/∂u by ϕ′(u − θ) in chain rule; propagate gradients through time using BPTT or truncated BPTT.

Pseudocode: surrogate-gradient training loop (discrete time)
```
# Parameters: theta, beta, optimizer
# Surrogate derivative phi_prime(x) = beta * sigmoid(beta * x) * (1 - sigmoid(beta * x))
def forward_surrogate(inputs, targets):
    u = u0  # membrane initialization
    S = []  # spike outputs
    for t in time_grid:
        u = update_membrane(u, inputs[t])        # e.g., LIF or Izhikevich
        s = heaviside(u - theta)                 # forward uses hard threshold
        S.append(s)
        u = reset_if_spike(u, s)                 # reset and refractory handling
    y = readout(S)                                # task-specific head
    loss = task_loss(y, targets)
    return loss, (S, u_trajectory)

optimizer.zero_grad()
loss, cache = forward_surrogate(x, y_true)
# Backward pass where autograd graph contains phi_prime instead of dH/du
loss.backward()  # gradients flow through surrogate derivatives
optimizer.step()
```

Notes
- Use BPTT truncation for long sequences to control memory.
- Calibrate β; too small underestimates gradients; too large approximates a step and can destabilize training.

Contrast with C‑IDEN
- C‑IDEN avoids non-differentiable spike generation during training: the intensity λ(h, t) = g(W_o h + b_o) is differentiable (e.g., softplus) and optimized by exact-likelihood terms handled by augmented ODE integration [`Python.nll_continuous()`](src/sr_ciden/losses.py:152). Spikes are produced only at inference via SR readout.

--------------------------------------------------------------------------------

## 2. Unsupervised learning with STDP variants

Pair-based STDP (Bi & Poo 1998; Song, Miller & Abbott 2000)
- Δw depends on Δt = t_post − t_pre:
  - LTP for Δt > 0: Δw = A⁺ exp(−Δt/τ⁺)
  - LTD for Δt < 0: Δw = −A⁻ exp(Δt/τ⁻)
- Trace-based implementation with decaying pre/post traces x_pre and x_post enables online updates (see 03 Synapses and Plasticity).

Triplet and rate-consistent STDP
- Incorporate additional pre/post coincidence terms to better match realistic firing-rate dependence and stabilize learning at high rates.

Homeostatic constraints
- Keep weights within [w_min, w_max] or use weight-dependent updates to prevent runaway potentiation/depression:
  - Multiplicative STDP: Δw ∝ (w_max − w) for LTP and Δw ∝ (w − w_min) for LTD.

Pseudocode: online pair-based STDP with traces
```
# As in 03, with per-synapse states w, x_pre, x_post
for each time bin t:
    x_pre  *= exp(-dt / tau_plus)
    x_post *= exp(-dt / tau_minus)
    if pre_spike_at(t):
        x_pre += 1.0
        w += -A_minus * x_post
    if post_spike_at(t):
        x_post += 1.0
        w += A_plus * x_pre
    w = clamp(w, w_min, w_max)
```

Remarks
- STDP is inherently local; for structured tasks it benefits from architectural biases or modulatory signals (see three-factor rules below).

--------------------------------------------------------------------------------

## 3. Reward-modulated STDP (three-factor rules)

Concept
- Combine local eligibility traces e(t) (capturing spike timing) with a delayed, possibly sparse scalar reward r(t). Weight updates integrate the product e(t) r(t) over time:
  - Δw ∝ ∫ e(t) r(t) dt
- Eligibility traces obey de/dt = −e/τ_e + f_pre_post(t), where f_pre_post captures local STDP-like increments on pre/post spikes.

Pseudocode: R‑STDP with delayed reward
```
# States per synapse: w, e, x_pre, x_post
for each time bin t:
    x_pre  *= exp(-dt / tau_plus)
    x_post *= exp(-dt / tau_minus)
    e      *= exp(-dt / tau_e)

    if pre_spike_at(t):
        x_pre += 1.0
        e += -A_minus * x_post

    if post_spike_at(t):
        x_post += 1.0
        e += A_plus * x_pre

    if reward_available(t):
        w += eta * e * r(t)
        w = clamp(w, w_min, w_max)
```

Remarks
- R‑STDP implements delayed credit assignment locally and can be combined with constraints (e.g., synaptic normalization) for stability.

--------------------------------------------------------------------------------

## 4. Local vs global error signals

- Global error backpropagation (BPTT) requires differentiable dynamics and a task loss; surrogate gradients make hard thresholds differentiable in the backward pass. This is the standard SNN training approach for classification/regression tasks.
- Local rules (STDP/R‑STDP) rely on pre/post coincidences and modulatory third factors; they are scalable and biologically plausible but typically require careful architectural design for task-level performance.

Position of C‑IDEN
- C‑IDEN’s objective is a probabilistic global criterion (exact TPP log-likelihood) with fully differentiable continuous-time dynamics. It sidesteps surrogate spike derivatives in training and retains sampling as an inference-only step. This enables:
  - Stable ODE-based credit assignment using adjoint methods when available via [`Python.odeint_wrapper()`](src/sr_ciden/solvers.py:298).
  - Post-training sampling with precise statistical semantics (Ogata thinning) [`Python.sample_ogata()`](src/sr_ciden/readout.py:152), and principled model validation by time rescaling [`Python.time_rescaling_test()`](src/sr_ciden/validation.py:169).

--------------------------------------------------------------------------------

## 5. Framework-agnostic templates

Template A: surrogate-gradient classification with LIF neurons
```
# Build an SNN layer stack with LIF units and a readout
# Use a surrogate derivative phi_prime in autograd for the threshold
for epoch in 1..E:
    for batch in data:
        loss = forward_surrogate(batch.inputs, batch.targets)
        optimizer.zero_grad()
        loss.backward()              # gradients via surrogate
        clip_grad_norm(parameters)   # stability
        optimizer.step()
```

Template B: C‑IDEN exact-likelihood training and SR readout
```
# Training
model = CIDENContinuous(hidden_dim=H, num_marks=K, ...)
# model.forward wires dynamics and intensity to exact NLL
loss = model(events=(times, marks), T_end=T, reduction="mean")  # uses nll_continuous inside
loss.backward()
optimizer.step()

# Inference readout
intensity_fn = model.get_intensity_fn(h_init=None)
spikes = sample_ogata(intensity_fn, T=T, lam_max=Lmax, seed=seed)
# or discrete-time fallback
spikes_dt = sample_bernoulli(intensity_fn, T=T, dt=dt, seed=seed)
```
See [`Python.CIDENContinuous()`](src/sr_ciden/adapters/torch.py:36), [`Python.CIDENContinuous.forward()`](src/sr_ciden/adapters/torch.py:157), [`Python.sample_ogata()`](src/sr_ciden/readout.py:152), and [`Python.sample_bernoulli()`](src/sr_ciden/readout.py:183).

--------------------------------------------------------------------------------

## 6. Suggested reading

- Neftci, E. O., Mostafa, H., & Zenke, F. (2019). Surrogate gradient learning in spiking neural networks.
- Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons.
- Song, S., Miller, K. D., & Abbott, L. F. (2000). Competitive Hebbian learning through spike-timing-dependent synaptic plasticity.

For full citations, see [docs/academy/REFERENCES.md](docs/academy/REFERENCES.md).

Where this fits in the curriculum
- Week 3 introduces surrogate gradients (this document) alongside encodings (04). Weeks 4–5 connect probabilistic training (06) to SR readout (07), showing how C‑IDEN achieves continuous-time learning without surrogate spike derivatives in the objective.