# 04 Temporal Encoding

Purpose
- Survey canonical spike-based encodings: rate, latency (time-to-first-spike), population coding, and rank-order coding.
- Discuss trade-offs among precision, bandwidth, robustness, and implementation cost.
- Provide framework-agnostic templates for converting continuous inputs into spike trains that later feed C‑IDEN training (as features/marks) and SR readout inference.

Implementation anchors in sr_ciden
- External continuous input x(t) can be injected into the latent ODE via [`Python.LinearNonlinearODE.set_input_provider()`](src/sr_ciden/dynamics.py:241) and modulate dh/dt (see [`Python.LinearNonlinearODE()`](src/sr_ciden/dynamics.py:191)).
- Event-time features can drive jump updates U(x_i, h⁻) using [`Python.JumpMLP()`](src/sr_ciden/dynamics.py:282) applied in the piecewise integrator [`Python.integrate()`](src/sr_ciden/dynamics.py:395) or during likelihood segments in [`Python.nll_continuous()`](src/sr_ciden/losses.py:152).
- After training the intensity head λ(h, t), spike synthesis is performed by SR readout samplers [`Python.sample_ogata()`](src/sr_ciden/readout.py:152) or [`Python.sample_bernoulli()`](src/sr_ciden/readout.py:183).

Inline references
- Textbook background on neural coding: Dayan and Abbott (2001).

--------------------------------------------------------------------------------

## 1. Rate coding

Definition
- Encode a scalar signal s(t) ≥ 0 as an instantaneous firing rate r(t) = α · s(t), and generate an inhomogeneous Poisson spike train with intensity r(t).

Properties
- Pros: robust to timing noise, simple to implement, matches Poisson assumptions.
- Cons: discards precise timing; may require high firing rates to achieve low variance estimates (Fano ~ 1).

Pseudocode: rate-to-Poisson spikes
```
# Inputs: continuous signal s(t) sampled on grid, scale alpha > 0
# Output: spike times for a single channel
spikes = []
t = 0.0
while t < T:
    r = max(alpha * s(t), 0.0)         # instantaneous rate
    p = 1 - exp(-r * dt)               # Bernoulli thinning per time bin
    if rand() < p:
        spikes.append(t)
    t += dt
```

Use with C‑IDEN
- Treat the resulting spikes as observed events for one mark (K=1), or distribute across K marks by channelizing inputs. C‑IDEN learns λ(h, t) to match observed rates while preserving continuous-time dynamics in h(t).

--------------------------------------------------------------------------------

## 2. Latency (time-to-first-spike) coding

Definition
- Map a scalar s in [s_min, s_max] to a single spike time within a window [t0, t1]:
  t_spike(s) = t0 + (t1 − t0) · (1 − φ(s)), where φ is a normalized monotonic map (e.g., φ(s) = (s − s_min) / (s_max − s_min)).

Properties
- Pros: very low spike count; preserves ordering information; latency can be decoded precisely.
- Cons: non-robust when noise perturbs a single spike; not ideal for time-varying s(t) unless windows slide or overlap.

Pseudocode: time-to-first-spike per window
```
# Inputs: scalar s in [s_min, s_max], window [t0, t1]
phi = clamp((s - s_min) / max(s_max - s_min, eps), 0.0, 1.0)
t_spk = t0 + (t1 - t0) * (1.0 - phi)
emit_spike(t_spk)
```

Use with C‑IDEN
- Produce one event per window per feature channel; use mark indices to represent channels. Latency carries magnitude; the learned intensity λ(h, t) can capture temporal structure induced by these latencies.

--------------------------------------------------------------------------------

## 3. Population coding

Definition
- Represent a scalar s by a population of M neurons with overlapping tuning curves g_m(s) (e.g., Gaussian bump), and encode each channel via rate or latency.

Gaussian tuning example
- g_m(s) = exp(−(s − μ_m)² / (2σ²)), with centers μ_m spanning the stimulus range.

Pseudocode: population rate coding
```
# Inputs: s(t), centers mu[m], width sigma
for m in 0..M-1:
    r_m = alpha * exp(-0.5 * ((s(t) - mu[m]) / sigma)**2)
    p_m = 1 - exp(-r_m * dt)
    if rand() < p_m:
        spikes.append((t, m))  # mark m
```

Properties
- Pros: supports robust, linear decoders; captures uncertainty via distributed activity.
- Cons: increases channel count K and bandwidth; requires calibration of μ_m and σ.

Use with C‑IDEN
- Map each population channel to a mark k ∈ {0..K−1}; C‑IDEN learns λ_k(h, t). Population design impacts identifiability and sample efficiency.

--------------------------------------------------------------------------------

## 4. Rank-order coding

Definition
- Encode a vector stimulus by the order in which channels first spike, ignoring exact spike times within a small window. Earlier spikes indicate stronger feature activations.

Properties
- Pros: extremely sparse and low latency; robust to small timing noise if order preserved.
- Cons: discards magnitude beyond ordering; requires attention to tie-breaking.

Pseudocode: rank-order emission
```
# Inputs: features f[m], window [t0, t1], monotone map psi
scores = [(m, psi(f[m])) for m in 0..M-1]
# higher score -> earlier spike
scores.sort(key=lambda x: -x[1])
for i, (m, s) in enumerate(scores):
    t_spk = t0 + i * (t1 - t0) / M
    emit_spike((t_spk, m))
```

Use with C‑IDEN
- Provide a burst of at most M spikes per window carrying rank information via marks. The learned intensity λ(h, t) can reproduce order statistics if h(t) and jump updates retain sufficient memory.

--------------------------------------------------------------------------------

## 5. Trade-offs and selection heuristics

- Precision vs robustness
  - Rate coding offers variance reduction via time averaging but at energy cost.
  - Latency and rank-order maximize information per spike but are sensitive to jitter; use short windows and calibration.

- Bandwidth and K
  - Population coding increases K; ensure the intensity head and data volume support identifiability (see [`Python.IntensityHead()`](src/sr_ciden/dynamics.py:346)).

- Task fit
  - Temporal recognition and event cameras: rate or latency per pixel polarity channel; sometimes rank-order within local receptive fields.
  - Control loops: population rate coding for state variables (angles, velocities) with continuous updates.

--------------------------------------------------------------------------------

## 6. Mapping encodings into the C‑IDEN pipeline

- As continuous drives
  - Use an external x(t) provider to modulate dh/dt via [`Python.LinearNonlinearODE.set_input_provider()`](src/sr_ciden/dynamics.py:241). Suitable for rate or population-amplitude drives when event times are dense or when you wish to avoid extra marks.

- As event marks and jumps
  - Emit events with marks for channels and pass features x_i to [`Python.JumpMLP()`](src/sr_ciden/dynamics.py:282) at event times (piecewise via [`Python.integrate()`](src/sr_ciden/dynamics.py:395) or internally in [`Python.nll_continuous()`](src/sr_ciden/losses.py:152)). Suitable when encoding is naturally sparse (latency, rank-order).

- As observation targets
  - When encodings produce observed output spikes, train C‑IDEN to model them via λ(h, t) and evaluate with time-rescaling and K–S tests (see [`Python.time_rescaling_test()`](src/sr_ciden/validation.py:169)).

--------------------------------------------------------------------------------

## 7. Practical calibration

- Windowing
  - Choose [t0, t1] to match task latency requirements; ensure windows tile the sequence without leakage for validation.

- Rate scales
  - Normalize s(t) to produce biologically plausible rates (Hz) and enforce bounds to control Bernoulli saturation (p < 1 − 1e−12).

- Population parameters
  - Set μ_m uniformly over the stimulus range; choose σ to ensure ~2–3 overlaps at any s for stable decoding.

- Energy considerations
  - Prefer sparse encodings where downstream SR readout maintains accuracy (see hardware deployment notes in 11).

--------------------------------------------------------------------------------

## 8. Suggested reading

- Dayan, P., & Abbott, L. F. (2001). Theoretical Neuroscience.

For full citations, see [docs/academy/REFERENCES.md](docs/academy/REFERENCES.md).

Where this fits in the curriculum
- Week 3 introduces encoding strategies (this document) alongside spike learning rules (05). Subsequent modules (06–07) connect encodings to continuous-time intensity modeling and SR readout.