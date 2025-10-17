# 12 Evaluation Metrics and Benchmarking Protocols

Purpose
- Define standardized, reproducible metrics for continuous-time intensity modeling (C‑IDEN) and SR readout.
- Provide concrete computation recipes, pseudocode, and reporting templates suitable for academic review.
- Tie each metric to the sr_ciden implementation for traceability.

Core sr_ciden anchors
- Exact NLL for marked TPPs: [`Python.nll_continuous()`](src/sr_ciden/losses.py:152), augmented integrals via [`Python.odeint_augmented()`](src/sr_ciden/solvers.py:455), solver configuration [`Python.ODESolverConfig()`](src/sr_ciden/solvers.py:86).
- Time rescaling and K–S: [`Python.time_rescaling_test()`](src/sr_ciden/validation.py:169) and components [`Python.time_rescale()`](src/sr_ciden/validation.py:44), [`Python.ks_test()`](src/sr_ciden/validation.py:126).
- SR readout efficiency: [`Python.sample_ogata_with_stats()`](src/sr_ciden/readout.py:242), [`Python.sample_ogata()`](src/sr_ciden/readout.py:152), [`Python.sample_bernoulli()`](src/sr_ciden/readout.py:183).

Inline references
- TPP theory and rescaling: Hawkes (1971); Brown et al. (2002).
- Thinning: Lewis and Shedler (1979); Ogata (1981).
- Neural ODEs: Chen et al. (2018).

--------------------------------------------------------------------------------

## 1. Likelihood-based metrics

Definitions
- Event-level negative log-likelihood (NLL) for marked process on [0, T]:
  NLL = −∑_i log λ_{m_i}(t_i) + ∫_0^T ∑_{k=1}^K λ_k(u) du
- Average NLL per event: NLL / N
- Average NLL per unit time: NLL / T
- Perplexity (optional): exp(NLL / N) for comparability with count models.

Computation
- Use [`Python.nll_continuous()`](src/sr_ciden/losses.py:152) with consistent solver config [`Python.ODESolverConfig()`](src/sr_ciden/solvers.py:86). Report reduction="none" to obtain per-sequence values, then aggregate (mean ± std).

Protocol
- Split data (train/val/test) with fixed seeds; tune on validation NLL; report test NLL.
- Include solver tolerances or fixed dt in the report; changing them changes values.

--------------------------------------------------------------------------------

## 2. Goodness-of-fit: Time-Rescaling and K–S

Brown et al. (2002)
- If λ*(t) is correct, τ_i = ∫_0^{t_i} ∑_k λ_k(u) du are arrival times of unit-rate Poisson; Δτ_i are i.i.d. Exp(1).

Metrics
- K–S statistic D and p-value from a one-sample test against Exp(1).
- Optional: PIT histogram (probability integral transform) and QQ-plot of Δτ vs Exp(1).

Computation
- Use [`Python.time_rescaling_test()`](src/sr_ciden/validation.py:169) which orchestrates τ computation and K–S.
- Report (D, p) on held-out sequences and aggregate across sequences (median or mean).

Pseudocode
```
# events = (times, marks)
result = time_rescaling_test(intensity_fn, events)  # {'D': float, 'p_value': float}
report(result['D'], result['p_value'])
```

Interpretation
- Larger p-value suggests no evidence to reject the null (better calibration).
- Extremely small p-value indicates misspecification (e.g., missing covariates/history).

--------------------------------------------------------------------------------

## 3. Task performance metrics

Classification tasks (e.g., SHD/SSC)
- Accuracy: top-1; include confidence intervals (bootstrap).
- F1-score, macro/micro averaged for imbalanced classes.
- AUROC for binary tasks.

Temporal decision metrics
- Decision latency: time to first correct decision given a decision rule over time.
- Early exit accuracy vs latency curve.

Regression tasks
- MAE/MSE of continuous targets derived from spike statistics or intensity summaries.

Reporting
- Show mean ± std across seeds (≥3). Co-report the NLL and K–S p-values to assess calibration vs accuracy.

--------------------------------------------------------------------------------

## 4. SR readout efficiency and reliability

Definitions
- Acceptance ratio ρ: accepted / proposed in thinning.
- Bound violations: number of instances where S(t) > lam_max (tolerance-adjusted); should be near zero.

Computation
- Use [`Python.sample_ogata_with_stats()`](src/sr_ciden/readout.py:242). It returns:
  - "num_proposals"
  - "num_accepts"
  - "acceptance_ratio"
  - "bound_violations"

Heuristics
- Target ρ in 0.2–0.6 for research (trade-off between compute and exactness).
- Report both global and windowed acceptance ratios if dynamic lam_max is used.

Pseudocode
```
spikes, stats = sample_ogata_with_stats(intensity_fn, T=T, lam_max=Lmax, seed=seed)
rho = stats['acceptance_ratio']
viol = stats['bound_violations']
```

--------------------------------------------------------------------------------

## 5. Latency and energy proxies

Latency
- Algorithmic latency (host): mean inference time per event or per second.
- End-to-end latency (hardware): sensor-to-decision including transmission.

Energy proxies (when direct power is unavailable)
- Event volume: number of generated events (accepted spikes).
- Proposal volume: number of proposals in thinning (proxy for compute).
- MUL/ADD proxy: estimated flops per ODE step × number of steps.

Hardware energy (when measurable)
- Co-report average power × time; normalize by sequence length or events.

--------------------------------------------------------------------------------

## 6. Benchmarking protocols

Data splits and seeds
- Fixed train/val/test splits with published seeds; record any data preprocessing.

Solver configuration
- Publish [`Python.ODESolverConfig()`](src/sr_ciden/solvers.py:86) (method, rtol, atol, max_steps) or fallback dt; ensure reproducibility.

Hyperparameters
- Hidden dimension, nonlinearity, stable A flag, jump usage, intensity activation; learning rate and schedule; gradient clipping; number of epochs.

SR readout
- lam_max policy (global or windowed), headroom factor; discrete dt if using Bernoulli.

Validation checklist
- K–S p-value on validation/test.
- Acceptance ratio and bound violations.
- NLL per event and per time.

--------------------------------------------------------------------------------

## 7. Standardized reporting template

Include the following fields in a JSON/yaml artifact per run:

```
model:
  hidden_dim: H
  nonlinearity: softplus|tanh
  stable_A: true|false
  jump_update: true|false
training:
  optimizer: Adam
  lr: 1e-3
  epochs: 100
  grad_clip: 1.0
solver:
  backend: torchdiffeq|rk4|euler
  rtol: 1e-5
  atol: 1e-7
  dt: null  # if fallback used
sr_readout:
  lam_max_policy: global|windowed
  lam_max_value: 50.0
  dt_discrete: null
metrics:
  nll_per_event: ...
  nll_per_time: ...
  ks_p_value: ...
  acceptance_ratio: ...
  bound_violations: ...
  accuracy: ...
  latency_ms: ...
reproducibility:
  seed: 42
  torch_version: ...
  sr_ciden_commit: ...
```

--------------------------------------------------------------------------------

## 8. Comparative baselines

- Homogeneous Poisson with constant λ (per mark).
- Hawkes process (linear kernels) fitted by classical methods; compare NLL and K–S.
- Discrete-time GLM/Poisson models with basis functions; compare under the same binning.

--------------------------------------------------------------------------------

## 9. Visualization checklist

- Raster plots of observed vs sampled spikes.
- Intensity λ_k(t) trajectories with confidence bands (if Bayesian variants are used).
- Cumulative integral I(t) vs time.
- PIT histograms and QQ plots for Δτ.
- Acceptance ratio trace over time windows; bound violation markers.

--------------------------------------------------------------------------------

## 10. Pseudocode: full evaluation pass

```
# Given a trained model and held-out events
intensity_fn = model.get_intensity_fn(h_init=None)  # see Python.CIDENContinuous.get_intensity_fn()
# Likelihood-based metrics
loss = nll_continuous(f_h=model.ode_core, lambda_fn=model._lambda_fn, h0=model.h0,
                      events=heldout_events, T_end=T, K=model.num_marks, reduction='none')
nll_per_event = loss / num_events(heldout_events)
nll_per_time = loss / T

# Goodness-of-fit
ks = time_rescaling_test(lambda t: intensity_fn(float(t.item())), heldout_events)
# SR readout efficiency
spikes, stats = sample_ogata_with_stats(intensity_fn, T=T, lam_max=lam_max, seed=seed)

# Task metrics computed downstream on classifier head or rule
acc = compute_accuracy(spikes, labels)
```

See [`Python.CIDENContinuous.get_intensity_fn()`](src/sr_ciden/adapters/torch.py:191), [`Python.nll_continuous()`](src/sr_ciden/losses.py:152), [`Python.time_rescaling_test()`](src/sr_ciden/validation.py:169), and [`Python.sample_ogata_with_stats()`](src/sr_ciden/readout.py:242).

--------------------------------------------------------------------------------

## 11. Suggested reading

- Hawkes, A. G. (1971).
- Lewis, P. A. W., & Shedler, G. S. (1979).
- Ogata, Y. (1981).
- Brown, E. N., et al. (2002).
- Chen, T. Q., et al. (2018).

For full citations, see [docs/academy/REFERENCES.md](docs/academy/REFERENCES.md).