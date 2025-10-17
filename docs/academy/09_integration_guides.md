# 09 Integration Guides: Framework-Agnostic Patterns

Purpose
- Provide high-level, framework-agnostic integration templates to connect a trained C‑IDEN intensity model to simulator ecosystems.
- Emphasize training–inference decoupling: training lives in sr_ciden; simulators consume intensities or sampled spikes for downstream inference and control.
- Cover patterns for snnTorch, Brian2, NEST, Nengo, and SpykeTorch with clear mapping to forthcoming notebooks and example scripts.

Core sr_ciden anchors
- Training objective and augmented ODE integration: [`Python.nll_continuous()`](src/sr_ciden/losses.py:152), [`Python.odeint_augmented()`](src/sr_ciden/solvers.py:455), [`Python.odeint_wrapper()`](src/sr_ciden/solvers.py:298).
- Latent field, stable generator, jump updates, and intensity head: [`Python.LinearNonlinearODE()`](src/sr_ciden/dynamics.py:191), [`Python.StableMatrixParam()`](src/sr_ciden/dynamics.py:126), [`Python.JumpMLP()`](src/sr_ciden/dynamics.py:282), [`Python.IntensityHead()`](src/sr_ciden/dynamics.py:346).
- Inference-only readout samplers: [`Python.sample_ogata()`](src/sr_ciden/readout.py:152), [`Python.sample_ogata_with_stats()`](src/sr_ciden/readout.py:242), [`Python.sample_bernoulli()`](src/sr_ciden/readout.py:183).
- PyTorch adapter for training loops and intensity function construction: [`Python.CIDENContinuous()`](src/sr_ciden/adapters/torch.py:36), [`Python.CIDENContinuous.get_intensity_fn()`](src/sr_ciden/adapters/torch.py:191).

--------------------------------------------------------------------------------

## A. Global integration principles

A.1 Keep training in sr_ciden
- Train the intensity λ(h, t) end-to-end using exact marked-TPP likelihood. Persist model parameters and solver configuration for reproducibility.

A.2 Export an intensity or spike stream
- Option 1: Intensity function intensity_fn(t) → ℝ^K via [`Python.CIDENContinuous.get_intensity_fn()`](src/sr_ciden/adapters/torch.py:191), then generate spikes with SR readout samplers for simulators that accept event inputs.
- Option 2: Convert intensities to rates/latencies on a grid for frameworks with rate interfaces or mandatory time steps.

A.3 Validate before interop
- Apply time rescaling and one-sample K–S (Brown et al., 2002) via [`Python.time_rescaling_test()`](src/sr_ciden/validation.py:169) to ensure statistical consistency, then integrate with the simulator.

A.4 Determinism and boundaries
- For thinning, choose lam_max conservatively and monitor acceptance ratio and bound violations with [`Python.sample_ogata_with_stats()`](src/sr_ciden/readout.py:242). For discrete grids, calibrate dt to keep p = 1 − exp(−λ dt) ≪ 1.

--------------------------------------------------------------------------------

## B. snnTorch (PyTorch) patterns

B.1 Spikes as exogenous inputs
- Use SR readout to produce spike trains per mark k and feed as inputs to snnTorch layers.

Pseudocode
```
# Train in sr_ciden (omitted); then inference:
intensity_fn = model.get_intensity_fn(h_init=None)   # 1D K-vector per query time
spikes, stats = sample_ogata_with_stats(intensity_fn, T=T, lam_max=Lmax, seed=seed)

# Convert event list to a dense binary tensor [T_bins, B=1, K]
S = events_to_tensor(spikes, T, dt)  # set 1 at bin floor(t/dt), channel k

# Feed into snnTorch network
for t in range(T_bins):
    out = net(S[t])      # standard snnTorch forward
    # accumulate readout and metrics as needed
```

B.2 Discrete-time rate interface
- Compute p_k(t) = 1 − exp(−λ_k(t) dt) per bin using intensity_fn at bin times; sample using [`Python.SRReadoutTorch()`](src/sr_ciden/adapters/torch.py:260) for on-the-fly Bernoulli spikes.

Forthcoming assets
- notebooks/academy/08_frameworks_snntorch_integration.ipynb
- examples/frameworks/snntorch_infer.py

--------------------------------------------------------------------------------

## C. Brian2 patterns

C.1 Event injection (preferred for precision)
- Generate spikes with SR readout and inject them as presynaptic events to Brian2 Synapses/SpikeGeneratorGroup.

Pseudocode
```
# Offline in Python: generate spikes = [(t_i, k_i)]
# In Brian2:
from brian2 import *
events = array(spikes)  # convert to (indices, times) aligned with Brian2 indexing
G = SpikeGeneratorGroup(N=K, indices=events[:,1], times=events[:,0]*second)
# Connect G into your network; run the simulation for duration T
```

C.2 Rate-driven inputs
- Convert λ_k(t) into time-varying rates for PoissonGroup or custom current injections; ensure dt aligns with Brian2 timestep to avoid drift.

C.3 Hybrid latent ODE alignment
- If modeling hybrid systems where Brian2 state depends on sr_ciden latents, synchronize via a shared time grid or windowed callbacks. Prefer ogata-based spikes for strict event timing.

Forthcoming assets
- notebooks/academy/09_brian2_integration.ipynb
- examples/frameworks/brian2_ciden_rate_bridge.py

--------------------------------------------------------------------------------

## D. NEST patterns

D.1 SR spikes into NEST
- Use [`Python.sample_ogata()`](src/sr_ciden/readout.py:152) to produce spike times and feed them via spike_generator devices.

Pseudocode
```
# spikes: list of (t, k) with k in [0..K-1]
nest.ResetKernel()
gens = []
for k in range(K):
    ts = [t for (t, m) in spikes if m == k]
    g = nest.Create('spike_generator', params={'spike_times': ts})
    gens.append(g)
# Connect gens[k] to target neurons/synapses as desired
# nest.Simulate(T)
```

D.2 Rate interface
- For rate-based stimulation, instantiate poisson_generator with time-varying rates sampled from λ_k(t). Batch via piecewise-constant windows per NEST’s API.

Considerations
- Ensure simulator resolution resolves the minimum inter-spike interval implied by λ; reconcile units (ms vs s).

Forthcoming assets
- notebooks/academy/10_nest_nengo_bridge.ipynb
- examples/frameworks/nest_nengo_adapter.py

--------------------------------------------------------------------------------

## E. Nengo patterns

E.1 Rate bridge
- Convert intensities to instantaneous rates for Nengo Nodes/Ensembles driving spiking backends. For NEF pipelines, λ can be used as a rate signal; consider saturating nonlinearities.

E.2 Latency bridge
- For latency coding, emit one spike per window with time-to-first-spike computed from λ-derived features. Provide marks as channel indices to Nengo’s spiking ensembles.

Synchronization
- Choose a fixed dt and ensure intensity_fn queries align with Nengo’s simulator step. Cache λ on a grid to avoid Python callback overhead.

Forthcoming assets
- notebooks/academy/10_nest_nengo_bridge.ipynb
- examples/frameworks/nest_nengo_adapter.py

--------------------------------------------------------------------------------

## F. SpykeTorch patterns

F.1 Discrete spike tensors
- Produce Bernoulli samples or rank-order spikes per receptive field using intensities on a grid and convert to sparse spike tensors expected by SpykeTorch.

Pseudocode
```
# Build [T_bins, K] Bernoulli spikes from intensity_fn
for t in bins:
    lam = intensity_fn(t)           # [K]
    p = 1 - exp(-lam * dt)
    s = Bernoulli(p).sample()
    S[t] = s
# Convert S to SpykeTorch sparse format and feed to convolutional SNN blocks
```

Forthcoming assets
- notebooks/academy/11_spyketorch_discrete_integration.ipynb
- examples/frameworks/spyketorch_discrete_readout.py

--------------------------------------------------------------------------------

## G. Framework-agnostic adapters and templates

G.1 Intensity-to-event adapter
```
def intensity_to_spikes(intensity_fn, T, method='ogata', dt=None, lam_max=None, seed=0):
    if method == 'ogata':
        assert lam_max is not None
        return sample_ogata(intensity_fn, T=T, lam_max=lam_max, seed=seed)
    elif method == 'bernoulli':
        assert dt is not None
        return sample_bernoulli(intensity_fn, T=T, dt=dt, seed=seed)
    else:
        raise ValueError('method must be ogata or bernoulli')
```

G.2 Event list to dense tensor
```
def events_to_tensor(spikes, T, dt, K):
    T_bins = int(T / dt)
    S = zeros((T_bins, 1, K), dtype=int)
    for (t, k) in spikes:
        i = min(int(t // dt), T_bins-1)
        S[i, 0, k] = 1
    return S
```

G.3 Windowed lam_max planning
- Estimate S(t) on a coarse grid, then assign lam_max per window with headroom (see 07). Cache S(t) for introspection and acceptance ratio predictions prior to sampling.

--------------------------------------------------------------------------------

## H. Reproducibility and evaluation hooks

- Record seeds, solver tolerances, and lam_max choices alongside simulator configs. Persist the event list used for the simulator run.
- Validate with time rescaling (pre-interop), then report task metrics (post-interop). See evaluation metrics in 12 and reproducibility guidance in 13.

--------------------------------------------------------------------------------

## I. Mapping to notebooks and examples

Planned notebooks (not included in this step)
- notebooks/academy/08_frameworks_snntorch_integration.ipynb — PyTorch interop and small SNN pipelines.
- notebooks/academy/09_brian2_integration.ipynb — event injection and state alignment.
- notebooks/academy/10_nest_nengo_bridge.ipynb — rate and latency interfaces; mixed-simulator determinism.
- notebooks/academy/11_spyketorch_discrete_integration.ipynb — tensor construction and DVS preprocessing.

Planned example scripts
- examples/frameworks/snntorch_infer.py — snnTorch inference driven by SR spikes.
- examples/frameworks/brian2_ciden_rate_bridge.py — Brian2 rate bridge for λ-driven nodes.
- examples/frameworks/nest_nengo_adapter.py — minimal adapter for NEST and Nengo interfaces.
- examples/frameworks/spyketorch_discrete_readout.py — discrete SR pipeline for SpykeTorch.

Citations
- See [docs/academy/REFERENCES.md](docs/academy/REFERENCES.md) for canonical framework references.