# 11 Hardware Deployment and Energy

Purpose
- Provide a practical blueprint to deploy C‑IDEN + SR readout in neuromorphic and conventional accelerators with an emphasis on energy efficiency and reproducibility.
- Summarize Loihi/Lava, SpiNNaker, and Brian2GeNN (GPU) pathways, their constraints, and end‑to‑end deployment flows.
- Offer checklists and profiling guidelines to measure latency, power, and scaling effects.

sr_ciden anchors
- Build inference intensity functions from trained models via [`Python.CIDENContinuous.get_intensity_fn()`](src/sr_ciden/adapters/torch.py:191); generate spikes continuously with [`Python.sample_ogata()`](src/sr_ciden/readout.py:152) or on a grid with [`Python.sample_bernoulli()`](src/sr_ciden/readout.py:183).
- Training, integrators, and NLL are device‑agnostic: [`Python.nll_continuous()`](src/sr_ciden/losses.py:152), [`Python.odeint_wrapper()`](src/sr_ciden/solvers.py:298), [`Python.odeint_augmented()`](src/sr_ciden/solvers.py:455).
- Validation metrics include time‑rescaling K–S and thinning efficiency: [`Python.time_rescaling_test()`](src/sr_ciden/validation.py:169), [`Python.sample_ogata_with_stats()`](src/sr_ciden/readout.py:242).

Inline references
- Nengo and Nengo Loihi (Bekolay et al., 2014), NEST (Gewaltig and Diesmann, 2007), Brian2/GeNN (Stimberg et al., 2019), Loihi/Lava documentation, SpiNNaker documentation.

--------------------------------------------------------------------------------

## 1. Deployment models and interface choices

Two primary deployment modes for C‑IDEN outputs:

- Event injection
  - Use SR readout to synthesize spike events with exact TPP semantics and feed them to target simulators or hardware interfaces. This preserves continuous‑time timing and sparsity.

- Rate/latency bridging
  - Convert intensities to per‑step rates (Poisson) or windows of latency codes and feed standard rate or spike generator nodes. This is useful when the target enforces time‑stepped integration or lacks an event API.

Trade‑offs
- Event injection is statistically faithful and often more energy‑efficient due to sparsity, but requires reliable bounds lam_max and deterministic scheduling on the target.
- Rate/latency bridges are simpler to integrate into existing toolchains but may increase spike counts or impose grid latency.

--------------------------------------------------------------------------------

## 2. Loihi and Lava (Intel)

Overview
- Loihi is a digital neuromorphic architecture with event‑driven cores, fixed‑point state, and dedicated learning engines for supported rules. Lava is the open‑source software framework for neuromorphic applications.

Constraints and considerations
- Precision: fixed‑point representation; quantize rates and state variables.
- Learning: on‑chip plasticity rules are constrained; C‑IDEN training occurs off‑chip, deploy only inference pathways.
- Timing: discrete tick scheduling; emulate continuous‑time arrivals by mapping event times to ticks.

Deployment flow (bridge via Nengo Loihi or Lava)
- Train in sr_ciden; export intensity_fn and generate either
  - event streams (preferable) with tick mapping; or
  - rate profiles over ticks.
- Build Nengo graphs (Nengo Loihi) or Lava Processes that receive spikes/rates from the host and drive Loihi inputs.
- Validate on recorded runs: compare acceptance ratios and output statistics to host SR runs.

Energy tips
- Exploit sparsity: tune lam_max per window (07) to maintain a moderate acceptance ratio and limit proposals.
- Use low‑precision encodings and compressed spike protocols where supported by the API.
- Batch transmissions to minimize host‑to‑device overhead.

--------------------------------------------------------------------------------

## 3. SpiNNaker

Overview
- Massively parallel ARM‑based neuromorphic platform targeting spiking networks with packet‑based spike routing.

Constraints and considerations
- Timing resolution: discrete time steps; event arrival is packetized.
- On‑chip learning: limited flexibility; prefer off‑chip training with sr_ciden and deploy spike trains or rates.

Deployment flow
- Generate SR spikes on host for windows of [t0, t1], quantize times to SpiNNaker ticks, and load as spike source arrays.
- For rate mode, generate piecewise constant Poisson generators synchronized to the simulation tick.
- Monitor packet drop counters and buffer capacities for high‑burst loads.

Energy tips
- Reduce peak burstiness by smoothing lam_max windows; consider short‑horizon prediction and streaming to even out traffic.
- Profile router activity; reduce fan‑out or route lengths via partitioning.

--------------------------------------------------------------------------------

## 4. Brian2GeNN and GPU acceleration

Overview
- Brian2 offers equation‑based modeling; Brian2GeNN compiles to GPU via GeNN for acceleration.

Constraints and considerations
- Time‑stepped integration; event‑driven effects via threshold/reset but the overall engine advances on a grid.
- Good for large batch simulations and parameter sweeps.

Deployment flow
- SR spikes generated offline feed into `SpikeGeneratorGroup`; or per‑bin rate profiles into `PoissonGroup`.
- For hybrid models, precompute intensity_fn on a coarse grid to reduce Python callback overhead and transfer costs.

Energy tips
- Use mixed precision (FP16/BF16) where numerically safe, especially for downstream readout layers not tied to intensity calculation.
- Profile kernel occupancy and memory bandwidth; prefer coalesced structures for spike buffers.

--------------------------------------------------------------------------------

## 5. Energy‑efficient inference practices

- Quantization
  - Quantize intensities and thresholds; for Bernoulli fallback clamp p to [0, 1 − 1e−12] (as in [`Python.sample_bernoulli()`](src/sr_ciden/readout.py:183)) to avoid pathological saturation.
  - Quantize marks and time indices to reduce communication overhead.

- Sparsity and duty cycle
  - Maintain acceptance ratio in a target range (e.g., 20–60%) by planning lam_max with per‑window headroom (07). High lam_max without need wastes proposals and energy.

- Event‑driven compute
  - Prefer continuous‑time thinning to avoid empty‑bin computations; if a grid is mandatory, use adaptive binning with variable dt in quiet vs active phases.

- Batching and streaming
  - Stream events in windows to amortize host‑device synchronization; prioritize deterministic chunk boundaries for reproducibility.

--------------------------------------------------------------------------------

## 6. Deployment checklists

Pre‑deployment
- [ ] Validate model fit on held‑out data with [`Python.time_rescaling_test()`](src/sr_ciden/validation.py:169) and report p‑values.
- [ ] Measure thinning stats with [`Python.sample_ogata_with_stats()`](src/sr_ciden/readout.py:242): acceptance_ratio, bound_violations.
- [ ] Select lam_max policy (global or per‑window) with headroom factor; lock random seeds and solver tolerances.

Target configuration
- [ ] Choose time resolution (tick) and quantization scheme; define mapping t → tick.
- [ ] Map channels/marks to input ports; verify routing and buffer capacities.
- [ ] Decide on event vs rate interface; prototype both on a short sequence.

Runbook
- [ ] Record software versions (driver, SDK), hardware identifiers, power sampling interval.
- [ ] Log acceptance ratios, tick utilization, packet drops or buffer overflows.
- [ ] Save raw event logs for replay.

--------------------------------------------------------------------------------

## 7. Profiling tools overview

- Host
  - CPU: perf or vtune for host profiling; track per‑window event rate and sampler cost.
  - GPU: Nsight Systems/Compute for kernel timing; nvidia‑smi for coarse power and utilization.

- Neuromorphic stacks
  - Loihi: Lava/Nengo Loihi instrumentation for spike counts, process timing; external power meters recommended.
  - SpiNNaker: router counters, dropped packets, core load; external power measurements where possible.

Reporting
- Always co‑report energy proxy metrics (spikes processed, proposals, acceptances) alongside task metrics. See 12 for standardized reporting.

--------------------------------------------------------------------------------

## 8. Mapping to notebooks and examples

Planned notebooks
- notebooks/academy/15_hardware_loihi_spinnaker.ipynb — end‑to‑end deployment with event bridging and tick quantization.
- notebooks/academy/16_energy_quantization_sparsity.ipynb — acceptance‑ratio tuning, quantization sweeps, and power proxy metrics.

Planned examples
- examples/frameworks/nest_nengo_adapter.py — rate and latency bridge stubs that can be adapted for Loihi/Nengo.
- examples/frameworks/brian2_ciden_rate_bridge.py — Brian2 input adapters.

Citations
- See [docs/academy/REFERENCES.md](docs/academy/REFERENCES.md) for canonical framework references and suggested reading on neuromorphic deployment.