# Results JSON schema

All benchmark and experiment runs should emit JSON files matching:

{
  "task": "string",                // short task identifier, e.g., "benchmarks" or "ks_timescaling"
  "seed": 0,                       // integer random seed used
  "commit": "sha1",                // git commit sha if available
  "env": {                         // environment capture subset
    "platform": {...},
    "python": {...},
    "versions": {...},
    "torch": {...}
  },
  "metrics": {                     // task-specific metrics object
    ... task-dependent fields ...
  }
}

- File naming:
  - Benchmarks:
    - results/benchmarks_all.json       (aggregate for --bench all)
    - results/benchmark_memory.json     (memory-only)
    - results/benchmark_efficiency.json (thinning efficiency-only)
    - results/benchmark_speed.json      (throughput-only)
  - Examples/experiments:
    - artifacts/examples/minimal_training_loop.json

- Conventions:
  - Numerical metrics are floats (not strings)
  - Units must be documented in the top-level README or adjacent docs
  - Add a "skipped" field at the top level for tasks not applicable on current hardware (e.g., no CUDA)
