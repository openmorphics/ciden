#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
SEED="${SEED:-0}"

echo "[reproduce] Using python: $PYTHON_BIN"
echo "[reproduce] Seed: $SEED"

# Determinism env (optional ODE tolerances)
export SR_CIDEN_RTOL="${SR_CIDEN_RTOL:-1e-5}"
export SR_CIDEN_ATOL="${SR_CIDEN_ATOL:-1e-7}"

# Output directories
mkdir -p artifacts/examples artifacts/figures artifacts/profiles results

# 1) Minimal example (training entry + inference + KS test)
echo "[reproduce] Running minimal example..."
"$PYTHON_BIN" examples/minimal_training_loop.py

# 2) Benchmarks (memory/efficiency/speed) — CPU/GPU aware
echo "[reproduce] Running benchmarks..."
"$PYTHON_BIN" benchmarks/benchmark_memory_and_speed.py --bench all

# 3) Manifest
GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo "unknown")"
DATE_ISO="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

cat > artifacts/reproduce_manifest.json <<JSON
{
  "generated_at": "$DATE_ISO",
  "commit": "$GIT_SHA",
  "seed": $SEED,
  "artifacts": {
    "env": "artifacts/env.json",
    "example_metrics": "artifacts/examples/minimal_training_loop.json",
    "benchmarks": "results/benchmarks_all.json"
  }
}
JSON

echo "[reproduce] Done. See artifacts/reproduce_manifest.json, artifacts/examples/, artifacts/figures/, and results/."
