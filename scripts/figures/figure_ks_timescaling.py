#!/usr/bin/env python3
"""
Generate KS time-rescaling figure from results/benchmark_ks_timescaling.json.

Outputs:
- artifacts/figures/ks_timescaling.png
- artifacts/figures/ks_timescaling.svg
- artifacts/figures/ks_timescaling.csv
"""
from __future__ import annotations

import csv
import json
import os
import sys

import matplotlib.pyplot as plt


def main() -> None:
    src = "results/benchmark_ks_timescaling.json"
    if not os.path.exists(src):
        print(f"Missing input: {src}", file=sys.stderr)
        sys.exit(1)

    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    met = data.get("metrics", {})
    rows = [("poisson_p", float(met.get("poisson_p", float("nan")))),
            ("sinusoid_p", float(met.get("sinusoid_p", float("nan"))))]

    os.makedirs("artifacts/figures", exist_ok=True)
    # CSV
    with open("artifacts/figures/ks_timescaling.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["series", "p_value"])
        for k, v in rows:
            w.writerow([k, v])

    # Plot
    labels = [k for k, _ in rows]
    values = [v for _, v in rows]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(labels, values, color=["#4a7c59", "#4169e1"])
    ax.set_ylabel("K–S p-value")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("artifacts/figures/ks_timescaling.png", dpi=200)
    fig.savefig("artifacts/figures/ks_timescaling.svg")
    print("Wrote artifacts/figures/ks_timescaling.(png,svg,csv)")


if __name__ == "__main__":
    main()
