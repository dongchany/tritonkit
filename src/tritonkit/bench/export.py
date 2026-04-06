"""Export benchmark results to JSON and CSV."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tritonkit.bench.result import BenchmarkResult


def export_json(result: BenchmarkResult, path: str) -> None:
    """Export BenchmarkResult as JSON."""
    data = {
        "version": result.software.get("tritonkit", "0.1.0"),
        "kernel": result.kernel,
        "hardware": result.hardware.to_dict(),
        "software": result.software,
        "mode": result.mode,
        "timestamp": result.timestamp,
        "results": [r.to_dict() for r in result.results],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results exported to {path}")


def export_csv(result: BenchmarkResult, path: str) -> None:
    """Export BenchmarkResult as CSV."""
    if not result.results:
        return

    fieldnames = ["implementation", "shape", "dtype", "median_us", "p20_us", "p80_us"]
    sample = result.results[0]
    if sample.tflops is not None:
        fieldnames.append("tflops")
    if sample.throughput_gbps is not None:
        fieldnames.append("throughput_gbps")

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in result.results:
            row = r.to_dict()
            row["shape"] = "x".join(str(s) for s in r.shape)
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"Results exported to {path}")
