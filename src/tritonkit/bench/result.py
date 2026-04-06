"""Benchmark result data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import torch

from tritonkit.bench.hardware import HardwareFingerprint


def _get_software_versions() -> dict:
    import triton
    import sys

    cuda_version = "unknown"
    if torch.cuda.is_available():
        cuda_version = ".".join(str(x) for x in torch.version.cuda.split(".")) if torch.version.cuda else "unknown"

    return {
        "triton": triton.__version__,
        "cuda": cuda_version,
        "pytorch": torch.__version__,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "tritonkit": "0.1.0",
    }


@dataclass
class SingleResult:
    name: str
    shape: tuple[int, ...]
    dtype: str
    median_us: float
    p20_us: float
    p80_us: float
    flops: int | None = None
    bytes_accessed: int | None = None

    @property
    def tflops(self) -> float | None:
        if self.flops is None:
            return None
        return self.flops / (self.median_us * 1e-6) / 1e12

    @property
    def throughput_gbps(self) -> float | None:
        if self.bytes_accessed is None:
            return None
        return self.bytes_accessed / (self.median_us * 1e-6) / 1e9

    def to_dict(self) -> dict:
        d = {
            "implementation": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "median_us": round(self.median_us, 2),
            "p20_us": round(self.p20_us, 2),
            "p80_us": round(self.p80_us, 2),
        }
        if self.tflops is not None:
            d["tflops"] = round(self.tflops, 2)
        if self.throughput_gbps is not None:
            d["throughput_gbps"] = round(self.throughput_gbps, 2)
        return d


@dataclass
class BenchmarkResult:
    kernel: str
    results: list[SingleResult]
    hardware: HardwareFingerprint
    mode: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    software: dict = field(default_factory=_get_software_versions)

    def print_table(self, sort_by: str = "median_us") -> None:
        """Print formatted comparison table to stdout."""
        from tabulate import tabulate

        rows = []
        for r in sorted(self.results, key=lambda x: getattr(x, sort_by, 0)):
            row = [r.name, "x".join(str(s) for s in r.shape), r.dtype, f"{r.median_us:.1f}"]
            if r.tflops is not None:
                row.append(f"{r.tflops:.2f}")
            if r.throughput_gbps is not None:
                row.append(f"{r.throughput_gbps:.1f}")
            rows.append(row)

        headers = ["Implementation", "Shape", "Dtype", "Median (us)"]
        if any(r.tflops is not None for r in self.results):
            headers.append("TFLOPS")
        if any(r.throughput_gbps is not None for r in self.results):
            headers.append("GB/s")

        print(f"\n=== {self.kernel} benchmark ({self.mode} mode) ===")
        print(f"GPU: {self.hardware.gpu_name}")
        print(tabulate(rows, headers=headers, tablefmt="simple"))

    def plot_roofline(self, save: str | None = None) -> None:
        """Generate roofline plot. Requires matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required: pip install tritonkit[plot]")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)")
        ax.set_ylabel("Performance (TFLOPS)")
        ax.set_title(f"Roofline: {self.kernel} on {self.hardware.gpu_name}")

        for r in self.results:
            if r.tflops is not None and r.bytes_accessed is not None and r.flops is not None:
                ai = r.flops / r.bytes_accessed
                ax.scatter(ai, r.tflops, label=f"{r.name} {'x'.join(str(s) for s in r.shape)}", s=50)

        ax.legend(fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")

        if save:
            fig.savefig(save, dpi=150, bbox_inches="tight")
            print(f"Roofline plot saved to {save}")
        else:
            plt.show()

    def export_json(self, path: str) -> None:
        """Export results as JSON."""
        from tritonkit.bench.export import export_json
        export_json(self, path)

    def export_csv(self, path: str) -> None:
        """Export results as CSV."""
        from tritonkit.bench.export import export_csv
        export_csv(self, path)

    def filter(
        self,
        names: list[str] | None = None,
        shapes: list[tuple[int, ...]] | None = None,
    ) -> BenchmarkResult:
        """Return filtered copy of results."""
        filtered = self.results
        if names is not None:
            filtered = [r for r in filtered if r.name in names]
        if shapes is not None:
            filtered = [r for r in filtered if r.shape in shapes]
        return BenchmarkResult(
            kernel=self.kernel,
            results=filtered,
            hardware=self.hardware,
            mode=self.mode,
            timestamp=self.timestamp,
            software=self.software,
        )
