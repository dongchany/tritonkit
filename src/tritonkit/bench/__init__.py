"""Unified benchmark framework with hardware-aware measurement."""

from tritonkit.bench.result import BenchmarkResult, SingleResult
from tritonkit.bench.hardware import HardwareFingerprint, detect_hardware
from tritonkit.bench.runner import compare, run_single
from tritonkit.bench.roofline import measure_peak_bandwidth, measure_peak_flops
from tritonkit.bench.export import export_json, export_csv

__all__ = [
    "compare",
    "run_single",
    "BenchmarkResult",
    "SingleResult",
    "HardwareFingerprint",
    "detect_hardware",
    "measure_peak_bandwidth",
    "measure_peak_flops",
    "export_json",
    "export_csv",
]
