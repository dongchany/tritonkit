"""Roofline model: measure actual peak bandwidth and FLOPS."""

from __future__ import annotations

import torch
import triton


def measure_peak_bandwidth(size_bytes: int = 4 * 1024**3) -> float:
    """Measure actual DRAM bandwidth via buffer zeroing.

    Args:
        size_bytes: Buffer size in bytes (default 4 GB).

    Returns:
        Bandwidth in GB/s.
    """
    n_elements = size_bytes // 4
    buf = torch.empty(n_elements, dtype=torch.int32, device="cuda")

    def fn():
        buf.zero_()

    ms = triton.testing.do_bench(fn, warmup=50, rep=200)
    return size_bytes / (ms * 1e-3) / 1e9


def measure_peak_flops(dtype: torch.dtype = torch.float16, M: int = 8192) -> float:
    """Measure actual peak TFLOPS via cuBLAS GEMM.

    Args:
        dtype: Data type for matmul.
        M: Matrix dimension (M x M x M).

    Returns:
        Peak TFLOPS.
    """
    a = torch.randn(M, M, dtype=dtype, device="cuda")
    b = torch.randn(M, M, dtype=dtype, device="cuda")

    def fn():
        torch.mm(a, b)

    ms = triton.testing.do_bench(fn, warmup=50, rep=200)
    flops = 2 * M * M * M
    return flops / (ms * 1e-3) / 1e12
