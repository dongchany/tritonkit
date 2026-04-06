"""Core benchmark runner."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
import triton

from tritonkit.bench.hardware import HardwareFingerprint, detect_hardware
from tritonkit.bench.result import BenchmarkResult, SingleResult
from tritonkit.testing.shapes import ShapePreset


def run_single(
    fn: Callable,
    inputs: tuple[torch.Tensor, ...],
    warmup_ms: float = 25.0,
    rep_ms: float = 100.0,
    quantiles: list[float] | None = None,
    flush_l2: bool = True,
) -> tuple[float, float, float]:
    """Benchmark a single function call.

    Returns:
        (median_ms, p20_ms, p80_ms) latencies in milliseconds.
    """
    if quantiles is None:
        quantiles = [0.5, 0.2, 0.8]

    cache = None
    if flush_l2:
        cache = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")

    # Warm up (JIT compile)
    fn(*inputs)
    torch.cuda.synchronize()

    def bench_fn():
        if cache is not None:
            cache.zero_()
        return fn(*inputs)

    results = triton.testing.do_bench(
        bench_fn,
        warmup=int(warmup_ms),
        rep=int(rep_ms),
        quantiles=quantiles,
    )

    if isinstance(results, (list, tuple)):
        return tuple(results[:3])
    return (results, results, results)


def compare(
    candidates: dict[str, Callable],
    shapes: list[tuple[int, ...]] | ShapePreset,
    dtypes: list[torch.dtype] | None = None,
    kernel_name: str = "unknown",
    mode: Literal["default", "best"] = "default",
    warmup_ms: float = 25.0,
    rep_ms: float = 100.0,
    quantiles: list[float] | None = None,
    input_generator: Callable | None = None,
    flop_counter: Callable | None = None,
    byte_counter: Callable | None = None,
) -> BenchmarkResult:
    """Compare multiple kernel implementations.

    Args:
        candidates: Dict mapping name -> callable.
        shapes: Input shapes to benchmark.
        dtypes: Data types to benchmark.
        kernel_name: Name of the kernel being benchmarked.
        mode: "default" or "best".
        warmup_ms: Warm-up time budget.
        rep_ms: Measurement time budget.
        quantiles: Quantiles to report.
        input_generator: Custom input generator.
            Signature: (shape, dtype, device) -> tuple[Tensor, ...]
        flop_counter: Signature: (shape, dtype) -> int
        byte_counter: Signature: (shape, dtype) -> int

    Returns:
        BenchmarkResult with all measurements.
    """
    if dtypes is None:
        dtypes = [torch.float16]
    if quantiles is None:
        quantiles = [0.5, 0.2, 0.8]

    hw = detect_hardware()
    all_results: list[SingleResult] = []

    for shape in shapes:
        for dtype in dtypes:
            if input_generator is not None:
                inputs = input_generator(shape, dtype, "cuda")
            else:
                inputs = (torch.randn(shape, dtype=dtype, device="cuda"),)

            flops = flop_counter(shape, dtype) if flop_counter else None
            nbytes = byte_counter(shape, dtype) if byte_counter else None
            dtype_str = str(dtype).replace("torch.", "")

            for name, fn in candidates.items():
                try:
                    med_ms, p20_ms, p80_ms = run_single(
                        fn, inputs,
                        warmup_ms=warmup_ms,
                        rep_ms=rep_ms,
                        quantiles=quantiles,
                    )
                    all_results.append(SingleResult(
                        name=name,
                        shape=shape,
                        dtype=dtype_str,
                        median_us=med_ms * 1000,
                        p20_us=p20_ms * 1000,
                        p80_us=p80_ms * 1000,
                        flops=flops,
                        bytes_accessed=nbytes,
                    ))
                except Exception as e:
                    print(f"  [SKIP] {name} on {shape} {dtype}: {e}")

    return BenchmarkResult(
        kernel=kernel_name,
        results=all_results,
        hardware=hw,
        mode=mode,
    )
