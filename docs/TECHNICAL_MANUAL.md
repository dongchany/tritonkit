# TritonKit — Technical Manual

> **Status**: Draft v0.1
> **Author**: Dongcheng Ye
> **Date**: 2026-04-07
> **Companion**: [Product Manual](PRODUCT_MANUAL.md)

---

## 1. Repository Structure

```
tritonkit/
├── pyproject.toml                 # Build config, dependencies, entry points
├── LICENSE                        # Apache 2.0
├── README.md
├── docs/
│   ├── PRODUCT_MANUAL.md
│   ├── PRODUCT_MANUAL_zh.md
│   ├── TECHNICAL_MANUAL.md
│   └── TECHNICAL_MANUAL_zh.md
├── src/
│   └── tritonkit/
│       ├── __init__.py            # Version, top-level exports
│       ├── _version.py            # __version__ = "0.1.0"
│       ├── primitives/
│       │   ├── __init__.py        # Re-export all primitives
│       │   ├── load_store.py      # masked_load, masked_store
│       │   ├── reduction.py       # tiled_reduce, warp_reduce
│       │   ├── softmax.py         # online_softmax
│       │   ├── stats.py           # online_mean_var
│       │   ├── gemm_utils.py      # split_k_accumulate, swizzle_tile
│       │   └── quantize.py        # block_quantize
│       ├── testing/
│       │   ├── __init__.py        # Re-export public API
│       │   ├── correctness.py     # assert_matches, assert_close
│       │   ├── numerical.py       # check_numerical_stability
│       │   ├── hardware.py        # run_on_available_backends
│       │   └── shapes.py          # STANDARD_SHAPES, EDGE_SHAPES, LLM_SHAPES
│       ├── bench/
│       │   ├── __init__.py        # Re-export public API
│       │   ├── runner.py          # compare(), run_single()
│       │   ├── result.py          # BenchmarkResult dataclass
│       │   ├── hardware.py        # HardwareFingerprint, detect_hardware()
│       │   ├── baselines.py       # Built-in baseline wrappers (cuBLAS, FA, Liger, FlagGems)
│       │   ├── roofline.py        # measure_peak_bandwidth(), measure_peak_flops(), plot_roofline()
│       │   └── export.py          # export_json(), export_csv()
│       ├── integrate/
│       │   ├── __init__.py
│       │   ├── torch_op.py        # register_torch_op()
│       │   ├── compile.py         # make_compilable()
│       │   └── autotune.py        # export_autotune_config()
│       ├── examples/
│       │   ├── __init__.py
│       │   ├── gemm_fp16.py       # Tiled GEMM with software pipelining
│       │   ├── rmsnorm_fused.py   # Fused RMSNorm + residual
│       │   ├── flash_attention.py # Flash Attention v2
│       │   ├── swiglu_fused.py    # Fused SwiGLU activation
│       │   └── int8_gemm.py       # INT8 quantized GEMM
│       └── cli/
│           ├── __init__.py
│           └── main.py            # CLI entry point (tritonkit bench, tritonkit test)
├── tests/
│   ├── conftest.py                # pytest fixtures (GPU availability, etc.)
│   ├── test_testing.py            # Tests for the testing framework
│   ├── test_bench.py              # Tests for the benchmark framework
│   ├── test_primitives.py         # Tests for each primitive
│   ├── test_integrate.py          # Tests for integration tools
│   └── test_examples.py           # Tests for example kernels
└── benchmarks/
    ├── run_all.py                 # Run full benchmark suite
    ├── suites/
    │   ├── gemm.py                # GEMM benchmark suite
    │   ├── attention.py           # Attention benchmark suite
    │   ├── norm.py                # Normalization benchmark suite
    │   ├── activation.py          # Activation benchmark suite
    │   ├── softmax.py             # Softmax benchmark suite
    │   └── moe.py                 # MoE benchmark suite
    └── results/                   # Generated benchmark results (gitignored)
        └── .gitkeep
```

---

## 2. Module Specifications

### 2.1 `tritonkit.testing`

Unified correctness testing for Triton kernels.

#### Public API

```python
# tritonkit/testing/__init__.py

from tritonkit.testing.correctness import assert_matches, assert_close
from tritonkit.testing.numerical import check_numerical_stability
from tritonkit.testing.hardware import run_on_available_backends
from tritonkit.testing.shapes import (
    STANDARD_SHAPES,
    EDGE_SHAPES,
    LLM_SHAPES,
    ShapePreset,
)
```

#### `assert_matches`

```python
def assert_matches(
    triton_fn: Callable,
    reference_fn: Callable,
    shapes: list[tuple[int, ...]] | ShapePreset,
    dtypes: list[torch.dtype] = [torch.float16, torch.bfloat16],
    atol: float = 1e-2,
    rtol: float = 1e-2,
    input_generator: Callable | None = None,
    device: str = "cuda",
) -> None:
    """Test that triton_fn produces the same output as reference_fn.

    Args:
        triton_fn: The Triton kernel wrapper. Signature: (input_tensors...) -> Tensor.
        reference_fn: PyTorch reference. Same signature as triton_fn.
        shapes: List of input shapes to test, or a ShapePreset.
        dtypes: List of dtypes to test.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        input_generator: Custom input generator. If None, uses torch.randn.
            Signature: (shape, dtype, device) -> tuple[Tensor, ...].
        device: Device to run on.

    Raises:
        AssertionError: With detailed mismatch report including:
            - Shape and dtype that failed
            - Max absolute error
            - Max relative error
            - Index of worst element
            - Percentage of mismatched elements
    """
```

**Internal design**:
1. For each `(shape, dtype)` combination:
   - Generate inputs via `input_generator` or `torch.randn(shape, dtype=dtype, device=device)`
   - Call `reference_fn(*inputs)` → `ref_out`
   - Call `triton_fn(*inputs)` → `tri_out`
   - `torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=rtol)`
2. On failure, catch `AssertionError`, enrich with diagnostics, re-raise.

#### `check_numerical_stability`

```python
def check_numerical_stability(
    fn: Callable,
    shapes: list[tuple[int, ...]] | ShapePreset = STANDARD_SHAPES,
    dtypes: list[torch.dtype] = [torch.float16, torch.bfloat16],
    input_range: tuple[float, float] = (-100.0, 100.0),
    check_nan: bool = True,
    check_inf: bool = True,
    device: str = "cuda",
) -> None:
    """Check that fn does not produce NaN or Inf for inputs in input_range.

    Generates random inputs uniformly in [input_range[0], input_range[1]],
    calls fn, and asserts no NaN/Inf in output.
    """
```

#### `run_on_available_backends`

```python
def run_on_available_backends(
    test_fn: Callable[[], None],
) -> dict[str, bool | str]:
    """Run test_fn on all available GPU backends.

    Detects available backends (CUDA, ROCm, XPU) and runs test_fn on each.
    Returns dict mapping backend name to True (passed) or error string.
    """
```

#### Shape Presets

```python
# tritonkit/testing/shapes.py

from typing import TypeAlias

ShapePreset: TypeAlias = list[tuple[int, ...]]

# Common LLM dimensions
STANDARD_SHAPES: ShapePreset = [
    (128,), (256,), (512,), (1024,), (2048,), (4096,), (8192,),
    (128, 128), (256, 256), (512, 512), (1024, 1024),
    (4096, 4096), (8192, 8192),
]

# Non-power-of-2, primes, edge cases
EDGE_SHAPES: ShapePreset = [
    (1,), (3,), (7,), (17,), (127,), (255,), (1000,),
    (1, 1), (1, 4096), (4096, 1), (13, 17),
    (33, 65), (1023, 1025),
]

# Typical Llama / GPT linear layer shapes (M, K) or (M, N, K)
LLM_SHAPES: ShapePreset = [
    (1, 4096),          # single-token decode
    (32, 4096),         # small batch decode
    (128, 4096),        # medium batch
    (512, 4096),        # prefill chunk
    (2048, 4096),       # long prefill
    (4096, 4096),       # square
    (4096, 11008),      # Llama MLP up-projection
    (4096, 14336),      # Llama-2 70B MLP
    (11008, 4096),      # Llama MLP down-projection
]
```

---

### 2.2 `tritonkit.bench`

Unified benchmark framework with hardware-aware measurement.

#### Public API

```python
# tritonkit/bench/__init__.py

from tritonkit.bench.runner import compare, run_single
from tritonkit.bench.result import BenchmarkResult
from tritonkit.bench.hardware import HardwareFingerprint, detect_hardware
from tritonkit.bench.roofline import (
    measure_peak_bandwidth,
    measure_peak_flops,
    plot_roofline,
)
from tritonkit.bench.export import export_json, export_csv
```

#### `compare`

```python
def compare(
    candidates: dict[str, Callable],
    shapes: list[tuple[int, ...]] | ShapePreset,
    dtypes: list[torch.dtype] = [torch.float16],
    metrics: list[str] = ["latency_us", "throughput_gbps", "tflops"],
    mode: Literal["default", "best"] = "default",
    warmup_ms: float = 25.0,
    rep_ms: float = 100.0,
    quantiles: list[float] = [0.5, 0.2, 0.8],
    input_generator: Callable | None = None,
    flop_counter: Callable | None = None,
    byte_counter: Callable | None = None,
) -> BenchmarkResult:
    """Compare multiple kernel implementations.

    Args:
        candidates: Dict mapping name -> callable.
            Each callable: (*input_tensors) -> Tensor.
        shapes: Input shapes to benchmark.
        dtypes: Data types to benchmark.
        metrics: Which metrics to compute. Options:
            - "latency_us": Median latency in microseconds
            - "throughput_gbps": Memory throughput in GB/s
            - "tflops": Compute throughput in TFLOPS
            - "roofline_efficiency": Fraction of hardware peak
        mode: "default" uses each implementation's default config.
            "best" runs autotuning first (if available).
        warmup_ms: Warm-up time budget.
        rep_ms: Measurement time budget.
        quantiles: Quantiles to report (default: median, p20, p80).
        input_generator: Custom input generator.
            Signature: (shape, dtype, device) -> tuple[Tensor, ...]
        flop_counter: Custom FLOP counter.
            Signature: (shape, dtype) -> int
        byte_counter: Custom byte counter.
            Signature: (shape, dtype) -> int

    Returns:
        BenchmarkResult containing all measurements.
    """
```

**Internal design**:

```python
# Pseudocode for compare()
def compare(candidates, shapes, dtypes, ...):
    hw = detect_hardware()
    results = []

    for shape in shapes:
        for dtype in dtypes:
            inputs = input_generator(shape, dtype, "cuda")

            # L2 cache flush buffer (256 MB)
            cache = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")

            for name, fn in candidates.items():
                # Warm up (JIT compile)
                fn(*inputs)
                torch.cuda.synchronize()

                # Measure using triton.testing.do_bench
                def bench_fn():
                    cache.zero_()  # L2 flush
                    return fn(*inputs)

                ms, min_ms, max_ms = triton.testing.do_bench(
                    bench_fn,
                    warmup=int(warmup_ms),
                    rep=int(rep_ms),
                    quantiles=quantiles,
                )

                results.append(SingleResult(
                    name=name, shape=shape, dtype=dtype,
                    median_us=ms * 1000,
                    p20_us=min_ms * 1000,
                    p80_us=max_ms * 1000,
                    flops=flop_counter(shape, dtype) if flop_counter else None,
                    bytes=byte_counter(shape, dtype) if byte_counter else None,
                ))

    return BenchmarkResult(results=results, hardware=hw, mode=mode)
```

#### `BenchmarkResult`

```python
@dataclass
class SingleResult:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    median_us: float
    p20_us: float
    p80_us: float
    flops: int | None = None       # Total FLOPs for this operation
    bytes: int | None = None       # Total bytes read + written

    @property
    def tflops(self) -> float | None:
        if self.flops is None:
            return None
        return self.flops / (self.median_us * 1e-6) / 1e12

    @property
    def throughput_gbps(self) -> float | None:
        if self.bytes is None:
            return None
        return self.bytes / (self.median_us * 1e-6) / 1e9


@dataclass
class BenchmarkResult:
    results: list[SingleResult]
    hardware: HardwareFingerprint
    mode: str                       # "default" or "best"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    software: dict = field(default_factory=_get_software_versions)

    def print_table(self, sort_by: str = "median_us") -> None:
        """Print formatted comparison table to stdout."""

    def plot_roofline(self, save: str | None = None) -> None:
        """Generate roofline plot. Requires matplotlib."""

    def export_json(self, path: str) -> None:
        """Export results as JSON (schema in Section 3)."""

    def export_csv(self, path: str) -> None:
        """Export results as CSV."""

    def filter(
        self,
        names: list[str] | None = None,
        shapes: list[tuple[int, ...]] | None = None,
        dtypes: list[torch.dtype] | None = None,
    ) -> "BenchmarkResult":
        """Return filtered copy of results."""
```

#### `HardwareFingerprint`

```python
@dataclass
class HardwareFingerprint:
    gpu_name: str                  # "NVIDIA GeForce RTX 3080 Ti"
    compute_capability: str        # "8.6"
    driver_version: str            # "570.86.16"
    memory_gb: float               # 12.0
    sm_count: int                  # 80
    sm_clock_mhz: int             # 1665 (locked or boost)
    mem_clock_mhz: int            # 9501
    pcie_gen: str                  # "4.0"
    peak_bandwidth_gbps: float    # measured, not spec
    peak_tflops_fp16: float       # measured, not spec

    @staticmethod
    def detect() -> "HardwareFingerprint":
        """Auto-detect current GPU hardware."""
```

**Implementation of `detect()`**:
```python
@staticmethod
def detect() -> "HardwareFingerprint":
    props = torch.cuda.get_device_properties(0)
    # nvidia-smi for driver, clock info
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version,clocks.sm,clocks.mem,pcie.link.gen.current",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    driver, sm_clock, mem_clock, pcie = smi.stdout.strip().split(", ")

    return HardwareFingerprint(
        gpu_name=props.name,
        compute_capability=f"{props.major}.{props.minor}",
        driver_version=driver,
        memory_gb=props.total_mem / 1e9,
        sm_count=props.multi_processor_count,
        sm_clock_mhz=int(sm_clock),
        mem_clock_mhz=int(mem_clock),
        pcie_gen=pcie,
        peak_bandwidth_gbps=measure_peak_bandwidth(),
        peak_tflops_fp16=measure_peak_flops(torch.float16),
    )
```

#### Roofline Measurement

```python
def measure_peak_bandwidth(size_bytes: int = 4 * 1024**3) -> float:
    """Measure actual DRAM bandwidth via cuMemsetAsync.

    Returns: bandwidth in GB/s.
    """
    buf = torch.empty(size_bytes // 4, dtype=torch.int32, device="cuda")
    def fn():
        buf.zero_()
    ms = triton.testing.do_bench(fn, warmup=50, rep=200)
    return size_bytes / (ms * 1e-3) / 1e9


def measure_peak_flops(dtype: torch.dtype = torch.float16, M: int = 8192) -> float:
    """Measure actual peak TFLOPS via cuBLAS GEMM.

    Returns: peak TFLOPS.
    """
    a = torch.randn(M, M, dtype=dtype, device="cuda")
    b = torch.randn(M, M, dtype=dtype, device="cuda")
    def fn():
        torch.mm(a, b)
    ms = triton.testing.do_bench(fn, warmup=50, rep=200)
    flops = 2 * M * M * M
    return flops / (ms * 1e-3) / 1e12
```

#### Built-in Baselines

```python
# tritonkit/bench/baselines.py

class BaselineRegistry:
    """Registry of known baseline implementations."""

    _baselines: dict[str, dict[str, Callable]] = {}

    @classmethod
    def register(cls, operation: str, name: str, fn: Callable) -> None:
        cls._baselines.setdefault(operation, {})[name] = fn

    @classmethod
    def get(cls, operation: str) -> dict[str, Callable]:
        return cls._baselines.get(operation, {})


def _register_builtins():
    """Register built-in baselines. Called on module import."""

    # PyTorch (cuBLAS) GEMM
    BaselineRegistry.register("gemm", "pytorch_cublas", lambda a, b: torch.mm(a, b))

    # PyTorch RMSNorm
    def pytorch_rmsnorm(x, weight, eps=1e-6):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + eps) * weight
    BaselineRegistry.register("rmsnorm", "pytorch", pytorch_rmsnorm)

    # PyTorch SDPA
    BaselineRegistry.register("attention", "pytorch_sdpa",
        lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(q, k, v))

    # Optional: Liger-Kernel (if installed)
    try:
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction
        BaselineRegistry.register("rmsnorm", "liger", LigerRMSNormFunction.apply)
    except ImportError:
        pass

    # Optional: FlagGems (if installed)
    try:
        import flag_gems
        # FlagGems registers via ATen backend, wrap appropriately
    except ImportError:
        pass

    # Optional: FlashAttention (if installed)
    try:
        from flash_attn import flash_attn_func
        BaselineRegistry.register("attention", "flash_attn", flash_attn_func)
    except ImportError:
        pass
```

---

### 2.3 `tritonkit.primitives`

Composable `@triton.jit` building blocks. These are Triton-level functions
(called inside other `@triton.jit` kernels), not standalone kernels.

#### `masked_load`

```python
@triton.jit
def masked_load(
    ptr,            # Pointer to data
    offsets,        # tl.arange-based offsets
    mask,           # Boolean mask (offsets < N)
    other: tl.constexpr = 0.0,  # Fill value for out-of-bounds
):
    """Boundary-safe tile load with configurable padding value.

    Usage:
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tk.primitives.masked_load(X_ptr + offs, offs, mask)
    """
    return tl.load(ptr + offsets, mask=mask, other=other)
```

#### `masked_store`

```python
@triton.jit
def masked_store(
    ptr,
    offsets,
    val,
    mask,
):
    """Boundary-safe tile store."""
    tl.store(ptr + offsets, val, mask=mask)
```

#### `online_softmax`

```python
@triton.jit
def online_softmax(
    qk,            # 2D tile of attention scores [BLOCK_M, BLOCK_N]
    m_prev,        # Previous row-wise max [BLOCK_M]
    l_prev,        # Previous row-wise sum of exp [BLOCK_M]
):
    """Numerically stable online softmax for Flash Attention.

    Implements the online softmax algorithm from Milakov & Gimelshein (2018).
    Updates running max and sum-of-exp, returns (p, m_new, l_new).

    Usage in Flash Attention inner loop:
        p, m_new, l_new = tk.primitives.online_softmax(qk, m_prev, l_prev)
        # Update accumulator: acc = acc * (l_prev / l_new) + p @ v

    Returns:
        p: Softmax probabilities [BLOCK_M, BLOCK_N]
        m_new: Updated row-wise max [BLOCK_M]
        l_new: Updated row-wise sum of exp [BLOCK_M]
    """
    m_curr = tl.max(qk, axis=1)
    m_new = tl.maximum(m_prev, m_curr)
    # Correction factor for previous accumulation
    alpha = tl.math.exp2(m_prev - m_new)
    # New exponentials
    p = tl.math.exp2(qk - m_new[:, None])
    l_new = alpha * l_prev + tl.sum(p, axis=1)
    return p, m_new, l_new
```

#### `online_mean_var`

```python
@triton.jit
def online_mean_var(
    x,             # 1D tile of values [BLOCK_SIZE]
    mask,          # Valid element mask
    n_cols,        # Actual number of valid elements (scalar)
):
    """Single-pass mean and variance with padding-safe masking.

    Correctly handles padded elements (sets them to 0 before variance).

    Returns:
        mean: Scalar mean
        var: Scalar variance
        rstd: Scalar 1/sqrt(var + eps)  (eps = 1e-6)
    """
    x_masked = tl.where(mask, x, 0.0)
    mean = tl.sum(x_masked, axis=0) / n_cols
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + 1e-6)
    return mean, var, rstd
```

#### `tiled_reduce`

```python
@triton.jit
def tiled_reduce(
    x,             # Tile of values
    axis: tl.constexpr,
    op: tl.constexpr,  # "sum", "max", "min"
):
    """Configurable reduction over a tile.

    Args:
        x: Input tile.
        axis: Reduction axis (0 or 1 for 2D).
        op: Reduction operation.

    Returns:
        Reduced tile.
    """
    if op == "sum":
        return tl.sum(x, axis=axis)
    elif op == "max":
        return tl.max(x, axis=axis)
    elif op == "min":
        return tl.min(x, axis=axis)
```

#### `split_k_accumulate`

```python
@triton.jit
def split_k_accumulate(
    acc,               # Current accumulator tile [BLOCK_M, BLOCK_N]
    a_tile,            # A tile [BLOCK_M, BLOCK_K]
    b_tile,            # B tile [BLOCK_K, BLOCK_N]
    dtype: tl.constexpr = tl.float32,
):
    """Accumulate a GEMM partial product into accumulator.

    Performs: acc += a_tile @ b_tile
    Handles dtype promotion to fp32 accumulator.

    Returns:
        Updated accumulator.
    """
    return acc + tl.dot(a_tile, b_tile, out_dtype=dtype)
```

#### `swizzle_tile`

```python
@triton.jit
def swizzle_tile(
    pid,                     # Program ID
    M: tl.constexpr,        # Grid dimension M
    N: tl.constexpr,        # Grid dimension N
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr = 8,
):
    """L2-cache-friendly tile ordering for GEMM.

    Instead of row-major tile ordering, groups tiles to improve L2
    cache reuse for B matrix tiles.

    Returns:
        (block_m_idx, block_n_idx): Tile coordinates.
    """
    num_blocks_m = tl.cdiv(M, BLOCK_M)
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    num_blocks_in_group = GROUP_SIZE * num_blocks_n
    group_id = pid // num_blocks_in_group
    first_block_m = group_id * GROUP_SIZE
    group_size_m = min(num_blocks_m - first_block_m, GROUP_SIZE)
    block_m_idx = first_block_m + ((pid % num_blocks_in_group) % group_size_m)
    block_n_idx = (pid % num_blocks_in_group) // group_size_m
    return block_m_idx, block_n_idx
```

#### `block_quantize`

```python
@triton.jit
def block_quantize(
    x,                       # FP16/BF16 tile to quantize
    BLOCK_SIZE: tl.constexpr,
    BITS: tl.constexpr = 8,  # 8 for INT8, 4 for INT4
):
    """Per-block symmetric quantization.

    Computes scale = max(abs(x)) / (2^(BITS-1) - 1), then
    x_quant = round(x / scale).

    Returns:
        x_quant: Quantized tile (int8 or int32 depending on BITS)
        scale: Per-block scale factor (scalar)
    """
    abs_max = tl.max(tl.abs(x), axis=0)
    qmax = (1 << (BITS - 1)) - 1
    scale = abs_max / qmax
    scale = tl.where(scale == 0.0, 1.0, scale)  # avoid div-by-zero
    x_quant = tl.math.nearbyint(x / scale)
    x_quant = tl.minimum(tl.maximum(x_quant, -qmax), qmax)
    return x_quant, scale
```

---

### 2.4 `tritonkit.integrate`

Tools to connect Triton kernels to PyTorch ecosystem.

#### `register_torch_op`

```python
def register_torch_op(
    name: str,
    fn: Callable,
    schema: str,
    namespace: str = "tritonkit",
) -> None:
    """Register a Triton kernel wrapper as a PyTorch custom op.

    Args:
        name: Op name (e.g., "rmsnorm"). Will be registered as {namespace}::{name}.
        fn: Python function wrapping the Triton kernel.
        schema: PyTorch op schema string.
            E.g., "(Tensor x, Tensor weight, float eps) -> Tensor"
        namespace: Custom op namespace.

    Example:
        tk.integrate.register_torch_op(
            "rmsnorm", my_rmsnorm, "(Tensor x, Tensor weight, float eps) -> Tensor"
        )
        # Now callable as torch.ops.tritonkit.rmsnorm(x, w, 1e-6)
    """
    lib = torch.library.Library(namespace, "DEF")
    lib.define(f"{name}{schema}")
    lib.impl(name, fn, "CUDA")
```

#### `make_compilable`

```python
def make_compilable(
    fn: Callable,
    dynamic_shapes: bool = True,
) -> Callable:
    """Wrap a Triton kernel function for torch.compile compatibility.

    Registers the function as a custom op with FakeTensor support
    so that torch.compile can trace through it.

    Args:
        fn: The Python wrapper around a Triton kernel.
        dynamic_shapes: Whether to support dynamic shapes.

    Returns:
        Wrapped function compatible with torch.compile.
    """
```

#### `export_autotune_config`

```python
def export_autotune_config(
    kernel: triton.JITFunction,
    path: str,
    hardware: HardwareFingerprint | None = None,
) -> None:
    """Export the best autotuning config for current hardware to JSON.

    Runs the kernel's autotune configs, finds the best, and saves it
    alongside the hardware fingerprint.

    Args:
        kernel: A @triton.jit kernel with @triton.autotune.
        path: Output JSON path.
        hardware: Hardware fingerprint. Auto-detected if None.
    """
```

---

### 2.5 `tritonkit.examples`

Reference kernels built with the primitives. Each example has:
- A `@triton.jit` kernel function
- A Python wrapper callable from PyTorch
- A corresponding test in `tests/test_examples.py`
- A corresponding benchmark in `benchmarks/suites/`

#### Example: `rmsnorm_fused`

```python
# tritonkit/examples/rmsnorm_fused.py

import torch
import triton
import triton.language as tl
from tritonkit.primitives import masked_load, masked_store, online_mean_var


@triton.jit
def _rmsnorm_fused_kernel(
    X, Y, W, Residual, stride,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(X + row * stride + offs, mask=mask, other=0.0).to(tl.float32)

    if HAS_RESIDUAL:
        res = tl.load(Residual + row * stride + offs, mask=mask, other=0.0).to(tl.float32)
        x = x + res

    # Variance (RMSNorm: no mean subtraction)
    x_sq = tl.where(mask, x * x, 0.0)
    var = tl.sum(x_sq, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize and scale
    w = tl.load(W + offs, mask=mask)
    y = x * rstd * w

    tl.store(Y + row * stride + offs, y, mask=mask)


def rmsnorm_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused RMSNorm + optional residual addition.

    Args:
        x: Input tensor [..., N].
        weight: Scale parameter [N].
        eps: Epsilon for numerical stability.
        residual: Optional residual tensor (same shape as x).

    Returns:
        Normalized tensor, same shape as x.
    """
    x_2d = x.reshape(-1, x.shape[-1])
    M, N = x_2d.shape
    y = torch.empty_like(x_2d)
    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

    _rmsnorm_fused_kernel[(M,)](
        x_2d, y, weight,
        residual.reshape(-1, N) if residual is not None else x_2d,
        x_2d.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_RESIDUAL=residual is not None,
        num_warps=num_warps,
    )
    return y.reshape(x.shape)
```

Other examples follow the same pattern. Full implementations in `src/tritonkit/examples/`.

---

### 2.6 `tritonkit` CLI

Entry point registered via `pyproject.toml`:

```toml
[project.scripts]
tritonkit = "tritonkit.cli.main:main"
```

#### Commands

```
tritonkit bench [OPTIONS]
    --suite SUITE          Benchmark suite: gemm, attention, norm, activation, softmax, moe, all
    --kernel KERNEL        Specific kernel to benchmark
    --impl IMPL            Specific implementation(s), comma-separated
    --dtype DTYPE           fp16, bf16, fp32, int8 (default: fp16)
    --shapes SHAPES        Custom shapes, e.g. "4096x4096,8192x8192"
    --mode MODE            "default" or "best" (default: default)
    --warmup-ms FLOAT      Warmup time budget (default: 25)
    --rep-ms FLOAT         Measurement time budget (default: 100)
    --output PATH          Output JSON/CSV path
    --format FORMAT        "table" (default), "json", "csv"
    --roofline             Generate roofline plot (requires matplotlib)
    --submit               Export results for PR submission

tritonkit test [OPTIONS]
    --suite SUITE          Test suite to run: all, examples, primitives
    --kernel KERNEL        Test specific kernel
    --dtypes DTYPES        Dtypes to test, comma-separated
    --verbose              Verbose output

tritonkit info
    Print hardware fingerprint and software versions.
```

#### CLI Implementation

```python
# tritonkit/cli/main.py

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="tritonkit")
    subparsers = parser.add_subparsers(dest="command")

    # bench subcommand
    bench_parser = subparsers.add_parser("bench", help="Run benchmarks")
    bench_parser.add_argument("--suite", default="all")
    bench_parser.add_argument("--kernel", default=None)
    bench_parser.add_argument("--impl", default=None)
    bench_parser.add_argument("--dtype", default="fp16")
    bench_parser.add_argument("--shapes", default=None)
    bench_parser.add_argument("--mode", default="default", choices=["default", "best"])
    bench_parser.add_argument("--warmup-ms", type=float, default=25.0)
    bench_parser.add_argument("--rep-ms", type=float, default=100.0)
    bench_parser.add_argument("--output", default=None)
    bench_parser.add_argument("--format", default="table", choices=["table", "json", "csv"])
    bench_parser.add_argument("--roofline", action="store_true")
    bench_parser.add_argument("--submit", action="store_true")

    # test subcommand
    test_parser = subparsers.add_parser("test", help="Run correctness tests")
    test_parser.add_argument("--suite", default="all")
    test_parser.add_argument("--kernel", default=None)
    test_parser.add_argument("--dtypes", default="fp16,bf16")
    test_parser.add_argument("--verbose", action="store_true")

    # info subcommand
    subparsers.add_parser("info", help="Print hardware and software info")

    args = parser.parse_args()

    if args.command == "bench":
        _run_bench(args)
    elif args.command == "test":
        _run_test(args)
    elif args.command == "info":
        _run_info()
    else:
        parser.print_help()
```

---

## 3. Data Models

### Benchmark Result JSON Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["version", "hardware", "software", "mode", "timestamp", "results"],
  "properties": {
    "version": { "type": "string", "description": "tritonkit version" },
    "hardware": {
      "type": "object",
      "required": ["gpu_name", "compute_capability", "driver_version", "memory_gb", "sm_count", "sm_clock_mhz", "mem_clock_mhz", "peak_bandwidth_gbps", "peak_tflops_fp16"],
      "properties": {
        "gpu_name": { "type": "string" },
        "compute_capability": { "type": "string" },
        "driver_version": { "type": "string" },
        "memory_gb": { "type": "number" },
        "sm_count": { "type": "integer" },
        "sm_clock_mhz": { "type": "integer" },
        "mem_clock_mhz": { "type": "integer" },
        "peak_bandwidth_gbps": { "type": "number" },
        "peak_tflops_fp16": { "type": "number" }
      }
    },
    "software": {
      "type": "object",
      "required": ["triton", "cuda", "pytorch", "python", "tritonkit"],
      "properties": {
        "triton": { "type": "string" },
        "cuda": { "type": "string" },
        "pytorch": { "type": "string" },
        "python": { "type": "string" },
        "tritonkit": { "type": "string" }
      }
    },
    "mode": { "enum": ["default", "best"] },
    "timestamp": { "type": "string", "format": "date-time" },
    "results": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["kernel", "implementation", "shape", "dtype", "median_us", "p20_us", "p80_us"],
        "properties": {
          "kernel": { "type": "string" },
          "implementation": { "type": "string" },
          "shape": { "type": "array", "items": { "type": "integer" } },
          "dtype": { "type": "string" },
          "median_us": { "type": "number" },
          "p20_us": { "type": "number" },
          "p80_us": { "type": "number" },
          "tflops": { "type": ["number", "null"] },
          "throughput_gbps": { "type": ["number", "null"] },
          "roofline_efficiency": { "type": ["number", "null"] },
          "reproduce": { "type": "string" }
        }
      }
    }
  }
}
```

---

## 4. Dependencies

### `pyproject.toml`

```toml
[project]
name = "tritonkit"
version = "0.1.0"
description = "Build, test, and benchmark Triton kernels"
readme = "README.md"
license = { text = "Apache-2.0" }
requires-python = ">=3.10"
authors = [{ name = "Dongcheng Ye" }]

dependencies = [
    "triton>=3.6.0",
    "torch>=2.10.0",
    "numpy>=1.24.0",
    "tabulate>=0.9.0",        # For print_table()
]

[project.optional-dependencies]
plot = [
    "matplotlib>=3.7.0",      # For roofline plots
]
bench = [
    "liger-kernel",            # Baseline: training kernels
    "flag-gems",               # Baseline: operator library
    "flash-attn",              # Baseline: flash attention
]
dev = [
    "pytest>=7.0",
    "ruff>=0.4.0",
    "black>=24.0",
    "mypy>=1.0",
]
all = [
    "tritonkit[plot,bench,dev]",
]

[project.scripts]
tritonkit = "tritonkit.cli.main:main"

[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.black]
target-version = ["py310"]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

---

## 5. Testing Strategy

### Self-Testing (Testing the Testing Framework)

```python
# tests/test_testing.py

def test_assert_matches_passes_for_identical():
    """assert_matches should pass when triton_fn == reference_fn."""
    def identity(x):
        return x
    tk.testing.assert_matches(identity, identity, shapes=[(128,)])


def test_assert_matches_fails_for_wrong():
    """assert_matches should fail with clear error message."""
    def wrong(x):
        return x + 1
    def correct(x):
        return x
    with pytest.raises(AssertionError, match="mismatched"):
        tk.testing.assert_matches(wrong, correct, shapes=[(128,)])


def test_numerical_stability_catches_nan():
    """check_numerical_stability catches NaN-producing kernels."""
    def produces_nan(x):
        return x / 0.0
    with pytest.raises(AssertionError, match="NaN"):
        tk.testing.check_numerical_stability(produces_nan)
```

### Example Kernel Tests

```python
# tests/test_examples.py

import torch
import tritonkit as tk
from tritonkit.examples import rmsnorm_fused

def test_rmsnorm_fused_correctness():
    tk.testing.assert_matches(
        triton_fn=rmsnorm_fused,
        reference_fn=lambda x, w, eps=1e-6: torch.nn.functional.rms_norm(x, w.shape, w, eps),
        shapes=tk.testing.LLM_SHAPES,
        dtypes=[torch.float16, torch.bfloat16],
    )

def test_rmsnorm_fused_with_residual():
    for shape in [(128, 4096), (512, 4096)]:
        x = torch.randn(shape, device="cuda", dtype=torch.float16)
        res = torch.randn(shape, device="cuda", dtype=torch.float16)
        w = torch.randn(shape[-1], device="cuda", dtype=torch.float16)
        out = rmsnorm_fused(x, w, residual=res)
        ref = torch.nn.functional.rms_norm(x + res, (shape[-1],), w, 1e-6)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
```

### CI Strategy

- **Local GPU required** for all tests and benchmarks
- `pytest tests/` for correctness tests
- `tritonkit bench --suite=all --format=json --output=results/ci.json` for benchmarks
- GitHub Actions: use self-hosted runner with RTX 3080 Ti
- Later: vast.ai runners for H100/MI300

---

## 6. Coding Standards

| Aspect | Standard |
|--------|---------|
| Formatter | `black --line-length 100` |
| Linter | `ruff` with default rules |
| Type checking | `mypy --strict` on non-Triton code |
| Docstrings | Google style |
| Triton kernels | `@triton.jit`, prefixed with `_` (private), camelCase constexpr params |
| Python wrappers | snake_case, full type hints |
| Tests | pytest, one test file per module |
| Imports | `import tritonkit as tk` for public API |

---

## 7. Implementation Priority

Phase 1 implementation order (weeks 1-6):

| Order | File | Why First |
|-------|------|-----------|
| 1 | `pyproject.toml` | Project skeleton, pip install works |
| 2 | `src/tritonkit/__init__.py`, `_version.py` | Package exists |
| 3 | `src/tritonkit/testing/shapes.py` | Zero dependencies, immediate value |
| 4 | `src/tritonkit/testing/correctness.py` | Core feature #1 |
| 5 | `src/tritonkit/bench/hardware.py` | Needed by benchmark framework |
| 6 | `src/tritonkit/bench/roofline.py` | Needed for peak measurements |
| 7 | `src/tritonkit/bench/result.py` | Data model for results |
| 8 | `src/tritonkit/bench/runner.py` | Core feature #2 |
| 9 | `src/tritonkit/bench/baselines.py` | Built-in comparison targets |
| 10 | `src/tritonkit/primitives/*.py` | Building blocks |
| 11 | `src/tritonkit/examples/rmsnorm_fused.py` | First showcase kernel |
| 12 | `src/tritonkit/examples/gemm_fp16.py` | GEMM showcase |
| 13 | `src/tritonkit/examples/flash_attention.py` | Attention showcase |
| 14 | `src/tritonkit/cli/main.py` | CLI interface |
| 15 | `tests/*` | Validate everything |
| 16 | `benchmarks/suites/*` | Generate first benchmark data |
