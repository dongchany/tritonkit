"""Reference kernels built with tritonkit primitives."""

try:
    from tritonkit.examples.rmsnorm_fused import rmsnorm_fused
except ImportError:
    rmsnorm_fused = None

try:
    from tritonkit.examples.gemm_fp16 import gemm_fp16
except ImportError:
    gemm_fp16 = None

try:
    from tritonkit.examples.swiglu_fused import swiglu_fused
except ImportError:
    swiglu_fused = None

__all__ = ["rmsnorm_fused", "gemm_fp16", "swiglu_fused"]
