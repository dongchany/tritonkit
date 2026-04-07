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

try:
    from tritonkit.examples.flash_attention import flash_attention
except ImportError:
    flash_attention = None

try:
    from tritonkit.examples.int8_gemm import int8_gemm
except ImportError:
    int8_gemm = None

try:
    from tritonkit.examples.w4a16_gemm import w4a16_gemm
except ImportError:
    w4a16_gemm = None

__all__ = ["rmsnorm_fused", "gemm_fp16", "swiglu_fused", "flash_attention", "int8_gemm", "w4a16_gemm"]
