"""Quantization primitives for low-precision Triton kernels."""

import triton
import triton.language as tl


@triton.jit
def block_quantize(x, BLOCK_SIZE: tl.constexpr, BITS: tl.constexpr = 8):
    """Per-block symmetric quantization with a shared scale."""
    tl.static_assert(BLOCK_SIZE > 0)
    tl.static_assert(BITS > 1)

    qmax = (1 << (BITS - 1)) - 1
    absmax = tl.max(tl.abs(x), axis=0)
    scale = tl.where(absmax > 0, absmax / qmax, 1.0)

    x_scaled = x / scale
    x_rounded = tl.where(
        x_scaled >= 0,
        tl.floor(x_scaled + 0.5),
        tl.ceil(x_scaled - 0.5),
    )
    x_quant = tl.clamp(x_rounded, -qmax, qmax).to(tl.int32)
    return x_quant, scale
