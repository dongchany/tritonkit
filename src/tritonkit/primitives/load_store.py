"""Boundary-safe tile load/store primitives."""

import triton
import triton.language as tl


@triton.jit
def masked_load(ptr, offsets, mask, other: tl.constexpr = 0.0):
    """Boundary-safe tile load with configurable padding value."""
    return tl.load(ptr + offsets, mask=mask, other=other)


@triton.jit
def masked_store(ptr, offsets, val, mask):
    """Boundary-safe tile store."""
    tl.store(ptr + offsets, val, mask=mask)
