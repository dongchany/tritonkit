"""Reduction primitives."""

import triton
import triton.language as tl


@triton.jit
def tiled_reduce(x, axis: tl.constexpr, op: tl.constexpr):
    """Configurable reduction over a tile.

    Args:
        x: Input tile.
        axis: Reduction axis (0 or 1 for 2D).
        op: "sum", "max", or "min".
    """
    if op == "sum":
        return tl.sum(x, axis=axis)
    elif op == "max":
        return tl.max(x, axis=axis)
    elif op == "min":
        return tl.min(x, axis=axis)
