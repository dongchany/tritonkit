"""Statistical primitives for normalization kernels."""

import triton
import triton.language as tl


@triton.jit
def online_mean_var(x, mask, n_cols):
    """Single-pass mean and variance with padding-safe masking.

    Args:
        x: 1D tile of values [BLOCK_SIZE].
        mask: Valid element mask.
        n_cols: Actual number of valid elements (scalar).

    Returns:
        (mean, var, rstd): Scalar mean, variance, and 1/sqrt(var + eps).
    """
    x_masked = tl.where(mask, x, 0.0)
    mean = tl.sum(x_masked, axis=0) / n_cols
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + 1e-6)
    return mean, var, rstd
