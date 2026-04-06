"""Fused RMSNorm + optional residual addition."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fused_kernel(
    X,
    Y,
    W,
    Residual,
    stride,
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

    # RMSNorm: no mean subtraction, just variance of x
    x_sq = tl.where(mask, x * x, 0.0)
    var = tl.sum(x_sq, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

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
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    M, N = x_2d.shape
    y = torch.empty_like(x_2d)
    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

    res_ptr = residual.reshape(-1, N) if residual is not None else x_2d

    _rmsnorm_fused_kernel[(M,)](
        x_2d,
        y,
        weight,
        res_ptr,
        x_2d.stride(0),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_RESIDUAL=residual is not None,
        num_warps=num_warps,
    )
    return y.reshape(orig_shape)
