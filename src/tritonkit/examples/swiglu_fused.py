"""Fused SwiGLU activation kernel."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_fused_kernel(
    Gate,
    Up,
    Out,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    gate = tl.load(Gate + offs, mask=mask).to(tl.float32)
    up = tl.load(Up + offs, mask=mask).to(tl.float32)

    # SiLU(gate) * up
    silu_gate = gate * tl.sigmoid(gate)
    out = silu_gate * up

    tl.store(Out + offs, out, mask=mask)


def swiglu_fused(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """Fused SwiGLU: SiLU(gate) * up.

    Args:
        gate: Gate tensor.
        up: Up-projection tensor (same shape as gate).

    Returns:
        SwiGLU output, same shape as inputs.
    """
    assert gate.shape == up.shape, f"Shape mismatch: gate {gate.shape} vs up {up.shape}"
    out = torch.empty_like(gate)
    N = gate.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    _swiglu_fused_kernel[grid](
        gate.reshape(-1),
        up.reshape(-1),
        out.reshape(-1),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
