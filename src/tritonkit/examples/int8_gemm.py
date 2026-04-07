"""INT8 GEMM with symmetric per-tensor dequantization."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE": 8}, num_warps=4, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _int8_gemm_kernel(
    A,
    B,
    SCALE_A,
    SCALE_B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks_m = tl.cdiv(M, BLOCK_M)
    num_blocks_n = tl.cdiv(N, BLOCK_N)

    num_blocks_in_group = GROUP_SIZE * num_blocks_n
    group_id = pid // num_blocks_in_group
    first_block_m = group_id * GROUP_SIZE
    group_size_m = min(num_blocks_m - first_block_m, GROUP_SIZE)
    block_m = first_block_m + ((pid % num_blocks_in_group) % group_size_m)
    block_n = (pid % num_blocks_in_group) // group_size_m

    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k_start in range(0, K, BLOCK_K):
        k_mask = (k_start + offs_k) < K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0)
        acc += tl.dot(a, b, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    scale = tl.load(SCALE_A).to(tl.float32) * tl.load(SCALE_B).to(tl.float32)
    out = acc.to(tl.float32) * scale

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, out.to(tl.float16), mask=c_mask)


def int8_gemm(
    a_int8: torch.Tensor,
    b_int8: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """Multiply int8 matrices and dequantize the result to fp16."""
    if a_int8.device.type != "cuda" or b_int8.device.type != "cuda":
        raise ValueError("int8_gemm requires CUDA tensors")
    if a_int8.dtype != torch.int8 or b_int8.dtype != torch.int8:
        raise ValueError("int8_gemm expects int8 inputs")
    if a_int8.ndim != 2 or b_int8.ndim != 2:
        raise ValueError("int8_gemm expects rank-2 matrices")
    if a_int8.shape[1] != b_int8.shape[0]:
        raise ValueError(f"Incompatible shapes: {a_int8.shape} and {b_int8.shape}")
    if scale_a.numel() != 1 or scale_b.numel() != 1:
        raise ValueError("scale_a and scale_b must be scalar tensors")

    a_int8 = a_int8.contiguous()
    b_int8 = b_int8.contiguous()
    scale_a = scale_a.to(device=a_int8.device, dtype=torch.float16).reshape(())
    scale_b = scale_b.to(device=a_int8.device, dtype=torch.float16).reshape(())

    m, k = a_int8.shape
    _, n = b_int8.shape
    c = torch.empty((m, n), device=a_int8.device, dtype=torch.float16)

    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)

    _int8_gemm_kernel[grid](
        a_int8,
        b_int8,
        scale_a,
        scale_b,
        c,
        m,
        n,
        k,
        a_int8.stride(0),
        a_int8.stride(1),
        b_int8.stride(0),
        b_int8.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c
