"""W4A16 GEMM with symmetric per-group dequantization."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
    ],
    key=["M", "N", "K", "QUANT_GROUP_SIZE"],
)
@triton.jit
def _w4a16_gemm_kernel(
    A,
    QWEIGHT,
    SCALES,
    C,
    M,
    N,
    K,
    QUANT_GROUP_SIZE,
    stride_am,
    stride_ak,
    stride_qwk,
    stride_qwn,
    stride_sk,
    stride_sn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks_m = tl.cdiv(M, BLOCK_M)
    num_blocks_n = tl.cdiv(N, BLOCK_N)

    num_blocks_in_group = GROUP_M * num_blocks_n
    group_id = pid // num_blocks_in_group
    first_block_m = group_id * GROUP_M
    group_size_m = min(num_blocks_m - first_block_m, GROUP_M)
    block_m = first_block_m + ((pid % num_blocks_in_group) % group_size_m)
    block_n = (pid % num_blocks_in_group) // group_size_m

    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        curr_k = k_start + offs_k
        k_mask = curr_k < K
        n_mask = offs_n < N

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)

        qweight_rows = curr_k // 8
        qweight_shifts = (curr_k % 8) * 4
        qweight_ptrs = QWEIGHT + qweight_rows[:, None] * stride_qwk + offs_n[None, :] * stride_qwn
        packed = tl.load(qweight_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)

        q_unsigned = (packed >> qweight_shifts[:, None]) & 0xF
        q_signed = tl.where(q_unsigned >= 8, q_unsigned - 16, q_unsigned)

        scale_rows = curr_k // QUANT_GROUP_SIZE
        scale_ptrs = SCALES + scale_rows[:, None] * stride_sk + offs_n[None, :] * stride_sn
        scale = tl.load(scale_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        b = q_signed.to(tl.float16) * scale
        acc += tl.dot(a, b, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def w4a16_gemm(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Multiply FP16 activations by GPTQ/AWQ-style packed INT4 weights."""
    if a.device.type != "cuda" or qweight.device.type != "cuda" or scales.device.type != "cuda":
        raise ValueError("w4a16_gemm requires CUDA tensors")
    if a.device != qweight.device or a.device != scales.device:
        raise ValueError("a, qweight, and scales must be on the same device")
    if a.dtype != torch.float16:
        raise ValueError("w4a16_gemm expects a to be float16")
    if qweight.dtype != torch.int32:
        raise ValueError("w4a16_gemm expects qweight to be int32")
    if scales.dtype != torch.float16:
        raise ValueError("w4a16_gemm expects scales to be float16")
    if a.ndim != 2 or qweight.ndim != 2 or scales.ndim != 2:
        raise ValueError("w4a16_gemm expects rank-2 tensors")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    m, k = a.shape
    packed_k, n = qweight.shape

    if packed_k * 8 != k:
        raise ValueError(f"Expected qweight.shape[0] * 8 == K, got {packed_k * 8} and {k}")
    if k % group_size != 0:
        raise ValueError(f"K={k} must be divisible by group_size={group_size}")
    if scales.shape != (k // group_size, n):
        raise ValueError(
            f"Expected scales shape {(k // group_size, n)}, got {tuple(scales.shape)}"
        )

    a = a.contiguous()
    qweight = qweight.contiguous()
    scales = scales.contiguous()
    c = torch.empty((m, n), device=a.device, dtype=torch.float16)

    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)

    _w4a16_gemm_kernel[grid](
        a,
        qweight,
        scales,
        c,
        m,
        n,
        k,
        group_size,
        a.stride(0),
        a.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        scales.stride(0),
        scales.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c
