"""Flash Attention v2 forward kernel."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tritonkit.primitives import online_softmax


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _flash_attention_fwd_kernel(
    Q,
    K,
    V,
    O,
    SEQ_LEN,
    HEAD_DIM,
    stride_qz,
    stride_qm,
    stride_qd,
    stride_kz,
    stride_km,
    stride_kd,
    stride_vz,
    stride_vm,
    stride_vd,
    stride_oz,
    stride_om,
    stride_od,
    SM_SCALE_LOG2E,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_z = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = (
        Q
        + pid_z * stride_qz
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q_mask = (offs_m[:, None] < SEQ_LEN) & (offs_d[None, :] < HEAD_DIM)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    for start_n in range(0, SEQ_LEN, BLOCK_N):
        curr_n = start_n + offs_n

        k_ptrs = (
            K
            + pid_z * stride_kz
            + offs_d[:, None] * stride_kd
            + curr_n[None, :] * stride_km
        )
        v_ptrs = (
            V
            + pid_z * stride_vz
            + curr_n[:, None] * stride_vm
            + offs_d[None, :] * stride_vd
        )

        k_mask = (offs_d[:, None] < HEAD_DIM) & (curr_n[None, :] < SEQ_LEN)
        v_mask = (curr_n[:, None] < SEQ_LEN) & (offs_d[None, :] < HEAD_DIM)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        qk = tl.dot(q, k, out_dtype=tl.float32) * SM_SCALE_LOG2E
        score_mask = curr_n[None, :] < SEQ_LEN
        if CAUSAL:
            score_mask = score_mask & (curr_n[None, :] <= offs_m[:, None])
        qk = tl.where(score_mask, qk, -float("inf"))
        qk = tl.where(offs_m[:, None] < SEQ_LEN, qk, 0.0)

        p, m_new, l_new = online_softmax(qk, m_i, l_i)
        acc_scale = tl.where(l_new > 0, (l_i / l_new) * tl.math.exp2(m_i - m_new), 0.0)
        p_scale = tl.where(l_new > 0, 1.0 / l_new, 0.0)

        acc = acc * acc_scale[:, None]
        acc += tl.dot(p.to(v.dtype), v, out_dtype=tl.float32) * p_scale[:, None]
        m_i = m_new
        l_i = l_new

    o_ptrs = (
        O
        + pid_z * stride_oz
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    o_mask = (offs_m[:, None] < SEQ_LEN) & (offs_d[None, :] < HEAD_DIM)
    tl.store(o_ptrs, acc, mask=o_mask)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Flash Attention v2 forward pass for [batch, heads, seq, dim] tensors."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [batch, n_heads, seq_len, head_dim]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"Shape mismatch: q={q.shape}, k={k.shape}, v={v.shape}")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.device.type != "cuda":
        raise ValueError("flash_attention requires CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Unsupported dtype: {q.dtype}")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k, and v must have the same dtype")

    batch, n_heads, seq_len, head_dim = q.shape
    sm_scale_log2e = (head_dim**-0.5) * 1.4426950408889634

    q_3d = q.contiguous().reshape(batch * n_heads, seq_len, head_dim)
    k_3d = k.contiguous().reshape(batch * n_heads, seq_len, head_dim)
    v_3d = v.contiguous().reshape(batch * n_heads, seq_len, head_dim)
    o_3d = torch.empty_like(q_3d)

    grid = lambda meta: (
        triton.cdiv(seq_len, meta["BLOCK_M"]),
        batch * n_heads,
    )

    _flash_attention_fwd_kernel[grid](
        q_3d,
        k_3d,
        v_3d,
        o_3d,
        seq_len,
        head_dim,
        q_3d.stride(0),
        q_3d.stride(1),
        q_3d.stride(2),
        k_3d.stride(0),
        k_3d.stride(1),
        k_3d.stride(2),
        v_3d.stride(0),
        v_3d.stride(1),
        v_3d.stride(2),
        o_3d.stride(0),
        o_3d.stride(1),
        o_3d.stride(2),
        sm_scale_log2e,
        BLOCK_DMODEL=triton.next_power_of_2(head_dim),
        CAUSAL=causal,
    )
    return o_3d.reshape(batch, n_heads, seq_len, head_dim)
