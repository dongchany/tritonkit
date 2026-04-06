"""GEMM utility primitives."""

import triton
import triton.language as tl


@triton.jit
def split_k_accumulate(acc, a_tile, b_tile, dtype: tl.constexpr = tl.float32):
    """Accumulate a GEMM partial product: acc += a_tile @ b_tile."""
    return acc + tl.dot(a_tile, b_tile, out_dtype=dtype)


@triton.jit
def swizzle_tile(pid, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_SIZE: tl.constexpr = 8):
    """L2-cache-friendly tile ordering for GEMM.

    Groups tiles to improve L2 cache reuse for B matrix tiles.

    Returns:
        (block_m_idx, block_n_idx): Tile coordinates.
    """
    num_blocks_m = tl.cdiv(M, BLOCK_M)
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    num_blocks_in_group = GROUP_SIZE * num_blocks_n
    group_id = pid // num_blocks_in_group
    first_block_m = group_id * GROUP_SIZE
    group_size_m = min(num_blocks_m - first_block_m, GROUP_SIZE)
    block_m_idx = first_block_m + ((pid % num_blocks_in_group) % group_size_m)
    block_n_idx = (pid % num_blocks_in_group) // group_size_m
    return block_m_idx, block_n_idx
