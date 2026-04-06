"""Online softmax for Flash Attention."""

import triton
import triton.language as tl


@triton.jit
def online_softmax(qk, m_prev, l_prev):
    """Numerically stable online softmax (Milakov & Gimelshein 2018).

    Updates running max and sum-of-exp for streaming softmax computation.

    Args:
        qk: 2D tile of attention scores [BLOCK_M, BLOCK_N].
        m_prev: Previous row-wise max [BLOCK_M].
        l_prev: Previous row-wise sum of exp [BLOCK_M].

    Returns:
        (p, m_new, l_new): Softmax probs, updated max, updated sum.
    """
    m_curr = tl.max(qk, axis=1)
    m_new = tl.maximum(m_prev, m_curr)
    alpha = tl.math.exp2(m_prev - m_new)
    p = tl.math.exp2(qk - m_new[:, None])
    l_new = alpha * l_prev + tl.sum(p, axis=1)
    return p, m_new, l_new
