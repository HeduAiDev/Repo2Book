# SOURCE: third_party/ascend/language/kernels/gather.py:L32-106
#
# gather_2d_simd —— m13 的"第三条路"：不用任何昇腾扩展算子，纯上游 tl.arange/
# tl.load/tl.gather/tl.store 写成的 2D gather（axis=1）。索引搬运发生在片上
# （先把整行 src/idx 都 tl.load 进 tile，再在 tile 内部用基座 tl.gather 取数），
# 与 mem_ops.py 的 gather_out_to_ub（GM 侧直接按索引离散访存）是两种不同粒度的写法
# ——对照见本章 m13。逐字保留，未做任何删减（无长 docstring 需要精简）。

__all__ = ["gather_2d_simd"]

import triton
import triton.language as tl
from triton.language.core import constexpr


@triton.jit
def gather_2d_simd(  # SOURCE: third_party/ascend/language/kernels/gather.py:L32-106(gather_2d_simd)
    src_ptr,
    idx_ptr,
    out_ptr,
    M: constexpr,
    N: constexpr,
    K: constexpr,
    XBLOCK: constexpr,
    XBLOCK_SUB: constexpr
):
    """
    2D gather kernel for axis=1 (tail axis) with SIMD-style vectorization.

    Args:
        src_ptr: [M, N] source tensor in GM (Global Memory)
        idx_ptr: [M, K] indices tensor in GM
        out_ptr: [M, K] output tensor in GM
        M/N/K: shape sizes; XBLOCK/XBLOCK_SUB: outer/inner block sizes for the
            M dimension (outer for program distribution, inner for
            vectorization)
    """
    pid = tl.program_id(0)
    m_start = pid * XBLOCK
    m_end = min(m_start + XBLOCK, M)
    m_base = tl.arange(0, XBLOCK_SUB)

    # Process multiple rows at once using XBLOCK_SUB for vectorization
    for m_tile_start in range(m_start, m_end, XBLOCK_SUB):
        # M dimension offsets: [XBLOCK_SUB]
        m_offs = m_tile_start + m_base
        m_mask = m_offs < M

        # Load indices: [XBLOCK_SUB, K]
        k_offs = tl.arange(0, K)
        idx_tile = tl.load(
            idx_ptr + m_offs[:, None] * K + k_offs[None, :]
        )

        # Load source data: [XBLOCK_SUB, N]
        n_offs = tl.arange(0, N)
        src_tile = tl.load(
            src_ptr + m_offs[:, None] * N + n_offs[None, :]
        )

        # Gather operation along axis=1
        gathered_values = tl.gather(src_tile, idx_tile, axis=1)

        # Store results
        tl.store(
            out_ptr + m_offs[:, None] * K + k_offs[None, :],
            gathered_values,
            mask=m_mask[:, None]
        )
