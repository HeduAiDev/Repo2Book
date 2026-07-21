"""m16 worked example：tiling 之后循环体里仍是同一个结构化算子；结果与不切时
数值一致（"legal by design"，[Linalg §3.6] paper.md:L385）。
"""
import numpy as np

from named_ops import make_conv_1d_nwc_wcf, make_matmul
from tiling import tile_and_run
from verify import assert_same_result


def test_tiled_conv_matches_untiled_evenly_divisible():
    op = make_conv_1d_nwc_wcf()
    rng = np.random.default_rng(1)
    N, W, C, F, KW = 1, 16, 2, 3, 3
    out_w = W - KW + 1  # 14,恰好被 tile_w=7 整除
    I = rng.standard_normal((N, W, C))
    K = rng.standard_normal((KW, C, F))
    untiled = op.apply({"I": I, "K": K}, out_shape=(N, out_w, F))
    tiled = tile_and_run(op, {"I": I, "K": K}, out_shape=(N, out_w, F), tile_sizes={"w": 7})
    assert_same_result(untiled, tiled)


def test_tiled_conv_matches_untiled_with_boundary_tile():
    """[Linalg §3.1] 诚实交代:tile size 除不尽时边界 tile 更短
    （paper.md:L337-L339）。这里刻意选一个不整除的 tile 宽度触发边界 tile,
    验证 tile() 对末端短 tile 的处理仍然数值正确。"""
    op = make_conv_1d_nwc_wcf()
    rng = np.random.default_rng(2)
    N, W, C, F, KW = 1, 16, 2, 3, 3
    out_w = W - KW + 1  # 14, tile_w=5 -> 5,5,4(边界)
    I = rng.standard_normal((N, W, C))
    K = rng.standard_normal((KW, C, F))
    untiled = op.apply({"I": I, "K": K}, out_shape=(N, out_w, F))
    tiled = tile_and_run(op, {"I": I, "K": K}, out_shape=(N, out_w, F), tile_sizes={"w": 5})
    assert_same_result(untiled, tiled)


def test_tiled_matmul_matches_untiled_two_tiled_dims():
    """同时对两个 parallel 维 (i,j) 分块,归约维 k 保持满量程——验证多维 tiling
    的组合正确性,而不只是单维。"""
    op = make_matmul()
    rng = np.random.default_rng(3)
    M, K_, N = 10, 6, 7
    A = rng.standard_normal((M, K_))
    B = rng.standard_normal((K_, N))
    untiled = op.apply({"A": A, "B": B}, out_shape=(M, N))
    tiled = tile_and_run(op, {"A": A, "B": B}, out_shape=(M, N), tile_sizes={"i": 4, "j": 3})
    assert_same_result(untiled, tiled)
    np.testing.assert_allclose(untiled, A @ B, rtol=1e-10, atol=1e-12)


def test_tile_rejects_reduction_dim():
    """本参考实现的收窄:只允许对 parallel 维分块（见 tiling.py 文档）。"""
    import pytest

    from tiling import tile

    op = make_matmul()
    domain = op.iteration_domain(
        {"A": (10, 6), "B": (6, 7)}, out_shape=(10, 7)
    )
    with pytest.raises(ValueError):
        list(tile(op, domain, {"k": 2}))
