"""§8.3 worked example：向量化情形 (1) 逐点、(4) 归约（[Linalg §3.3]
paper.md:L353-L367）——覆盖 dossier 要求的两种情形，其余三种显式拒绝而非假装支持。
"""
import numpy as np
import pytest

from named_ops import make_conv_1d_nwc_wcf, make_matmul, make_pointwise_add
from vectorization import build_einsum_subscripts, vectorize


def test_case1_pointwise_vectorize_matches_apply():
    op = make_pointwise_add()
    rng = np.random.default_rng(6)
    A = rng.standard_normal((4, 5))
    B = rng.standard_normal((4, 5))
    scalar_result = op.apply({"A": A, "B": B}, out_shape=(4, 5))
    vector_result = vectorize(op, {"A": A, "B": B}, out_shape=(4, 5))
    np.testing.assert_allclose(scalar_result, vector_result, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(vector_result, A + B, rtol=1e-10, atol=1e-12)


def test_case4_reduction_vectorize_matches_apply_via_einsum():
    op = make_matmul()
    rng = np.random.default_rng(7)
    A = rng.standard_normal((6, 4))
    B = rng.standard_normal((4, 5))
    scalar_result = op.apply({"A": A, "B": B}, out_shape=(6, 5))
    vector_result = vectorize(op, {"A": A, "B": B}, out_shape=(6, 5))
    np.testing.assert_allclose(scalar_result, vector_result, rtol=1e-10, atol=1e-12)
    # 下标字母按迭代维在 dim_names 里的下标位置分配(i=0->'a', j=1->'b', k=2->'c'),
    # 不是按维名字面拼——语义等价于 "ik,kj->ij",这里核对实际产出的字母串。
    assert build_einsum_subscripts(op) == "ac,cb->ab"


def test_vectorize_rejects_sliding_window_case5():
    """conv 的 I 索引是 w+kw(不是纯置换)——情形 (5) 滑窗,本参考实现不实现,
    必须显式报错而不是给出错误数值。"""
    op = make_conv_1d_nwc_wcf()
    rng = np.random.default_rng(8)
    N, W, C, F, KW = 1, 8, 2, 3, 3
    out_w = W - KW + 1
    I = rng.standard_normal((N, W, C))
    K = rng.standard_normal((KW, C, F))
    with pytest.raises(NotImplementedError):
        vectorize(op, {"I": I, "K": K}, out_shape=(N, out_w, F))


def test_vectorize_rejects_reduction_without_sum_of_products_marker():
    """把 matmul 的 vectorizable_reduce 标记去掉,即便算子体形式上还是乘加,
    向量化路径也应拒绝——本参考实现故意不去反射 body 是不是乘加,只信标记
    (对应论文"视对算子体的进一步分析"这句留白,不代自动分析)。"""
    import dataclasses

    op = make_matmul()
    unmarked = dataclasses.replace(op, vectorizable_reduce=None)
    rng = np.random.default_rng(9)
    A = rng.standard_normal((3, 2))
    B = rng.standard_normal((2, 4))
    with pytest.raises(NotImplementedError):
        vectorize(unmarked, {"A": A, "B": B}, out_shape=(3, 4))
