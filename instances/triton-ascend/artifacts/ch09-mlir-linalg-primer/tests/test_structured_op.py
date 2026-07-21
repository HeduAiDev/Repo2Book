"""t1/t2 worked example：索引表达式 + 隐式迭代域（[Linalg §3]，paper.md:L248-L278）。"""
import numpy as np
import pytest

from named_ops import make_conv_1d_nwc_wcf


def test_iteration_domain_matches_paper_shapes():
    """论文自己的例子（paper.md:L260-L276）：O:1x988x64, I:1x990x32, K:3x32x64。
    只做形状层面的域推导（不跑完整卷积——那是 O(1e7) 级的纯 Python 循环，对一份
    primer 参考实现没必要），验证 derive_iteration_domain 精确复现论文给出的
    5 条边界：0<=n<O.0, 0<=w<O.1, 0<=f<O.2, 0<=kw<K.0, 0<=c<K.1。
    """
    op = make_conv_1d_nwc_wcf()
    ins_shapes = {"I": (1, 990, 32), "K": (3, 32, 64)}
    out_shape = (1, 988, 64)
    domain = op.iteration_domain(ins_shapes, out_shape)
    n, w, f, kw, c = (op.dim_names.index(x) for x in ("n", "w", "f", "kw", "c"))
    assert domain[n] == (0, 1)
    assert domain[w] == (0, 988)
    assert domain[f] == (0, 64)
    assert domain[kw] == (0, 3)
    assert domain[c] == (0, 32)


def test_conv_shape_relation_988_equals_990_minus_3_plus_1():
    """paper.md 的具体数字之间必须满足滑窗卷积的形状关系：valid 卷积
    out_w = in_w - kernel_w + 1。这条不是本参考实现发明的,是论文给的具体例子
    自洽性的一个可检验推论。"""
    assert 988 == 990 - 3 + 1


def test_apply_matches_hand_written_reference_small():
    """小参数（dossier 建议：N=1,W=16,C=2,F=3,KW=3）下，`StructuredOp.apply`
    的结果必须与手写的三重 for 循环参考实现逐元素相同。"""
    op = make_conv_1d_nwc_wcf()
    rng = np.random.default_rng(0)
    N, W, C, F, KW = 1, 16, 2, 3, 3
    out_w = W - KW + 1
    I = rng.standard_normal((N, W, C))
    K = rng.standard_normal((KW, C, F))

    expected = np.zeros((N, out_w, F))
    for n in range(N):
        for w in range(out_w):
            for f in range(F):
                acc = 0.0
                for kw in range(KW):
                    for c in range(C):
                        acc += I[n, w + kw, c] * K[kw, c, f]
                expected[n, w, f] = acc

    got = op.apply({"I": I, "K": K}, out_shape=(N, out_w, F))
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


def test_derive_iteration_domain_raises_without_pure_identity_source():
    """如果没有任何操作数给出某个维的纯恒等映射来源，必须显式报错——而不是悄悄
    猜一个边界（本参考实现的推导是受限的，见 structured_op.derive_iteration_domain
    的文档）。"""
    op = make_conv_1d_nwc_wcf()
    # 只给 I,不给 out_shape 来源里对 f 维的纯恒等映射(此处刻意漏传 K 的形状信息
    # 是不可能的,因为 iteration_domain 强制要求 ins 覆盖所有 operand_names)—
    # 改为直接测试底层函数:给一个没有任何来源能覆盖某维的场景。
    from structured_op import AffineExpr, derive_iteration_domain

    bad_map = (AffineExpr(((0, 1), (1, 1))),)  # 维 0 和维 1 耦合,没人是纯恒等
    with pytest.raises(ValueError):
        derive_iteration_domain(("d0", "d1"), [(bad_map, (10,))])
