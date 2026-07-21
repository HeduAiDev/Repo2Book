"""m19 worked example：named op 只是 generic op 的声明式配置
（[Linalg §3.3, 脚注 6][§5.1] paper.md:L296-L313）——展开前后数值必须相同。
"""
import numpy as np

from named_ops import make_conv_1d_nwc_wcf, make_matmul, named_op_registry, to_generic


def test_registry_has_the_two_paper_named_ops():
    assert set(named_op_registry.keys()) == {"conv_1d_nwc_wcf", "matmul"}
    conv = named_op_registry["conv_1d_nwc_wcf"]()
    mm = named_op_registry["matmul"]()
    assert conv.is_named is True
    assert mm.is_named is True


def test_to_generic_does_not_mutate_original_and_preserves_semantics():
    op = make_conv_1d_nwc_wcf()
    assert op.is_named is True
    generic = to_generic(op)
    assert generic.is_named is False
    assert op.is_named is True  # 原对象不受影响

    rng = np.random.default_rng(4)
    N, W, C, F, KW = 1, 12, 2, 3, 3
    out_w = W - KW + 1
    I = rng.standard_normal((N, W, C))
    K = rng.standard_normal((KW, C, F))

    named_result = op.apply({"I": I, "K": K}, out_shape=(N, out_w, F))
    generic_result = generic.apply({"I": I, "K": K}, out_shape=(N, out_w, F))
    np.testing.assert_allclose(named_result, generic_result, rtol=1e-10, atol=1e-12)
    # 索引映射/迭代器类型/算子体三者必须是同一份,不是"重新配置了一遍恰好相等"
    assert generic.operand_maps == op.operand_maps
    assert generic.result_map == op.result_map
    assert generic.iterator_types == op.iterator_types
    assert generic.body is op.body


def test_matmul_named_and_generic_agree():
    op = make_matmul()
    generic = to_generic(op)
    rng = np.random.default_rng(5)
    A = rng.standard_normal((4, 3))
    B = rng.standard_normal((3, 5))
    named_result = op.apply({"A": A, "B": B}, out_shape=(4, 5))
    generic_result = generic.apply({"A": A, "B": B}, out_shape=(4, 5))
    np.testing.assert_allclose(named_result, generic_result, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(named_result, A @ B, rtol=1e-10, atol=1e-12)
