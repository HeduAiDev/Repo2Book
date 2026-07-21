"""t5 worked example：padding 的正确性条件是"补消费该 tile 的算子的幺元"
（[Linalg §3.2] paper.md:L346），不是随便补一个值。

用 conv 的归约维（输入通道 c）来演示：把一份只有部分有效通道的 "边界 tile"
补到满通道数，补对了幺元(0)结果与真值一致；补错了(1)结果就错——把"补错幺元
会算错"变成一个可运行的反例，而不是一句断言。
"""
import numpy as np
import pytest

from named_ops import make_conv_1d_nwc_wcf
from padding import neutral_element, pad_to_static


def test_neutral_element_table():
    assert neutral_element("sum") == 0.0
    assert neutral_element("prod") == 1.0
    assert neutral_element("max") == float("-inf")
    assert neutral_element("min") == float("inf")
    with pytest.raises(ValueError):
        neutral_element("no-such-combine")


def test_pad_to_static_rejects_shrinking():
    with pytest.raises(ValueError):
        pad_to_static(np.zeros((3, 3)), target_shape=(2, 3), neutral=0.0)


def _conv_setup_reduction_tiled(seed=0):
    rng = np.random.default_rng(seed)
    N, W, C, F, KW = 1, 10, 5, 3, 3
    out_w = W - KW + 1
    I = rng.standard_normal((N, W, C))
    K = rng.standard_normal((KW, C, F))
    return N, W, C, F, KW, out_w, I, K


def test_padding_with_correct_neutral_reproduces_full_reduction():
    """把归约维(输入通道 c,真实宽度 5)手工切成两段 `[0,3)` 与 `[3,5)`——模拟
    "归约也按 tile 分块、最后把各段部分和相加"这一常见模式(本参考实现的
    `tile()`/`tile_and_run` 出于范围收窄不通用支持它,见 tiling.py 文档；这里
    只手工搭一次场景来验证 padding 本身的正确性条件)。第二段只有 2 个真实通道,
    要补到静态尺寸 3 才能和第一段拼成同样大小的两次调用:
      - `I` 的补位必须是 sum 的幺元 0——对应"第 3 个通道根本不存在,不该贡献"；
      - `K` 的补位补什么都无所谓(这里故意补一个明显不是 0 的数 999),因为乘的
        是 `I` 侧已经归零的那个位置。
    两段部分和相加,必须等于用全部 5 个真实通道一次算出的参考结果。
    """
    op = make_conv_1d_nwc_wcf()
    N, W, C, F, KW, out_w, I, K = _conv_setup_reduction_tiled()
    reference = op.apply({"I": I, "K": K}, out_shape=(N, out_w, F))

    I_a, K_a = I[:, :, :3], K[:, :3, :]
    I_b, K_b = I[:, :, 3:], K[:, 3:, :]  # 只有 2 个真实通道(真实 C=5)

    neutral = neutral_element("sum")
    I_b_padded = pad_to_static(I_b, target_shape=(N, W, 3), neutral=neutral)
    K_b_padded = pad_to_static(K_b, target_shape=(KW, 3, F), neutral=999.0)

    partial_a = op.apply({"I": I_a, "K": K_a}, out_shape=(N, out_w, F))
    partial_b = op.apply({"I": I_b_padded, "K": K_b_padded}, out_shape=(N, out_w, F))
    combined = partial_a + partial_b
    np.testing.assert_allclose(combined, reference, rtol=1e-10, atol=1e-12)


def test_padding_with_wrong_neutral_breaks_result():
    """同样的场景,把 `I` 的补位幺元换成 1(而不是 sum 该用的 0),且 `K` 的补位是
    一个非零的"垃圾值"(现实中补位内存本就可能是任意残留内容,不能指望它恰好是
    0)——这下"不存在的第 3 个通道"不再是"不贡献",两段部分和相加会偏离真值。
    这是 dossier 明确要求的"补错幺元 -> 结果错"反例。
    """
    op = make_conv_1d_nwc_wcf()
    N, W, C, F, KW, out_w, I, K = _conv_setup_reduction_tiled(seed=1)
    reference = op.apply({"I": I, "K": K}, out_shape=(N, out_w, F))

    I_a, K_a = I[:, :, :3], K[:, :3, :]
    I_b, K_b = I[:, :, 3:], K[:, 3:, :]

    wrong_neutral = 1.0  # 对 sum 归约来说是错的幺元
    I_b_wrongly_padded = pad_to_static(I_b, target_shape=(N, W, 3), neutral=wrong_neutral)
    K_b_padded = pad_to_static(K_b, target_shape=(KW, 3, F), neutral=2.0)  # 非零"垃圾值"

    partial_a = op.apply({"I": I_a, "K": K_a}, out_shape=(N, out_w, F))
    partial_b = op.apply({"I": I_b_wrongly_padded, "K": K_b_padded}, out_shape=(N, out_w, F))
    combined = partial_a + partial_b
    assert not np.allclose(combined, reference)
