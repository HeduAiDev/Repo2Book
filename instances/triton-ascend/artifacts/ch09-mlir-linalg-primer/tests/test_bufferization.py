"""m21/t7 worked example：bufferization 少分配少拷贝([Linalg §3.4]
paper.md:L369-L373),destination-passing style 授权原地写而不改变函数式语义
([Linalg §3.4] paper.md:L315-L325)。
"""
import numpy as np

from bufferization import bufferize_dps, bufferize_naive
from named_ops import make_conv_1d_nwc_wcf
from verify import assert_same_result


def test_naive_and_dps_agree_numerically_but_allocate_differently():
    op = make_conv_1d_nwc_wcf()
    rng = np.random.default_rng(10)
    N, W, C, F, KW = 1, 20, 2, 3, 3
    out_w = W - KW + 1  # 18, tile_w=5 -> 4 块(含 1 个边界块)
    I = rng.standard_normal((N, W, C))
    K = rng.standard_normal((KW, C, F))

    naive = bufferize_naive(op, {"I": I, "K": K}, out_shape=(N, out_w, F), tile_sizes={"w": 5})
    dps = bufferize_dps(op, {"I": I, "K": K}, out_shape=(N, out_w, F), tile_sizes={"w": 5})

    assert_same_result(naive.output, dps.output)

    reference = op.apply({"I": I, "K": K}, out_shape=(N, out_w, F))
    assert_same_result(naive.output, reference)

    # 4 块 tile:naive 每块都新分配一次 + 初始 1 次 = 5 次;DPS 只分配 1 次。
    assert naive.OUT_OF_PLACE_ALLOC_COUNT == 5
    assert dps.DPS_ALLOC_COUNT == 1
    assert dps.DPS_ALLOC_COUNT < naive.OUT_OF_PLACE_ALLOC_COUNT


def test_naive_report_marks_dps_count_as_zero_and_vice_versa():
    """两份报告里"没跑的那条策略"计数应为 0,不是碰巧漏填。"""
    op = make_conv_1d_nwc_wcf()
    rng = np.random.default_rng(11)
    N, W, C, F, KW = 1, 12, 2, 3, 3
    out_w = W - KW + 1
    I = rng.standard_normal((N, W, C))
    K = rng.standard_normal((KW, C, F))

    naive = bufferize_naive(op, {"I": I, "K": K}, out_shape=(N, out_w, F), tile_sizes={"w": 4})
    dps = bufferize_dps(op, {"I": I, "K": K}, out_shape=(N, out_w, F), tile_sizes={"w": 4})
    assert naive.DPS_ALLOC_COUNT == 0
    assert dps.OUT_OF_PLACE_ALLOC_COUNT == 0
