"""边界与前置条件 —— 补齐 tester 契约的「边界/异常」层,并固化对账时发现的
等价前置条件:

1. arXiv:2210.17323 §3(量化网格在过程开始前固定)的 Algorithm 1 本体 =
   固定网格:Cholesky+lazy batch 与「每列一次 Eq.2+Eq.3」对**任意**分块大小
   逐位等价(§4 Step 2/3 只改执行方式不改数学,跨多种 block_size 验证,
   比单一配置的等价测试更强)。
2. §5 Additional Tricks 的 grouping 是网格扩展:组跨块边界时,lazy batch 推迟
   的更新会让组网格看到「截至上一块末」而非「截至上一列」的权重——与
   IST-DASLab/gptq 参考实现(B=128、group_size=128,组永远不跨块)语义一致。
   本测试固化可保证等价的前置条件:B % group_size == 0。
3. 各数值护栏与异常路径:常数向量/常数行的 scale 下限、奇异 H 的
   LinAlgError、awq_pack 的参数校验。"""
import numpy as np
import pytest

from awq import awq_pack, awq_unpack
from gptq import (
    dampen_hessian,
    gptq_naive_inverse_updates,
    gptq_quantize,
    layer_hessian,
    obq_quantize_row,
    rtn_quantize,
)
from uniform_quant import dequantize_asymmetric, quantize_asymmetric


def _toy_xy(seed, d_row, d_col, n_samples=40, corr=0.95):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((d_row, d_col))
    base = rng.standard_normal((n_samples, 1))
    noise = rng.standard_normal((n_samples, d_col))
    x = corr * base + np.sqrt(1 - corr**2) * noise
    return w, x


def test_fixed_grid_algorithm1_equivalent_for_any_block_size():
    # §3:网格固定在过程开始前 -> 论文 Algorithm 1 本体与「每列一次 Eq.2+Eq.3」
    # (§4 Step 1 的中间形态)在任意 lazy-batch 分块下逐位等价。
    for seed in (0, 1):
        for block in (1, 3, 5, 7, 128):
            w, x = _toy_xy(seed + 1, 8, 12)
            q_alg1, _, _ = gptq_quantize(
                w, x, num_bits=4, block_size=block
            )
            q_naive, _ = gptq_naive_inverse_updates(w, x, num_bits=4)
            np.testing.assert_array_equal(
                q_alg1, q_naive, err_msg=f"seed={seed} block={block}"
            )


def test_grouped_equivalence_holds_when_groups_align_with_blocks():
    # §5 Additional Tricks(grouping)+ 参考实现 B=128/g=128 的约定:组不跨块
    # (B % group_size == 0)时,lazy batch 与逐列更新仍逐位等价。
    for bs, gs in ((8, 4), (12, 6), (16, 8)):
        w, x = _toy_xy(bs + gs, 6, 24)
        q_blk, _, _ = gptq_quantize(
            w, x, num_bits=4, block_size=bs, group_size=gs
        )
        q_col, _ = gptq_naive_inverse_updates(
            w, x, num_bits=4, group_size=gs
        )
        np.testing.assert_array_equal(q_blk, q_col)


def test_obq_rank_deficient_hessian_and_dampening_remedy():
    # §3 的 OBQ 无 dampening(那是 §4 Step 3 才加的):校准样本数 < 特征数时
    # H = 2X^T X 秩亏(小特征值 ~0);numpy 的 inv 对此**不报警**、静默给出
    # ~1e15 量级的垃圾逆 —— §4 Step 3 所述「H_F^{-1} 数值问题导致补偿方向
    # 错误」的极端版。docstring 记录的补救:先 dampen_hessian 再跑。
    rng = np.random.default_rng(11)
    x = rng.standard_normal((3, 8))  # 3 样本 < 8 特征 -> rank(H) <= 3
    h = layer_hessian(x)
    eig = np.linalg.eigvalsh(h)
    assert np.sum(np.abs(eig) < 1e-9) == 5  # 5 个 ~0 特征值:精确算术下奇异
    hinv = np.linalg.inv(h)  # 不 raise(仅精确零主元才 raise)
    assert np.abs(hinv).max() > 1e12  # 静默垃圾逆
    # 补救(§4 Step 3:dampening 1% 平均对角元)后 H 正定、OBQ 正常工作。
    hd = dampen_hessian(h, 0.01)
    assert np.all(np.linalg.eigvalsh(hd) > 0)
    w = rng.standard_normal(8)
    q, w_hat = obq_quantize_row(w, hd, num_bits=3)
    assert np.all(np.isfinite(w_hat))


def test_asymmetric_constant_vector_hits_scale_floor_and_round_trips():
    # quantize_asymmetric 的数值护栏:xmax == xmin 时 scale 下限 1e-12 兜底,
    # 不产生 NaN/除零;常数向量量化-反量化回到自身。
    x = np.full(6, 0.37)
    q, scale, zp = quantize_asymmetric(x, num_bits=4)
    assert scale == pytest.approx(1e-12)
    assert np.all(np.isfinite(q.astype(float)))
    np.testing.assert_allclose(dequantize_asymmetric(q, scale, zp), x, atol=1e-9)


def test_rtn_constant_rows_do_not_crash():
    # row_grid_params 同款护栏:常数行的 min-max 网格 scale 触底,RTN 全程有限。
    w = np.full((3, 5), -1.4)
    q, w_hat = rtn_quantize(w, num_bits=4)
    assert np.all(np.isfinite(w_hat))
    np.testing.assert_allclose(w_hat, w, atol=1e-9)


def test_awq_pack_rejects_bad_length_and_bit_width():
    # §4.2 只定义 4-bit 打包(interleave [0,2,4,6,1,3,5,7]);长度须为 8 的倍数。
    with pytest.raises(ValueError):
        awq_pack(np.zeros(7, dtype=np.int64), num_bits=4)
    with pytest.raises(ValueError):
        awq_pack(np.zeros(8, dtype=np.int64), num_bits=8)
    with pytest.raises(ValueError):
        awq_unpack(np.zeros(1, dtype=np.int32), num_bits=8)
