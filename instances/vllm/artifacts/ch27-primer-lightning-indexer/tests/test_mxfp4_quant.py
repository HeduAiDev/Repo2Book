import numpy as np

from mxfp4_quant import (
    MXFP4_BLOCK_SIZE,
    dequantize_mxfp4,
    quantize_mxfp4,
    quantize_scores_bf16,
    topk_recall,
)


def test_block_size_matches_paper_constant():
    assert MXFP4_BLOCK_SIZE == 32


def test_quantize_dequantize_roundtrip_within_4bit_error():
    # PAPER: §5.2.1 —— MXFP4 每 block 16 个可表示电平，误差应远粗于 FP8/int8，
    # 但仍应有界（由 block 内 max(|x|)/7 的 scale 控制）。
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, size=(2, 32)).astype(np.float32)
    q, scale = quantize_mxfp4(x, block_size=32)
    deq = dequantize_mxfp4(q, scale)
    assert deq.shape == x.shape
    # 最大量化误差 <= scale/2 每元素，scale <= max(|x|)/7 <= 1/7
    np.testing.assert_allclose(deq, x, atol=1.0 / 7.0)


def test_quantize_levels_are_4bit_bounded():
    x = np.array([[10.0, -10.0] + [0.0] * 30], dtype=np.float32)
    q, _ = quantize_mxfp4(x, block_size=32)
    assert q.min() >= -7 and q.max() <= 7


def test_quantize_scores_bf16_truncates_mantissa_not_exponent_range():
    # PAPER: §5.2.1 —— BF16 截断尾数、保留 FP32 的指数范围（大/小数量级不被压垮）。
    x = np.array([1.0 + 2**-20, 1e30, -1e-30], dtype=np.float32)
    y = quantize_scores_bf16(x)
    assert y.dtype == np.float32
    # 相对误差应在 BF16 尾数精度量级（7 位尾数 -> 约 2^-7 相对误差上界）内
    rel_err = np.abs((y - x) / x)
    assert np.all(rel_err < 2**-6)
    # 尾数确被截断：非零值与原值不再逐位相等（除非低 16 位恰好本就是 0）
    assert (y[0] != x[0]) or (x[0].view(np.uint32) & np.uint32(0xFFFF) == 0)
    # 指数范围保留：数量级未被压到 0 或 inf
    assert np.isfinite(y).all()
    assert y[1] != 0.0 and y[2] != 0.0


def test_topk_recall_is_one_when_scores_identical():
    scores = np.array([[5.0, 1.0, 9.0, 3.0]])
    recall = topk_recall(scores, scores, k=2)
    assert recall == 1.0


def test_topk_recall_drops_when_quantization_flips_order():
    # 构造一个扰动，使量化后的分数把原本不在 top-k 的条目挤进来。
    scores_ref = np.array([[10.0, 9.9, 1.0, 0.0]])
    scores_quant = np.array([[9.0, 9.9, 9.95, 0.0]])  # 索引 2 被抬高，挤掉索引 0
    recall = topk_recall(scores_ref, scores_quant, k=2)
    assert recall < 1.0
