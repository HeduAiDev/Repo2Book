import numpy as np
import pytest

from csa_compression import (
    CompressorShape,
    csa_compress_sequence,
    csa_project_kv_entries,
    hca_compress_sequence,
    hca_project_kv_entries,
    overlap_transform,
    softmax_over_positions,
    weighted_pool,
)


def test_overlap_transform_shape_and_borrow_pattern():
    # n=6 tokens, m=2 -> 3 blocks; c=1 for easy manual tracing
    a_seq = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    b_seq = np.array([[10.0], [20.0], [30.0], [40.0], [50.0], [60.0]])
    windows = overlap_transform(a_seq, b_seq, m=2, pad_value=0.0)
    assert windows.shape == (3, 4, 1)
    # block 0: own a-values [1,2], borrowed b (padded) [0,0]
    np.testing.assert_allclose(windows[0, 2:, 0], [1.0, 2.0])
    np.testing.assert_allclose(windows[0, :2, 0], [0.0, 0.0])
    # block 1: own a-values [3,4], borrowed b from block 0's b-values [10,20]
    np.testing.assert_allclose(windows[1, 2:, 0], [3.0, 4.0])
    np.testing.assert_allclose(windows[1, :2, 0], [10.0, 20.0])
    # block 2: own a-values [5,6], borrowed b from block 1's b-values [30,40]
    np.testing.assert_allclose(windows[2, 2:, 0], [5.0, 6.0])
    np.testing.assert_allclose(windows[2, :2, 0], [30.0, 40.0])


def test_overlap_transform_rejects_non_divisible_length():
    a_seq = np.zeros((5, 1))
    b_seq = np.zeros((5, 1))
    with pytest.raises(ValueError):
        overlap_transform(a_seq, b_seq, m=2)


def test_softmax_over_positions_sums_to_one():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(4, 6, 3))
    S = softmax_over_positions(Z)
    np.testing.assert_allclose(S.sum(axis=1), np.ones((4, 3)), atol=1e-8)


def test_softmax_over_positions_handles_neg_inf_padding():
    # first block's lower half is -inf (i=0 padding) -> those weights should be exactly 0
    Z = np.array([[[1.0], [2.0], [-np.inf], [-np.inf]]])  # (1 block, 4 positions, c=1)
    S = softmax_over_positions(Z)
    assert not np.any(np.isnan(S))
    np.testing.assert_allclose(S[0, 2:, 0], [0.0, 0.0])
    assert np.isclose(S.sum(), 1.0)


def test_weighted_pool_matches_manual_computation():
    S = np.array([[[0.25], [0.75]]])   # (1, 2, 1)
    C = np.array([[[10.0], [20.0]]])   # (1, 2, 1)
    out = weighted_pool(S, C)
    np.testing.assert_allclose(out, [[0.25 * 10.0 + 0.75 * 20.0]])


def _random_csa_weights(d, c, m, seed=0):
    rng = np.random.default_rng(seed)
    W_a_kv = rng.normal(scale=0.1, size=(d, c))
    W_b_kv = rng.normal(scale=0.1, size=(d, c))
    W_a_z = rng.normal(scale=0.1, size=(d, c))
    W_b_z = rng.normal(scale=0.1, size=(d, c))
    B_a = rng.normal(scale=0.1, size=(m, c))
    B_b = rng.normal(scale=0.1, size=(m, c))
    return W_a_kv, W_b_kv, W_a_z, W_b_z, B_a, B_b


def test_csa_compress_sequence_output_shape():
    n, d, c, m = 8, 5, 4, 4
    rng = np.random.default_rng(1)
    H = rng.normal(size=(n, d))
    W_a_kv, W_b_kv, W_a_z, W_b_z, B_a, B_b = _random_csa_weights(d, c, m)
    out = csa_compress_sequence(H, W_a_kv, W_b_kv, W_a_z, W_b_z, B_a, B_b, m)
    assert out.shape == (n // m, c)


def test_csa_compress_sequence_soft_argmax_limit():
    """当某个位置的 Z 值远大于同窗口其余位置时,压缩输出应趋近该位置的 C 值
    (softmax 权重趋近 one-hot,这是核对 Eq.11-12 数学含义的最直接方式)。"""
    n, d, c, m = 4, 3, 2, 2
    H = np.random.default_rng(2).normal(size=(n, d))
    W_a_kv = np.random.default_rng(3).normal(scale=0.1, size=(d, c))
    W_b_kv = np.random.default_rng(4).normal(scale=0.1, size=(d, c))
    W_a_z = np.zeros((d, c))
    W_b_z = np.zeros((d, c))
    B_a = np.array([[0.0, 0.0], [50.0, 50.0]])   # 第 2 个 own-position 的 bias 远大于其余
    B_b = np.array([[-50.0, -50.0], [-50.0, -50.0]])
    out = csa_compress_sequence(H, W_a_kv, W_b_kv, W_a_z, W_b_z, B_a, B_b, m)
    C_a, _, _, _ = csa_project_kv_entries(H, W_a_kv, W_b_kv, W_a_z, W_b_z)
    # block 1 (第二块) 的 own a-values 是 H[2:4] 投影结果, 其中第二个 own-position(全局 index 3)
    # 拿到极大 bias,应该主导 block 1 的输出
    np.testing.assert_allclose(out[1], C_a[3], atol=1e-3)


def _random_hca_weights(d, c, m_prime, seed=0):
    rng = np.random.default_rng(seed)
    W_kv = rng.normal(scale=0.1, size=(d, c))
    W_z = rng.normal(scale=0.1, size=(d, c))
    B = rng.normal(scale=0.1, size=(m_prime, c))
    return W_kv, W_z, B


def test_hca_compress_sequence_output_shape():
    n, d, c, m_prime = 16, 5, 4, 4  # 用小 m' 代替论文的 128,便于测试
    rng = np.random.default_rng(5)
    H = rng.normal(size=(n, d))
    W_kv, W_z, B = _random_hca_weights(d, c, m_prime)
    out = hca_compress_sequence(H, W_kv, W_z, B, m_prime)
    assert out.shape == (n // m_prime, c)


def test_hca_compress_sequence_soft_argmax_limit():
    n, d, c, m_prime = 4, 3, 2, 4
    H = np.random.default_rng(6).normal(size=(n, d))
    W_kv = np.random.default_rng(7).normal(scale=0.1, size=(d, c))
    W_z = np.zeros((d, c))
    B = np.zeros((m_prime, c))
    B[1] = 50.0   # 第二个位置的 bias 远大
    out = hca_compress_sequence(H, W_kv, W_z, B, m_prime)
    C, _ = hca_project_kv_entries(H, W_kv, W_z)
    np.testing.assert_allclose(out[0], C[1], atol=1e-3)


def test_hca_compress_sequence_rejects_non_divisible_length():
    H = np.zeros((5, 3))
    W_kv, W_z, B = _random_hca_weights(3, 2, 4)
    with pytest.raises(ValueError):
        hca_compress_sequence(H, W_kv, W_z, B, m_prime=4)


def test_compressor_shape_csa():
    shp = CompressorShape(compress_ratio=4)
    assert shp.overlap is True
    assert shp.coff == 2


def test_compressor_shape_hca():
    shp = CompressorShape(compress_ratio=128)
    assert shp.overlap is False
    assert shp.coff == 1


def test_compressor_shape_rejects_unsupported_ratio():
    with pytest.raises(ValueError):
        CompressorShape(compress_ratio=16)
